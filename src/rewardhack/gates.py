"""Checks made on a shell command before it runs: one gate for commands that spend money, one for git.

Both are driven by `entrypoints/gate.py hook`, a Claude Code PreToolUse hook on Bash. They read the command,
never run it, and return the reasons to refuse it. An empty list lets the command through.

Spend gate. A pod job may start only from a launch record (`entrypoints/gate.py ready`), which is written
after the tree is clean, the time cap has a stated basis at the project's multiple, the budget check passes
and the test suite passes. The record names the commit, the ledger, the remote script and the deadline; a
driver command that differs from it, or a later commit or ledger row, is refused. A remote job script
started over SSH by any route but `remote_job.sh` with `REQUIRE_AUTOSTOP=1` is refused. Removing a pod is
never refused. A metered API judge needs a spend cap and real prices on the command line.

Commit gate. What `git add`, `git commit` or `git push` would publish is read for the owner's withheld
terms (kept in an untracked file, so the repo does not name them), for secrets, and for binary or very
large files.
"""
import hashlib
import json
import os
import re
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from .budget import check, read_ledger

RECORD = Path('.tools') / 'launch-ready.json'
LEDGER = Path('spending') / 'ledger.csv'
TERMS = Path('.claude') / 'commit-gate-terms.txt'
# What the pod runs is the working tree under these paths (ops/drive_runpod.sh copies it), so they must match the tested commit.
SHIPPED = ('src', 'ops', 'entrypoints', 'tests', 'pyproject.toml', 'requirements.txt')
# Minutes a pod can bill outside the job's own deadline: the 15 minute provisional timer the driver arms
# before the job starts, and the 10 minute grace after the deadline before the pod removes itself.
OVERHEAD_MINUTES = 25
DRIVER_JUDGE_MAX_USD = 1.0  # what ops/drive_runpod.sh passes to the pod when JUDGE_MAX_USD is not given
MULTIPLES = {'measured': 2.0, 'estimated': 3.0}  # CLAUDE.md, "Sizing time caps and approvals"
UNIT_MINUTES = {'s': 1/60, 'min': 1.0, 'h': 60.0}
BASIS = re.compile(r'\s*(\d+(?:\.\d+)?)\s*x\s+(measured|estimated)\s+(\d+(?:\.\d+)?)\s*(s|min|h)\s*')
READY = '.tools/venv/bin/python entrypoints/gate.py ready --remote ops/<script>.sh --deadline <minutes> ' \
        "--basis '<2x measured|3x estimated> <time> <s|min|h>' --line <budget line> --hourly <usd> --other-usd <judge cap and other costs>"

MAX_BYTES = 15_000_000  # the largest result file tracked on purpose is 11 MB of rows; model files start above this
SECRET_NAMES = re.compile(r'(^|/)(\.env(\..+)?|\.netrc|id_rsa.*|id_ed25519.*|.*\.pem|.*\.key|.*_key)$')
SECRETS = [
    ('an OpenAI-style key', re.compile(r'\bsk-[A-Za-z0-9_-]{20,}')),
    ('a RunPod key', re.compile(r'\brpa_[A-Za-z0-9]{20,}')),
    ('a Hugging Face token', re.compile(r'\bhf_[A-Za-z0-9]{30,}')),
    ('a GitHub token', re.compile(r'\bgh[pousr]_[A-Za-z0-9]{30,}')),
    ('an AWS access key id', re.compile(r'\bAKIA[0-9A-Z]{16}\b')),
    ('a private key', re.compile(r'-----BEGIN [A-Z ]*PRIVATE KEY-----')),
]

# Every shape either gate acts on. A command without one of these is passed without being read further.
GATED = re.compile(r'\bgit\s+(?:-\S+\s+(?:[^-\s]\S*\s+)?)*(add|commit|push)\b|drive_run\w*\.sh|launch_aws\.sh|remote_\w+\.sh'
                   r'|runpodctl\s+(pod\s+)?(create|deploy)|--judge[ =]api')
HEREDOC = re.compile(r'''<<-?\s*(['"]?)(\w+)\1[^\n]*\n(.*?)\n[ \t]*\2[ \t]*(?=\n|\)|$)''', re.S)
ASSIGNMENT = re.compile(r'([A-Za-z_]\w*)=(.*)', re.S)
WRAPPERS = {'bash', 'sh', 'zsh', 'nohup', 'exec', 'time', 'command', 'env', 'setsid', 'sudo'}
SSH_WITH_VALUE = set('pioFJLRDEbcelmOQSWw')


def git(root, *args, text=True):
    return subprocess.run(['git', '-C', str(root), '-c', 'core.quotepath=off', *args], check=True, capture_output=True, text=text).stdout


# --- reading a command ---------------------------------------------------------------------------------------------

def simple_commands(command):
    """The command as a list of simple commands (token lists), and the text of its here-documents.

    Raises ValueError when the quoting cannot be read.
    """
    documents = [match.group(3) for match in HEREDOC.finditer(command)]
    lexer = shlex.shlex(HEREDOC.sub(' ', command).replace('\n', ' ; '), posix=True, punctuation_chars=True)
    lexer.whitespace_split = True
    groups, current = [], []
    for token in lexer:
        if token and not token.strip(';&|()<>'):
            groups.append(current)
            current = []
        else:
            current.append(token)
    return [group for group in groups + [current] if group], documents


def command_words(tokens):
    """Leading `NAME=value` settings as a dict, and the words from the command name on, wrappers such as `nohup bash` dropped.

    `bash -c '<script>'` comes back as (settings, ['bash -c', script]).
    """
    settings, index = {}, 0
    while index < len(tokens):
        token = tokens[index]
        assignment = ASSIGNMENT.fullmatch(token)
        if assignment:
            settings[assignment.group(1)] = assignment.group(2)
        elif token == 'timeout':
            while index+1 < len(tokens) and tokens[index+1].startswith('-'):
                index += 1
            index += 1  # the duration
        elif token in ('bash', 'sh', 'zsh') and index+2 < len(tokens) and re.fullmatch(r'-[a-z]*c', tokens[index+1]):
            return settings, ['bash -c', tokens[index+2]]
        elif token not in WRAPPERS:
            break
        index += 1
    return settings, tokens[index:]


def ssh_remote_command(words):
    """The command an `ssh` invocation runs on the far side, as text ('' for an interactive login)."""
    index = 1
    while index < len(words) and words[index].startswith('-'):
        index += 2 if words[index][-1] in SSH_WITH_VALUE and len(words[index]) == 2 else 1
    return ' '.join(words[index+1:])


def flag_value(words, name):
    """The value given to `--name value` or `--name=value`, or None when the flag is absent."""
    for index, word in enumerate(words):
        if word == name and index+1 < len(words):
            return words[index+1]
        if word.startswith(name + '='):
            return word[len(name)+1:]
    return None


# --- the launch record ---------------------------------------------------------------------------------------------

def basis_minutes(basis):
    """Minutes of cap that a basis such as '2x measured 30 min' justifies. Raises ValueError on any other form or a low multiple."""
    match = BASIS.fullmatch(basis)
    if not match:
        raise ValueError(f"the cap basis must read '<multiple>x measured|estimated <time> <s|min|h>', not {basis!r}")
    multiple, kind, amount, unit = float(match.group(1)), match.group(2), float(match.group(3)), match.group(4)
    if multiple < MULTIPLES[kind]:
        raise ValueError(f'a cap on a {kind} time is at least {MULTIPLES[kind]:g}x it; {basis.strip()!r} is {multiple:g}x')
    return multiple * amount * UNIT_MINUTES[unit]


def dirty_paths(root):
    return [line for line in git(root, 'status', '--porcelain', '--untracked-files=all', '--', *SHIPPED).splitlines() if line.strip()]


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ready(root, remote, deadline, basis, line, hourly, other_usd, launch_cap, test_command):
    """Make every check a pod launch depends on and write the launch record. Returns (record, problems); no record is written with problems."""
    root = Path(root)
    problems = []
    if not re.fullmatch(r'ops/[A-Za-z0-9_-]+\.sh', remote) or not (root/remote).is_file():
        problems.append(f'{remote} is not an existing ops/*.sh script')
    dirty = dirty_paths(root)
    if dirty:
        problems.append('uncommitted changes in what the pod would run, so it is not the tested commit:\n    ' + '\n    '.join(dirty[:12]))
    try:
        needed = basis_minutes(basis)
        if deadline < needed:
            problems.append(f'the deadline of {deadline} min is under the {needed:g} min that {basis.strip()!r} calls for')
    except ValueError as error:
        problems.append(str(error))
    hours = (deadline + OVERHEAD_MINUTES) / 60
    worst, refused = check(read_ledger(root/LEDGER), line, hourly, hours, other_usd, launch_cap)
    problems += [f'budget: {reason}' for reason in refused]
    if problems:
        return None, problems
    tests = subprocess.run(test_command, cwd=root, capture_output=True, text=True)
    if tests.returncode != 0:
        tail = '\n    '.join((tests.stdout + tests.stderr).strip().splitlines()[-12:])
        return None, [f"the tests do not pass ({' '.join(test_command)} exited {tests.returncode}):\n    {tail}"]
    record = dict(head=git(root, 'rev-parse', 'HEAD').strip(), ledger_sha256=file_sha256(root/LEDGER), remote=remote,
                  deadline_minutes=deadline, basis=basis.strip(), line=line, hourly_usd=hourly, other_usd=other_usd, worst_case_usd=worst,
                  worst_case_minutes=deadline + OVERHEAD_MINUTES, written_at=datetime.now(timezone.utc).isoformat(timespec='seconds'))
    (root/RECORD).parent.mkdir(parents=True, exist_ok=True)
    (root/RECORD).write_text(json.dumps(record, indent=1) + '\n')
    return record, []


def driver_judge_cap(settings):
    """Dollars the pod's API judge may spend under a driver command with these settings, or None when no judge key is sent."""
    if 'FORWARD_JUDGE_KEY' not in settings or settings['FORWARD_JUDGE_KEY'] != '1':
        return None
    if 'JUDGE_MAX_USD' not in settings:
        return DRIVER_JUDGE_MAX_USD
    return float(settings['JUDGE_MAX_USD'])


def launch_problems(root, remote=None, deadline=None, judge_cap=None):
    """Why the launch record does not cover a launch now, optionally of this remote script at this deadline with this judge cap."""
    root = Path(root)
    if not (root/RECORD).exists():
        return [f'there is no launch record. Settle the launch first:\n    {READY}']
    record = json.loads((root/RECORD).read_text())
    problems = []
    if record['head'] != git(root, 'rev-parse', 'HEAD').strip():
        problems.append(f"the launch record is for commit {record['head'][:7]}, and HEAD has moved since the tests ran")
    dirty = dirty_paths(root)
    if dirty:
        problems.append('uncommitted changes in what the pod would run:\n    ' + '\n    '.join(dirty[:12]))
    if record['ledger_sha256'] != file_sha256(root/LEDGER):
        problems.append('the ledger has changed since the budget check')
    if remote is not None and record['remote'] != remote:
        problems.append(f"the launch record is for {record['remote']}, not {remote}")
    if deadline is not None and record['deadline_minutes'] != deadline:
        problems.append(f"the launch record is for a {record['deadline_minutes']} min deadline, not {deadline}")
    if judge_cap is not None and judge_cap > record['other_usd']:
        problems.append(f"this command lets the API judge spend ${judge_cap:.2f}, and the budget check counted ${record['other_usd']:.2f} "
                        'beside the pod (--other-usd)')
    return problems + ([f'Settle the launch again before starting it:\n    {READY}'] if problems else [])


# --- spend gate ----------------------------------------------------------------------------------------------------

def judge_problems(words):
    """A metered judge needs its cap and prices on the command line; a judge served from this machine is not metered."""
    if flag_value(words, '--judge') != 'api':
        return []
    url = flag_value(words, '--judge-url')
    if url is not None and re.match(r'https?://(localhost|127\.0\.0\.1|\[::1\])[:/]', url):
        return []
    problems = []
    if flag_value(words, '--judge-max-usd') is None and flag_value(words, '--max-usd') is None:
        problems.append('a metered API judge needs a spend cap: add --judge-max-usd')
    for name in ('--judge-usd-in', '--judge-usd-out'):
        value = flag_value(words, name)
        if value is None or not re.fullmatch(r'\d+(\.\d+)?', value) or float(value) <= 0:
            problems.append(f'a metered API judge needs its real price: {name} is absent or zero, so the cap would count every call as free')
    return problems


def remote_problems(root, settings, words):
    """A job script named as the command on the far side of an SSH call."""
    name = os.path.basename(words[0])
    if name == 'remote_job.sh':
        if '--remove-now' in words or '--arm-only' in words:
            return []  # removing a pod, or arming its timer, is never held up
        if 'REQUIRE_AUTOSTOP' not in settings or settings['REQUIRE_AUTOSTOP'] != '1':
            return ['a pod job started without REQUIRE_AUTOSTOP=1 has no billing time cap; start it with ops/drive_runpod.sh']
        return launch_problems(root)
    return [f"{name} started directly skips ops/remote_job.sh, which arms the pod's removal timer; start it with ops/drive_runpod.sh"]


def spend_problems(root, settings, words, remote):
    name = os.path.basename(words[0])
    if remote:
        return remote_problems(root, settings, words) if re.fullmatch(r'remote_\w+\.sh', name) else []
    if name == 'drive_runpod.sh':
        if len(words) != 6 or not words[5].isdigit():
            return []  # the driver refuses this itself, before it touches a pod
        return launch_problems(root, remote=words[3], deadline=int(words[5]), judge_cap=driver_judge_cap(settings))
    if name == 'drive_run.sh' or (name == 'launch_aws.sh' and '--approved' in words) \
            or (name == 'runpodctl' and {'create', 'deploy'} & set(words[1:3])):
        problems = launch_problems(root)
        return problems and ['a machine that bills is started only once its job can start at once:'] + problems
    return judge_problems(words)


# --- commit gate ---------------------------------------------------------------------------------------------------

def withheld_terms(root):
    """The owner's patterns, one regular expression per line, from an untracked file. Raises when the file is absent or empty."""
    path = Path(root)/TERMS
    if not path.exists():
        raise FileNotFoundError(f'{TERMS} is not there: it lists, one regular expression per line, what must not be committed. '
                                'It is untracked on purpose. Ask the owner for its contents.')
    patterns = [re.compile(line.strip()) for line in path.read_text().splitlines() if line.strip() and not line.startswith('#')]
    if not patterns:
        raise ValueError(f'{TERMS} lists no patterns')
    return patterns


def text_problems(where, text, terms):
    problems = []
    for number, line in enumerate(text.splitlines(), 1):
        for index, pattern in enumerate(terms, 1):
            if pattern.search(line):
                problems.append(f'{where}:{number} matches withheld term {index} of {TERMS}')
        for label, pattern in SECRETS:
            if pattern.search(line):
                problems.append(f'{where}:{number} looks like {label}')
    return problems


def content_problems(path, data, terms):
    """Reasons not to publish this file: its name, its size, its being binary, or what its text says."""
    if SECRET_NAMES.search(path):
        return [f'{path} is named like a secret or a key file']
    if len(data) > MAX_BYTES:
        return [f'{path} is {len(data)/1e6:.0f} MB, over the {MAX_BYTES/1e6:.0f} MB limit for a tracked file; add an ignore rule for it']
    if b'\0' in data[:8192]:
        return [f'{path} is a binary file; model and build output belong in .gitignore']
    try:
        text = data.decode('utf-8')
    except UnicodeDecodeError as error:
        return [f'{path} is not UTF-8 text ({error.reason} at byte {error.start}); treated as a binary file']
    return text_problems(path, text, terms)


def message_texts(words, folder):
    texts = []
    for index, word in enumerate(words[:-1]):
        if word in ('--file', '-F'):
            if words[index+1] != '-':  # `-F -` reads the here-document, which the caller already holds
                texts.append(Path(folder, words[index+1]).read_text())
        elif word == '--message' or re.fullmatch(r'-[a-zA-Z]*m', word):
            texts.append(words[index+1])
    return texts + [word.split('=', 1)[1] for word in words if word.startswith('--message=')]


def git_problems(root, folder, words, documents):
    """`words` is a git invocation run in `folder`. Other repositories and other subcommands are not this gate's business."""
    index = 1
    while index < len(words) and words[index].startswith('-'):
        if words[index] == '-C' and index+1 < len(words):
            folder = os.path.join(folder, words[index+1])
        index += 2 if words[index] in ('-C', '-c') else 1
    if index >= len(words) or words[index] not in ('add', 'commit', 'push'):
        return []
    sub, rest = words[index], words[index+1:]
    top = subprocess.run(['git', '-C', folder, 'rev-parse', '--show-toplevel'], capture_output=True, text=True)
    ours = top.returncode == 0 and os.path.samefile(top.stdout.strip(), root)
    if not ours:
        return []
    terms = withheld_terms(root)
    files, texts = {}, []
    if sub == 'add':
        listed = subprocess.run(['git', '-C', folder, '-c', 'core.quotepath=off', 'add', '--dry-run', *rest], capture_output=True, text=True)
        if listed.returncode != 0:
            return [f'`git add --dry-run` does not accept these arguments, so what they would stage cannot be read: {listed.stderr.strip()[:300]}']
        for line in listed.stdout.splitlines():
            match = re.fullmatch(r"add '(.*)'", line)
            if match:
                files[match.group(1)] = Path(root, match.group(1)).read_bytes()
    elif sub == 'commit':
        for path in filter(None, git(root, 'diff', '--cached', '--name-only', '--diff-filter=ACMR', '-z').split('\0')):
            files[path] = git(root, 'show', ':' + path, text=False)
        if any(word == '--all' or re.fullmatch(r'-[a-zA-Z]*a[a-zA-Z]*', word) for word in rest):
            for path in filter(None, git(root, 'diff', '--name-only', '--diff-filter=ACMR', '-z').split('\0')):
                files[path] = Path(root, path).read_bytes()
        texts = message_texts(rest, folder) + documents
    else:
        # What the current branch would add to its upstream; with no upstream, everything in HEAD.
        upstream = subprocess.run(['git', '-C', str(root), 'rev-parse', '--verify', '--quiet', '@{u}'], capture_output=True, text=True)
        if upstream.returncode == 0:
            base = upstream.stdout.strip()
            span = f'{base}..HEAD'
        else:
            base = git(root, 'hash-object', '-t', 'tree', os.devnull).strip()
            span = 'HEAD'
        for path in filter(None, git(root, 'diff', '--name-only', '--diff-filter=ACMR', '-z', base, 'HEAD').split('\0')):
            files[path] = git(root, 'show', 'HEAD:' + path, text=False)
        texts = [git(root, 'log', '--format=%B', span)]
    problems = [problem for path, data in sorted(files.items()) for problem in content_problems(path, data, terms)]
    for text in texts:
        problems += text_problems('the commit message', text, terms)
    return problems


# --- the hook ------------------------------------------------------------------------------------------------------

def command_problems(root, command, folder, remote=False):
    """Every reason to refuse this shell command, run in `folder`. `remote` marks the far side of an SSH call."""
    if not GATED.search(command):
        return []
    try:
        groups, documents = simple_commands(command)
    except ValueError as error:
        return [f'this command runs git or something that bills, and its quoting could not be read ({error}). Run that part as a '
                'command of its own, and give a commit message with quotes in it through `git commit -F <file>`.']
    problems = []
    for tokens in groups:
        settings, words = command_words(tokens)
        if not words:
            continue
        if words[0] == 'bash -c':
            problems += command_problems(root, words[1], folder, remote)
        elif words[0] == 'cd' and len(words) > 1 and not remote:
            folder = os.path.join(folder, os.path.expanduser(words[1]))
        elif words[0] == 'ssh' and not remote:
            problems += command_problems(root, ssh_remote_command(words), folder, remote=True)
        elif words[0] == 'git' and not remote:
            problems += git_problems(root, folder, words, documents)
        else:
            problems += spend_problems(root, settings, words, remote)
    return problems
