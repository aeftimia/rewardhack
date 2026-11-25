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
import math
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
# A pod can bill outside the job's own deadline: for the provisional timer the driver arms before the job starts,
# and for the grace after the deadline before the pod removes itself.
PROVISIONAL_MINUTES = 15
DRIVER_GRACE_SECONDS = 600  # what ops/remote_job.sh uses when AUTOSTOP_GRACE_SECONDS is not given
DRIVER_JUDGE_MAX_USD = 1.0  # what ops/drive_runpod.sh passes to the pod when JUDGE_MAX_USD is not given
MULTIPLES = {'measured': 2.0, 'estimated': 3.0}  # CLAUDE.md, "Sizing time caps and approvals"
UNIT_MINUTES = {'s': 1/60, 'min': 1.0, 'h': 60.0}
BASIS = re.compile(r'\s*(\d+(?:\.\d+)?)\s*x\s+(measured|estimated)\s+(\d+(?:\.\d+)?)\s*(s|min|h)\s*')
READY = '.tools/venv/bin/python entrypoints/gate.py ready --remote ops/<script>.sh --deadline <minutes> ' \
        "--basis '<2x measured|3x estimated> <time> <s|min|h>' --line <budget line> --hourly <usd> --other-usd <judge cap and other costs> [--set NAME=value ...]"

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
GATED = re.compile(r'\bgit\s+(?:-\S+\s+(?:[^-\s]\S*\s+)?)*(add|commit|push)\b|drive_run|launch_aws\.sh|remote_\w+\.sh'
                   r'|runpodctl\s+(pod\s+)?(create|deploy)|--judge[ =]api')
HEREDOC = re.compile(r'''<<-?\s*(['"]?)(\w+)\1[^\n]*\n(.*?)\n[ \t]*\2[ \t]*(?=\n|\)|$)''', re.S)
ASSIGNMENT = re.compile(r'([A-Za-z_]\w*)=(.*)', re.S)
WRAPPERS = {'bash', 'sh', 'zsh', 'nohup', 'exec', 'time', 'command', 'env', 'setsid', 'sudo', 'caffeinate', 'nice', 'stdbuf',
            'if', 'then', 'else', 'elif', 'while', 'until', 'do', '!', '{'}
# Commands that take a script as something to read. Anything else that names the driver without being it is refused as unreadable.
READERS = {'cat', 'grep', 'rg', 'head', 'tail', 'less', 'wc', 'ls', 'sed', 'awk', 'git', 'diff', 'shellcheck', 'stat', 'file', 'echo', 'printf', 'chmod',
           'test', '[', '[[', 'find', 'du', 'shasum', 'sha256sum', 'md5', 'sort', 'nl', 'cut'}
DRIVERS = {'drive_runpod.sh', 'drive_run.sh'}
SHELLS = {'bash', 'sh', 'zsh', 'dash', 'ksh'}
INTERPRETER = re.compile(r'python[\d.]*|perl|ruby|node')
DRIVER_FORM = '`NAME=value ... bash ops/drive_runpod.sh <host> <port> <remote script> <results dir> <deadline>`'
# Environment settings ops/drive_runpod.sh reads. The launch record holds the ones a launch was checked with.
DRIVER_SETTINGS = re.compile(r'SMOKE_[A-Z_]+|LEAN_ISOLATION|JUDGE_KIND|JUDGE_MAX_USD|FORWARD_JUDGE_KEY|AUTOSTOP_GRACE_SECONDS|KEEP_POD')
CHECK_ONLY = 'DRIVE_CHECK_ONLY'  # set to 1, the driver validates the launch and exits before it reaches for a pod
SSH_WITH_VALUE = set('pioFJLRDEbcelmOQSWw')


def git(root, *args, text=True):
    return subprocess.run(['git', '-C', str(root), '-c', 'core.quotepath=off', *args], check=True, capture_output=True, text=text).stdout


# --- reading a command ---------------------------------------------------------------------------------------------

def simple_commands(command):
    """The command as a list of simple commands, each (the operator before it, its tokens), and the text of its here-documents.

    Raises ValueError when the quoting cannot be read.
    """
    documents = [match.group(3) for match in HEREDOC.finditer(command)]
    lexer = shlex.shlex(HEREDOC.sub(' ', command).replace('\n', ' ; '), posix=True, punctuation_chars=True)
    lexer.whitespace_split = True
    groups, before, current = [], '', []
    for token in lexer:
        if token and not token.strip(';&|()<>'):
            groups.append((before, current))
            before, current = token, []
        else:
            current.append(token)
    return [(operator, tokens) for operator, tokens in groups + [(before, current)] if tokens], documents


def hidden_driver(before, tokens, words):
    """Why a simple command that names the pod driver cannot be checked, or None when it is the driver itself or only reads it.

    The launch check reads `bash ops/drive_runpod.sh <five arguments>`. Any other way of naming the driver might run it
    where the check cannot follow (a variable, a glob, `eval`, a script on stdin), so it is refused, not passed.
    """
    if not words:
        return 'it puts the driver in a shell variable'
    if any('`' in token for token in tokens):
        return 'it runs something inside backticks'
    if before.startswith('<'):
        return 'it feeds the driver to a shell on standard input'
    if words[0] in ('bash -c', 'git', 'ssh', 'cd'):
        return None  # read further by the caller
    if os.path.basename(words[0]) in DRIVERS:
        return None
    if words[0] in READERS:
        return None
    return f'it names the driver after `{words[0][:40]}`, which may run it'


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
        elif token not in ('export', 'declare', 'typeset'):
            # `bash -x script`, `caffeinate -i cmd`: the wrapper's own flags are not the command. `bash -s -- args` reads its script from stdin.
            while index+1 < len(tokens) and tokens[index+1].startswith('-') and tokens[index+1] not in ('-s', '--'):
                index += 1
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


def grace_seconds(settings):
    """How long the pod outlives its job under these driver settings. Raises ValueError when the setting is not a whole number."""
    if 'AUTOSTOP_GRACE_SECONDS' not in settings:
        return DRIVER_GRACE_SECONDS
    if not settings['AUTOSTOP_GRACE_SECONDS'].isdigit():
        raise ValueError(f"AUTOSTOP_GRACE_SECONDS={settings['AUTOSTOP_GRACE_SECONDS']} is not a whole number of seconds")
    return int(settings['AUTOSTOP_GRACE_SECONDS'])


def driver_check(root, remote, deadline, settings):
    """Run the driver's own validation of this launch, with nothing deployed. Returns the reasons it refuses."""
    driver = root/'ops'/'drive_runpod.sh'
    if CHECK_ONLY not in driver.read_text():
        return [f'ops/drive_runpod.sh has no {CHECK_ONLY} mode, so its refusals would only be met with a pod already billing']
    # Paths relative to the repository, as the driver is run: it refuses a results path with a space in it.
    done = subprocess.run(['bash', 'ops/drive_runpod.sh', 'check.invalid', '22', remote, str(RECORD.parent/'driver-check'), str(deadline)],
                          cwd=root, env={**os.environ, **settings, CHECK_ONLY: '1'}, capture_output=True, text=True)
    if done.returncode != 0 or 'check passed' not in done.stdout:
        said = '\n    '.join((done.stdout + done.stderr).strip().splitlines()[-6:])
        return [f'ops/drive_runpod.sh refuses this launch with these settings (exit {done.returncode}):\n    {said}']
    return []


def ready(root, remote, deadline, basis, line, hourly, other_usd, launch_cap, settings, test_command):
    """Make every check a pod launch depends on and write the launch record. Returns (record, problems); no record is written with problems.

    `settings` are the driver's environment settings for this launch (LEAN_ISOLATION, SMOKE_*, ...): the driver command must carry exactly these.
    """
    root = Path(root)
    problems = []
    if not re.fullmatch(r'ops/[A-Za-z0-9_-]+\.sh', remote) or not (root/remote).is_file():
        return None, [f'{remote} is not an existing ops/*.sh script']
    unknown = sorted(name for name in settings if not DRIVER_SETTINGS.fullmatch(name))
    if unknown:
        return None, [f"not settings ops/drive_runpod.sh reads: {', '.join(unknown)}"]
    floor = re.search(r'^# deadline: (\d+)\b', (root/remote).read_text(), re.M)
    if floor and deadline < int(floor.group(1)):
        problems.append(f'{remote} asks for a deadline of at least {floor.group(1)} min (its `# deadline:` line); {deadline} would cut it off')
    try:
        overhead = PROVISIONAL_MINUTES + math.ceil(grace_seconds(settings) / 60)
    except ValueError as error:
        return None, [str(error)]
    judge_cap = driver_judge_cap(settings)
    if judge_cap is not None and judge_cap > other_usd:
        problems.append(f'these settings let the API judge spend ${judge_cap:.2f}; --other-usd must cover it and is ${other_usd:.2f}')
    problems += driver_check(root, remote, deadline, settings)
    dirty = dirty_paths(root)
    if dirty:
        problems.append('uncommitted changes in what the pod would run, so it is not the tested commit:\n    ' + '\n    '.join(dirty[:12]))
    try:
        needed = basis_minutes(basis)
        if deadline < needed:
            problems.append(f'the deadline of {deadline} min is under the {needed:g} min that {basis.strip()!r} calls for')
    except ValueError as error:
        problems.append(str(error))
    hours = (deadline + overhead) / 60
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
                  worst_case_minutes=deadline + overhead, settings=dict(sorted(settings.items())),
                  written_at=datetime.now(timezone.utc).isoformat(timespec='seconds'))
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


def mark_launched(root):
    """A record covers one launch: its budget check counted one worst case. Called when a driver command is let through."""
    path = Path(root)/RECORD
    record = json.loads(path.read_text())
    record['launched_at'] = datetime.now(timezone.utc).isoformat(timespec='seconds')
    path.write_text(json.dumps(record, indent=1) + '\n')


def launch_problems(root, remote=None, deadline=None, settings=None):
    """Why the launch record does not cover a launch now, optionally of this remote script at this deadline with these driver settings."""
    root = Path(root)
    if not (root/RECORD).exists():
        return [f'there is no launch record. Settle the launch first:\n    {READY}']
    record = json.loads((root/RECORD).read_text())
    problems = []
    if 'launched_at' in record:
        problems.append(f"the launch record already started a launch at {record['launched_at']}; its budget check counted one")
    if settings is not None and settings != record['settings']:
        def show(values):
            return ' '.join(f'{name}={value}' for name, value in sorted(values.items())) if values else '(none)'
        problems.append(f"the driver settings on this command are not the ones that were checked.\n    checked: {show(record['settings'])}\n"
                        f'    command: {show(settings)}')
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


def driver_problems(root, settings, words, exported, launches):
    """`words` starts at ops/drive_runpod.sh. A passing launch is noted in `launches`."""
    arguments = words[1:6]
    if len(arguments) < 5:
        return []  # too few arguments: the driver prints its usage and exits before it touches a pod
    unread = [word for word in arguments if '$' in word or '`' in word]
    if unread:
        return [f"the driver's arguments must be written out so the launch can be checked; {' '.join(unread)} is only known to the shell"]
    if not arguments[4].isdigit():
        return []  # the driver refuses a deadline that is not a whole number
    outside = sorted(name for name in exported if DRIVER_SETTINGS.fullmatch(name))
    if outside:
        return [f"{', '.join(outside)} is set earlier in the command, where the launch check cannot hold it to the record; "
                'put driver settings on the driver command itself: NAME=value bash ops/drive_runpod.sh ...']
    problems = launch_problems(root, remote=arguments[2], deadline=int(arguments[4]),
                               settings={name: value for name, value in settings.items() if DRIVER_SETTINGS.fullmatch(name)})
    if not problems:
        launches.append(arguments[2])
    return problems


def spend_problems(root, settings, words, remote, exported, launches):
    name = os.path.basename(words[0])
    if remote:
        return remote_problems(root, settings, words) if re.fullmatch(r'remote_\w+\.sh', name) else []
    if name == 'drive_runpod.sh':
        return driver_problems(root, settings, words, exported, launches)
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

def command_problems(root, command, folder, remote=False, launches=None, exported=()):
    """Every reason to refuse this shell command, run in `folder`. `remote` marks the far side of an SSH call.

    `launches`, when given, collects the remote script of each driver launch the command was allowed. `exported` names
    settings already in force around the command (those in front of a `bash -c` that contains it).
    """
    launches = [] if launches is None else launches
    exported = set(exported)
    if not GATED.search(command):
        return []
    try:
        groups, documents = simple_commands(command)
    except ValueError as error:
        return [f'this command runs git or something that bills, and its quoting could not be read ({error}). Run that part as a '
                'command of its own, and give a commit message with quotes in it through `git commit -F <file>`.']
    problems = []
    programs = {os.path.basename(token) for _, tokens in groups for token in tokens}
    scripted = [text for text in documents if 'drive_run' in text
                and (programs & SHELLS or (re.search(r'subprocess|os\.system|Popen|exec', text) and any(INTERPRETER.fullmatch(p) for p in programs)))]
    if scripted and not remote:
        problems.append('this runs a script from a here-document that names the pod driver, where the launch check cannot follow it. '
                        f'Start the driver as its own command: {DRIVER_FORM}')
    after_reader = False
    for before, tokens in groups:
        settings, words = command_words(tokens)
        names_driver = not remote and any('drive_run' in token for token in tokens)
        # `cat ops/drive_runpod.sh | bash -s -- ...`: the driver's text reaches a shell through a pipe, however many readers it passes.
        piped = before.startswith('|') and after_reader
        hidden = None
        if piped and SHELLS & {os.path.basename(token) for token in tokens}:
            hidden = 'it pipes the driver into a shell'
        elif names_driver:
            hidden = hidden_driver(before, tokens, words)
        after_reader = piped or (names_driver and bool(words) and words[0] in READERS)
        if hidden:
            problems.append(f'this command names the pod driver in a form the launch check cannot read: {hidden}. '
                            f'Start the driver as its own command: {DRIVER_FORM}')
            continue
        if not words or words[0] in ('export', 'declare', 'typeset'):
            # Set for the rest of the command, not for one program: `X=1; cmd` and `export X=1; cmd`.
            exported |= set(settings) | {match.group(1) for match in map(ASSIGNMENT.fullmatch, words[1:]) if match}
            continue
        if words[0] == 'bash -c':
            # Settings in front of `bash -c` reach everything inside it, like an export.
            problems += command_problems(root, words[1], folder, remote, launches, exported | set(settings))
        elif words[0] == 'cd' and len(words) > 1 and not remote:
            folder = os.path.join(folder, os.path.expanduser(words[1]))
        elif words[0] == 'ssh' and not remote:
            problems += command_problems(root, ssh_remote_command(words), folder, remote=True)
        elif words[0] == 'git' and not remote:
            problems += git_problems(root, folder, words, documents)
        else:
            problems += spend_problems(root, settings, words, remote, exported, launches)
    if len(launches) > 1 and not problems:
        problems.append(f'this command starts {len(launches)} launches, and a launch record covers one: its budget check counted one worst case')
    return problems
