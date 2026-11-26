"""The project's gates: one on what spends money, one on what git publishes.

Spend gate. A pod job may start only from a launch record (`entrypoints/gate.py ready`), which is written
after the tree is clean, the time cap has a stated basis at the project's multiple, the budget check passes
and the test suite passes. The record names the commit, the ledger, the remote script, the deadline and the
driver settings. `ops/drive_runpod.sh` asks `launch_problems` itself before it reaches for a pod
(`entrypoints/gate.py launch`), so the check follows the driver however it is started. What cannot live in a
script the project owns stays in a Claude Code PreToolUse hook on Bash (`command_problems`): a pod created
with a raw `runpodctl create`, a job script started over SSH around `remote_job.sh` or without
`REQUIRE_AUTOSTOP=1`, and a metered API judge with no spend cap. Removing a pod is never refused.

Commit gate. Git hooks (`ops/githooks`, installed with `entrypoints/gate.py install-hooks`) read the staged
blobs, the commit message and every commit a push would publish, for the owner's withheld terms (kept in an
untracked file in the main checkout, so the repo does not name them), for secrets, and for binary, non-UTF-8
or very large files. Git runs them for every commit, from any worktree, whoever types it.
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
SHIPPED = ('src', 'ops', 'entrypoints', 'tests', 'pyproject.toml', 'requirements.txt', 'requirements-vllm.txt')
# A pod can bill outside the job's own deadline: for the provisional timer the driver arms before the job starts,
# and for the grace after the deadline before the pod removes itself.
PROVISIONAL_MINUTES = 15
DRIVER_GRACE_SECONDS = 600  # what ops/remote_job.sh uses when AUTOSTOP_GRACE_SECONDS is not given
DRIVER_JUDGE_MAX_USD = 1.0  # what ops/drive_runpod.sh passes to the pod when JUDGE_MAX_USD is not given
# The one statement of the cap multiples: CLAUDE.md, the help text below and every check read them from here
# (tests/test_cap_comments.py keeps the numbers out of CLAUDE.md). Owner: 2026-11-25; estimated lowered from 3 on 26 Nov.
MULTIPLES = {'measured': 2.0, 'estimated': 1.5}
BASIS_FORMS = ' or '.join(f"'{multiple:g}x {kind} <time> <s|min|h>'" for kind, multiple in MULTIPLES.items())
UNIT_MINUTES = {'s': 1/60, 'min': 1.0, 'h': 60.0}
BASIS = re.compile(r'\s*(\d+(?:\.\d+)?)\s*x\s+(measured|estimated)\s+(\d+(?:\.\d+)?)\s*(s|min|h)\s*')
READY = '.tools/venv/bin/python entrypoints/gate.py ready --remote ops/<script>.sh --deadline <minutes> ' \
        f"--basis <{BASIS_FORMS}> --line <budget line> --hourly <usd> --other-usd <judge cap and other costs> [--set NAME=value ...]"
# Environment settings ops/drive_runpod.sh reads. The launch record holds the ones a launch was checked with.
DRIVER_SETTINGS = re.compile(r'SMOKE_[A-Z_]+|LEAN_ISOLATION|JUDGE_KIND|JUDGE_MAX_USD|FORWARD_JUDGE_KEY|AUTOSTOP_GRACE_SECONDS|KEEP_POD')
CHECK_ONLY = 'DRIVE_CHECK_ONLY'  # set to 1, the driver validates the launch and exits before it reaches for a pod

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
HOOKS = ('pre-commit', 'commit-msg', 'pre-push')  # the files in ops/githooks, each a call to `entrypoints/gate.py <its name>`


def git(root, *args, text=True):
    return subprocess.run(['git', '-C', str(root), '-c', 'core.quotepath=off', *args], check=True, capture_output=True, text=text).stdout


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


# Time limits in the pod scripts state what they are sized from, on their line or the one above: `# cap: 2x measured 750 s`.
# The limit is read in the unit the comment uses. Checked by tests/test_cap_comments.py against MULTIPLES, so the launch
# record and the scripts follow one rule (owner, 26 Nov: the check used to live in the global edit hook).
SCRIPT_LIMITS = [re.compile(r'\btimeout\s+(?:-\S+\s+)*(\d+(?:\.\d+)?)[smh]?\b'),
                 re.compile(r'\b\w*(?:TIMEOUT|DEADLINE|WAIT)\w*(?:=|:-)(\d+(?:\.\d+)?)\b')]  # not LIMIT: it names counts as often as times
CAP_COMMENT = re.compile(r'cap:\s*(\d+(?:\.\d+)?)\s*x\s+(measured|estimated)\s+(\d+(?:\.\d+)?)\s*(s|min|h)\b')


def unsized_limits(text):
    """(line number, why) for each time limit in a shell script with no `cap:` basis, a multiple below MULTIPLES, or a
    value under its own basis. This checks that a basis is stated and the arithmetic holds, not that the measurement is true."""
    lines = text.splitlines()
    found = []
    for number, line in enumerate(lines, 1):
        values = [float(match.group(1)) for pattern in SCRIPT_LIMITS for match in pattern.finditer(line.split('#')[0])]
        if not values:
            continue
        above = lines[number-2] if number > 1 else ''
        cap = CAP_COMMENT.search(line) or (CAP_COMMENT.search(above) if above.strip().startswith('#') else None)
        if not cap:
            found.append((number, 'a time limit with no stated basis: say on this line or the one above what it is sized from, '
                                  f"'# cap: <basis>' with a basis of {BASIS_FORMS}"))
            continue
        multiple, kind, amount, unit = float(cap.group(1)), cap.group(2), float(cap.group(3)), cap.group(4)
        if multiple < MULTIPLES[kind]:
            found.append((number, f'a limit on a {kind} time is at least {MULTIPLES[kind]:g}x it, and this says {multiple:g}x'))
        elif min(values) < multiple * amount:
            found.append((number, f'the limit {min(values):g} is under {multiple:g} x {amount:g} {unit}, the basis its own comment gives'))
    return found


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
                          cwd=root, env={**os.environ, **settings, CHECK_ONLY: '1'}, capture_output=True, text=True, check=False)
    if done.returncode != 0 or 'check passed' not in done.stdout:
        said = '\n    '.join((done.stdout + done.stderr).strip().splitlines()[-6:])
        return [f'ops/drive_runpod.sh refuses this launch with these settings (exit {done.returncode}):\n    {said}']
    return []


def ready(root, remote, deadline, basis, line, hourly, other_usd, launch_cap, settings, test_command):
    """Make every check a pod launch depends on and write the launch record. Returns (record, problems); no record is written with problems.

    `settings` are the driver's environment settings for this launch (LEAN_ISOLATION, SMOKE_*, ...): the driver must run with exactly these.
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
    tests = subprocess.run(test_command, cwd=root, capture_output=True, text=True, check=False)
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
    """A record covers one launch: its budget check counted one worst case. Called when the driver is let through."""
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
        problems.append(f"the driver settings in force are not the ones that were checked.\n    checked: {show(record['settings'])}\n"
                        f'    in force: {show(settings)}')
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


def driver_launch(root, remote, deadline, environment):
    """The driver's own check, made before it reaches for a pod. Returns the reasons to refuse; a launch let through is marked as started."""
    settings = {name: value for name, value in environment.items() if DRIVER_SETTINGS.fullmatch(name)}
    problems = launch_problems(root, remote, deadline, settings)
    if not problems:
        mark_launched(root)
    return problems


# --- the Bash hook: what no project script can check for itself -----------------------------------------------------

# A command without one of these shapes is passed without being read further.
GATED = re.compile(r'runpodctl|remote_\w+\.sh|--judge[ =]api')
REMOTE_SCRIPT = re.compile(r'remote_\w+\.sh')
HEREDOC = re.compile(r'''<<-?\s*(['"]?)(\w+)\1[^\n]*\n(.*?)\n[ \t]*\2[ \t]*(?=\n|\)|$)''', re.S)
ASSIGNMENT = re.compile(r'([A-Za-z_]\w*)=(.*)', re.S)
WRAPPERS = {'bash', 'sh', 'zsh', 'nohup', 'exec', 'time', 'command', 'env', 'setsid', 'sudo', 'caffeinate', 'nice', 'stdbuf',
            'export', 'if', 'then', 'else', 'elif', 'while', 'until', 'do', '!', '{'}
SSH_WITH_VALUE = set('pioFJLRDEbcelmOQSWw')
# Machines that bill from the moment they exist, started by a command the hook sees whole. The value says whether these words start one.
STARTS_A_MACHINE = {
    'runpodctl': lambda words: bool({'create', 'deploy'} & set(words[1:3])),  # `runpodctl create pod`, `runpodctl pod create`
}


def simple_commands(command):
    """The command as a list of simple commands (lists of tokens), and the text of its here-documents. Raises ValueError when the quoting cannot be read."""
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
    return [tokens for tokens in groups + [current] if tokens], documents


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
        else:
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


def judge_problems(words):
    """A metered judge needs its cap on the command line and a model with a row in model_io.PRICES (the one statement of
    a price); a judge served from this machine is not metered."""
    from .model_io import PRICES
    if flag_value(words, '--judge') != 'api':
        return []
    url = flag_value(words, '--judge-url')
    if url is not None and re.match(r'https?://(localhost|127\.0\.0\.1|\[::1\])[:/]', url):
        return []
    problems = []
    if flag_value(words, '--judge-max-usd') is None and flag_value(words, '--max-usd') is None:
        problems.append('a metered API judge needs a spend cap: add --judge-max-usd')
    model = flag_value(words, '--judge-model')
    if model not in PRICES:
        problems.append(f'a metered API judge needs a model with a price in model_io.PRICES, so the cap counts real dollars: --judge-model is {model!r}')
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


def local_problems(root, words):
    name = os.path.basename(words[0])
    if name in STARTS_A_MACHINE and STARTS_A_MACHINE[name](words):
        problems = launch_problems(root)
        return problems and ['a machine that bills is started only once its job can start at once:'] + problems
    return judge_problems(words)


def command_problems(root, command, remote=False):
    """Every reason to refuse this shell command. `remote` marks the far side of an SSH call, where only job scripts are looked for."""
    if not GATED.search(command):
        return []
    try:
        groups, documents = simple_commands(command)
    except ValueError as error:
        return [f'this command starts a pod, a pod job or a metered judge, and its quoting could not be read ({error}). Run that part as a '
                'command of its own.']
    problems = []
    for tokens in groups:
        settings, words = command_words(tokens)
        if not words:
            continue
        if words[0] == 'bash -c':
            problems += command_problems(root, words[1], remote)
        elif remote:
            problems += remote_problems(root, settings, words) if REMOTE_SCRIPT.fullmatch(os.path.basename(words[0])) else []
        elif os.path.basename(words[0]) == 'ssh':
            # What reaches the far side: the command after the host, and any here-document fed to the remote shell.
            problems += command_problems(root, ssh_remote_command(words), remote=True)
            problems += [problem for document in documents for problem in command_problems(root, document, remote=True)]
        else:
            problems += local_problems(root, words)
    return problems


# --- the commit gate -----------------------------------------------------------------------------------------------

ZERO_ID = re.compile(r'0+')
GITLINK = '160000'  # a submodule entry: a commit id, with no content of its own in this repository
SCISSORS = re.compile(r'^\S ------------------------ >8 ------------------------$', re.M)  # `git commit -v`: what follows is the diff, not the message


def main_checkout(root):
    """The working tree that holds the repository's own .git, found from it or from any linked worktree of it."""
    return Path(git(root, 'rev-parse', '--path-format=absolute', '--git-common-dir').strip()).parent


def withheld_terms(root):
    """The owner's patterns, one regular expression per line, from an untracked file in the main checkout. Raises when it is absent or empty."""
    path = main_checkout(root)/TERMS
    if not path.exists():
        raise FileNotFoundError(f'{path} is not there: it lists, one regular expression per line, what must not be committed. '
                                'It is untracked on purpose. Ask the owner for its contents.')
    patterns = [re.compile(line.strip()) for line in path.read_text().splitlines() if line.strip() and not line.startswith('#')]
    if not patterns:
        raise ValueError(f'{path} lists no patterns')
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


def changed_blobs(root, *diff):
    """{(path, blob id)} for each file a `git diff ...` or `git diff-tree ...` command (given without output options) adds or changes."""
    out = git(root, *diff, '--raw', '-z', '--no-renames', '--diff-filter=ACMT')
    fields = out.split('\0')
    blobs = set()
    for meta, path in zip(fields[0::2], fields[1::2], strict=False):  # `-z` output ends in a NUL: one empty field is left over
        _, new_mode, _, new_id, _ = meta.lstrip(':').split(' ')
        if new_mode != GITLINK:
            blobs.add((path, new_id))
    return blobs


def blob_problems(root, blobs, terms):
    return [problem for path, blob in sorted(blobs) for problem in content_problems(path, git(root, 'cat-file', 'blob', blob, text=False), terms)]


def staged_problems(root):
    """pre-commit: the files as staged in the index this commit is made from (git passes a temporary one for `commit -a` and `commit <path>`)."""
    terms = withheld_terms(root)
    return blob_problems(root, changed_blobs(root, 'diff', '--cached'), terms)


def message_problems(root, message_file):
    """commit-msg: the message as git will record it, up to the `git commit -v` scissors line. Comment lines are read too."""
    terms = withheld_terms(root)
    text = SCISSORS.split(Path(message_file).read_text(), maxsplit=1)[0]
    return text_problems('the commit message', text, terms)


def pushed_commits(root, remote, local_id, remote_id):
    """The commits a push of `local_id` publishes: those not already on any of this remote's branches as last fetched, nor under `remote_id`."""
    known = []
    # The remote's current tip is excluded when this clone has it; when it does not (someone else pushed since the last fetch), the
    # tracking branches still exclude what was last seen there, and anything else is read: more than the push adds, never less.
    if not ZERO_ID.fullmatch(remote_id) and subprocess.run(['git', '-C', str(root), 'cat-file', '-e', remote_id + '^{commit}'],
                                                            capture_output=True, check=False).returncode == 0:
        known.append(remote_id)
    return git(root, 'rev-list', local_id, '--not', f'--remotes={remote}', *known).split()


def push_problems(root, remote, updates):
    """pre-push: every commit each pushed ref adds, its files as that commit changed them and its message. `updates` are git's stdin lines."""
    terms = withheld_terms(root)
    commits = []
    for line in updates.splitlines():
        _, local_id, _, remote_id = line.split(' ')
        if ZERO_ID.fullmatch(local_id):
            continue  # deleting a remote branch publishes nothing
        commits += [commit for commit in pushed_commits(root, remote, local_id, remote_id) if commit not in commits]
    blobs = set()
    problems = []
    for commit in commits:
        blobs |= changed_blobs(root, 'diff-tree', '-r', '--no-commit-id', '--root', '--diff-merges=first-parent', commit)
        problems += text_problems(f'the message of commit {commit[:9]}', git(root, 'show', '-s', '--format=%B', commit), terms)
    return blob_problems(root, blobs, terms) + problems


HOOK_CHECKS = {'pre-commit': staged_problems, 'commit-msg': message_problems, 'pre-push': push_problems}


def linked_worktree(path):
    return git(path, 'rev-parse', '--path-format=absolute', '--git-dir') != git(path, 'rev-parse', '--path-format=absolute', '--git-common-dir')


def install_hooks(repo, hooks):
    """Point `repo`'s git at the hook scripts in `hooks` (core.hooksPath, an absolute path). Returns the path set.

    Refused when `hooks` lies in a linked worktree of the same repository: removing that worktree would leave
    git with no hooks to run, which it does without a word, so every later commit would go ungated.
    """
    hooks = Path(hooks).resolve()
    missing = [name for name in HOOKS if not os.access(hooks/name, os.X_OK)]
    if missing:
        raise FileNotFoundError(f"no executable {', '.join(missing)} in {hooks}")
    holder = hooks.parent.parent
    same_repository = main_checkout(holder) == main_checkout(repo)
    if same_repository and linked_worktree(holder):
        raise ValueError(f'{hooks} is in a linked worktree, which goes away with its branch, and git skips hooks it cannot find without a word. '
                         f'Run install-hooks from the main checkout, {main_checkout(repo)}.')
    git(repo, 'config', 'core.hooksPath', str(hooks))
    return hooks
