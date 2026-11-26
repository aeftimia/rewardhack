#!/usr/bin/env python3
"""Gate on what spends money or publishes.

    gate.py ready --remote ops/<script>.sh --deadline <min> --basis '2x measured 30 min' --line <line> --hourly <usd> --other-usd <usd>
        Settle a pod launch: clean tree, cap basis, budget check, test suite. Writes the launch record
        that the driver then checks itself against.
    gate.py launch --remote ops/<script>.sh --deadline <min>
        Run by ops/drive_runpod.sh before it reaches for a pod: exits 2 unless the launch record covers this
        launch with the driver settings in this environment, and marks the record used.
    gate.py hook
        Claude Code PreToolUse hook on Bash: the tool call on stdin; exit 2 refuses the command.
    gate.py codex-hook
        Codex PreToolUse adapter for the shared no-silent-fallbacks checker.
    gate.py pre-commit | commit-msg <file> | pre-push <remote> <url>
        The commit gate, run by git through ops/githooks; exit 1 refuses.
    gate.py install-hooks [--repo <path>]
        Point a repository's git at ops/githooks (core.hooksPath). Once per clone, from its main checkout.

Every subcommand but `ready` and `install-hooks` is normally run through ops/gate.sh, which picks the project's
Python and this checkout's library.
"""
import argparse
import json
import os
import sys
import traceback
from pathlib import Path

from rewardhack.budget import LINES
from rewardhack.gates import BASIS_FORMS, HOOK_CHECKS, command_problems, driver_launch, install_hooks, ready

ROOT = Path(__file__).resolve().parent.parent


def refuse(problems, code, heading):
    if problems:
        print(f'{heading}\n\n  ' + '\n  '.join(problems), file=sys.stderr)
        return code
    return 0


def hook(args):
    call = json.load(sys.stdin)
    if call['tool_name'] != 'Bash':
        return 0
    return refuse(command_problems(ROOT, call['tool_input']['command']), 2, 'gate: this command is refused before it runs.')


def launch(args):
    return refuse(driver_launch(ROOT, args.remote, args.deadline, os.environ), 2,
                  'gate: ops/drive_runpod.sh refuses to start: the launch record does not cover this launch.')


def codex_hook(args):
    from rewardhack.codex_hooks import main as check_codex_call
    return check_codex_call()


def commit_gate(args):
    # Git runs its hooks at the top of the working tree being committed to, which may be a linked worktree or another clone.
    return refuse(HOOK_CHECKS[args.cmd](Path.cwd(), *args.hook_args(args)), 1, f'gate: git {args.cmd} refuses: this would publish what must not be.')


def install(args):
    print(f'core.hooksPath = {install_hooks(args.repo, ROOT/"ops"/"githooks")}')
    return 0


def settle(args):
    if not all('=' in setting for setting in args.settings):
        raise SystemExit('--set takes NAME=VALUE')
    record, problems = ready(ROOT, args.remote, args.deadline, args.basis, args.line, args.hourly, args.other_usd, args.launch_cap,
                             dict(setting.split('=', 1) for setting in args.settings), [sys.executable, '-m', 'pytest', '-q'])
    for problem in problems:
        print('REFUSED:', problem)
    if problems:
        return 1
    print(f"ready: {record['remote']} at {record['deadline_minutes']} min on commit {record['head'][:7]}; worst case "
          f"${record['worst_case_usd']:.2f} for {record['worst_case_minutes']} min "
          f"({record['worst_case_minutes']-record['deadline_minutes']} of them outside the job's deadline). Start it with:\n  "
          + ' '.join(f'{name}={value}' for name, value in record['settings'].items())
          + f" bash ops/drive_runpod.sh <host> <port> {record['remote']} <results dir> {record['deadline_minutes']}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('hook').set_defaults(run=hook)
    sub.add_parser('codex-hook').set_defaults(run=codex_hook)
    sub.add_parser('pre-commit').set_defaults(run=commit_gate, hook_args=lambda args: ())
    message = sub.add_parser('commit-msg')
    message.add_argument('message_file', help='the file git holds the commit message in')
    message.set_defaults(run=commit_gate, hook_args=lambda args: (args.message_file,))
    pushed = sub.add_parser('pre-push', help="git's pre-push hook; the refs being pushed on stdin")
    pushed.add_argument('remote', help='the name of the remote, or its URL when it has none')
    pushed.add_argument('url')
    pushed.set_defaults(run=commit_gate, hook_args=lambda args: (args.remote, sys.stdin.read()))
    installer = sub.add_parser('install-hooks')
    installer.add_argument('--repo', type=Path, default=ROOT, help='the repository whose git runs the hooks (default: this one)')
    installer.set_defaults(run=install)
    started = sub.add_parser('launch')
    started.add_argument('--remote', required=True, help='the ops/*.sh script given to the driver')
    started.add_argument('--deadline', type=int, required=True, help='the deadline in minutes given to the driver')
    started.set_defaults(run=launch)
    settle_parser = sub.add_parser('ready')
    settle_parser.add_argument('--remote', required=True, help='the ops/*.sh script the pod will run')
    settle_parser.add_argument('--deadline', type=int, required=True, help='job deadline in minutes, as given to ops/drive_runpod.sh')
    settle_parser.add_argument('--basis', required=True, help=f'what the deadline is sized from: {BASIS_FORMS}')
    settle_parser.add_argument('--line', required=True, choices=sorted(LINES), help='grant budget line this launch is charged to')
    settle_parser.add_argument('--hourly', type=float, required=True, help='pod price in USD per hour')
    settle_parser.add_argument('--other-usd', type=float, required=True,
                               help='costs beside the pod: the judge spend cap (JUDGE_MAX_USD) when the job uses the API judge; 0 only when there are none')
    settle_parser.add_argument('--launch-cap', type=float, help='ceiling approved for this one launch')
    settle_parser.add_argument('--set', action='append', default=[], metavar='NAME=VALUE', dest='settings',
                               help='a driver setting for this launch (LEAN_ISOLATION=none, SMOKE_STEPS=4, FORWARD_JUDGE_KEY=1, ...); '
                                    'the driver must then run with exactly the settings given here')
    settle_parser.set_defaults(run=settle)
    args = ap.parse_args()
    return args.run(args)


if __name__ == '__main__':
    try:
        code = main()
    except Exception:  # noqa: BLE001  any crash of the gate is reported and refuses (exit 2), never waved through
        # A gate that crashes must not wave anything through: say that the check itself failed, and refuse.
        traceback.print_exc()
        print('gate: the check itself failed (see above). This is not a finding about the command.', file=sys.stderr)
        sys.exit(2)
    sys.exit(code)
