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
    gate.py terms | gitleaks  pre-commit | commit-msg <file> | pre-push
        The commit gate's local hooks, run by pre-commit (.pre-commit-config.yaml); exit 1 refuses. At pre-push the
        remote is read from PRE_COMMIT_REMOTE_NAME, which pre-commit sets.

Every subcommand but `ready` is normally run through ops/gate.sh, which picks the project's Python and this
checkout's library.
"""
import argparse
import json
import os
import sys
import traceback
from pathlib import Path

from rewardhack.budget import LINES
from rewardhack.gates import COMMIT_GATE, command_problems, driver_launch, ready

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


STAGE_ARGS = {
    'pre-commit': lambda args: (),
    'commit-msg': lambda args: (args.message_file,),
    'pre-push': lambda args: (os.environ['PRE_COMMIT_REMOTE_NAME'],),
}


def commit_gate(args):
    # pre-commit runs its hooks at the top of the working tree being committed to, which may be a linked worktree or another clone.
    problems = COMMIT_GATE[args.cmd](Path.cwd(), args.stage, *STAGE_ARGS[args.stage](args))
    return refuse(problems, 1, f'gate: {args.cmd} at git {args.stage} refuses: this would publish what must not be.')


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
    for check in COMMIT_GATE:
        hook_parser = sub.add_parser(check)
        hook_parser.set_defaults(run=commit_gate)
        stages = hook_parser.add_subparsers(dest='stage', required=True)
        stages.add_parser('pre-commit')
        stages.add_parser('commit-msg').add_argument('message_file', help='the file git holds the commit message in')
        stages.add_parser('pre-push')
    started = sub.add_parser('launch')
    started.add_argument('--remote', required=True, help='the ops/*.sh script given to the driver')
    started.add_argument('--deadline', type=int, required=True, help='the deadline in minutes given to the driver')
    started.set_defaults(run=launch)
    settle_parser = sub.add_parser('ready')
    settle_parser.add_argument('--remote', required=True, help='the ops/*.sh script the pod will run')
    settle_parser.add_argument('--deadline', type=int, required=True, help='job deadline in minutes, as given to ops/drive_runpod.sh')
    settle_parser.add_argument('--basis', required=True, help="what the deadline is sized from: '2x measured 30 min' or '1.5x estimated 20 min'")
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
    except Exception:
        # A gate that crashes must not wave anything through: say that the check itself failed, and refuse.
        traceback.print_exc()
        print('gate: the check itself failed (see above). This is not a finding about the command.', file=sys.stderr)
        sys.exit(2)
    sys.exit(code)
