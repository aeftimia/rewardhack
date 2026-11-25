#!/usr/bin/env python3
"""Gate on commands that spend money or publish.

    gate.py ready --remote ops/<script>.sh --deadline <min> --basis '2x measured 30 min' --line <line> --hourly <usd>
        Settle a pod launch: clean tree, cap basis, budget check, test suite. Writes the launch record
        that the driver command is then checked against.
    gate.py hook
        Claude Code PreToolUse hook on Bash: the tool call on stdin; exit 2 refuses the command.
"""
import argparse
import json
import sys
import traceback
from pathlib import Path

from rewardhack.budget import LINES
from rewardhack.gates import command_problems, mark_launched, ready

ROOT = Path(__file__).resolve().parent.parent


def hook():
    call = json.load(sys.stdin)
    if call['tool_name'] != 'Bash':
        return 0
    launches = []
    problems = command_problems(ROOT, call['tool_input']['command'], call['cwd'], launches=launches)
    if not problems:
        if launches:
            mark_launched(ROOT)
        return 0
    print('gate: this command is refused before it runs.\n\n  ' + '\n  '.join(problems), file=sys.stderr)
    return 2


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('hook')
    settle = sub.add_parser('ready')
    settle.add_argument('--remote', required=True, help='the ops/*.sh script the pod will run')
    settle.add_argument('--deadline', type=int, required=True, help='job deadline in minutes, as given to ops/drive_runpod.sh')
    settle.add_argument('--basis', required=True, help="what the deadline is sized from: '2x measured 30 min' or '3x estimated 20 min'")
    settle.add_argument('--line', required=True, choices=sorted(LINES), help='grant budget line this launch is charged to')
    settle.add_argument('--hourly', type=float, required=True, help='pod price in USD per hour')
    settle.add_argument('--other-usd', type=float, required=True,
                        help='costs beside the pod: the judge spend cap (JUDGE_MAX_USD) when the job uses the API judge; 0 only when there are none')
    settle.add_argument('--launch-cap', type=float, help='ceiling approved for this one launch')
    settle.add_argument('--set', action='append', default=[], metavar='NAME=VALUE', dest='settings',
                        help='a driver setting for this launch (LEAN_ISOLATION=none, SMOKE_STEPS=4, FORWARD_JUDGE_KEY=1, ...); '
                             'the driver command must then carry exactly the settings given here')
    args = ap.parse_args()
    if args.cmd == 'hook':
        return hook()
    if not all('=' in setting for setting in args.settings):
        ap.error('--set takes NAME=VALUE')
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


if __name__ == '__main__':
    try:
        code = main()
    except Exception:
        # A gate that crashes must not wave the command through: say that the check itself failed, and refuse.
        traceback.print_exc()
        print('gate: the check itself failed (see above). This is not a finding about the command.', file=sys.stderr)
        sys.exit(2)
    sys.exit(code)
