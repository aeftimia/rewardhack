#!/usr/bin/env python3
"""Start a pod job through dstack instead of ops/drive_runpod.sh. A prototype: no paid run has used it yet.

    dstack_launch.py show smoke --deadline 170 --results volume --volume rh-results --set LEAN_ISOLATION=none ...
        Print the task configuration this launch would submit, from the launch record's price and cap basis. Contacts nothing.
    dstack_launch.py launch smoke --deadline 170 --results volume --volume rh-results --set LEAN_ISOLATION=none ...
        Refuse unless the launch record (entrypoints/gate.py ready) covers this remote script, deadline and settings, the
        commit is on a remote branch, and dstack's worst case is within the one the budget check counted; then mark the
        record used and submit the configuration with `dstack apply -f - -y -d`.

The cap that holds while this machine sleeps is the pod's own removal timer (ops/remote_job.sh); dstack's max_duration and
its removal of the pod when the job ends need the dstack server to be running. See docs/notes-2026-11-21.md, section 62.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

from rewardhack.dstack_jobs import JOBS, RESULTS, configuration, launch_check
from rewardhack.gates import mark_launched

ROOT = Path(__file__).resolve().parent.parent
# cap: 3x estimated 1 min (with -d, dstack apply returns once the server has accepted the run)
APPLY_SECONDS = 180


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('action', choices=['show', 'launch'])
    ap.add_argument('job', choices=sorted(JOBS))
    ap.add_argument('--deadline', type=int, required=True, help='job deadline in minutes, as given to gate.py ready')
    ap.add_argument('--results', required=True, choices=sorted(RESULTS),
                    help='volume: a RunPod network volume outlives the pod; hold: kept on the pod for the grace, to be copied by hand')
    ap.add_argument('--volume', help='the dstack volume name, with --results volume')
    ap.add_argument('--set', action='append', default=[], metavar='NAME=VALUE', dest='settings',
                    help='a driver setting, exactly as given to gate.py ready')
    args = ap.parse_args()
    if not all('=' in setting for setting in args.settings):
        ap.error('--set takes NAME=VALUE')
    settings = dict(setting.split('=', 1) for setting in args.settings)
    if args.action == 'show':
        print(configuration(ROOT, args.job, args.deadline, settings, args.results, args.volume)[1], end='')
        return 0
    problems, config, text = launch_check(ROOT, args.job, args.deadline, settings, args.results, args.volume, os.environ)
    for problem in problems:
        print('REFUSED:', problem)
    if problems:
        return 1
    mark_launched(ROOT)  # one record, one launch: its budget check counted one worst case
    done = subprocess.run(['dstack', 'apply', '-f', '-', '-y', '-d'], cwd=ROOT, input=text, text=True, timeout=APPLY_SECONDS)
    if done.returncode:
        print(f'dstack apply exited {done.returncode}: check `dstack ps` and the RunPod console now')
        return done.returncode
    print(f"submitted {config['name']} with max_duration {config['max_duration']}. Its first command arms the pod's own removal: "
          'read autostop.log in the results for the `armed:` line')
    return 0


if __name__ == '__main__':
    sys.exit(main())
