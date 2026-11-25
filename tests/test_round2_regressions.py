"""Tests for what the second review round (2026-11-25) found in the pod scripts. Stubs only; nothing leaves this machine."""
import os
import signal
import subprocess
import time

import pytest

from rewardhack import outcomes, probe_rules


def write_stubs(tmp_path, **bodies):
    bin_dir = tmp_path/'bin'
    bin_dir.mkdir(exist_ok=True)
    for name, body in bodies.items():
        (bin_dir/name).write_text(body if body.startswith('#!') else '#!/bin/bash\n'+body)
        (bin_dir/name).chmod(0o755)
    return str(bin_dir)+':'+os.environ['PATH']


def drive(tmp_path, *, pull_fails=False, timer_running=True, keep_pod=False, close_stdout=False):
    """Run the driver against a stub pod. Returns its exit status and the ordered list of what it sent."""
    calls = tmp_path/'calls'
    calls.write_text('')
    ssh = f'''echo "ssh $*" >> {calls}
[[ -e {tmp_path}/removed ]] && exit 255
for arg in "$@"; do
  if [[ $arg == "cat out/EXIT_CODE" ]]; then echo 0; fi
  if [[ $arg == *pgrep* ]]; then exit {0 if timer_running else 1}; fi
  if [[ $arg == *--remove-now* ]]; then touch {tmp_path}/removed; fi
done
exit 0
'''
    rsync = f'''echo "rsync $*" >> {calls}
for arg in "$@"; do if [[ $arg == *:out/ ]]; then exit {1 if pull_fails else 0}; fi; done
exit 0
'''
    env = dict(os.environ, PATH=write_stubs(tmp_path, ssh=ssh, rsync=rsync, sleep='exit 0\n'), JUDGE_KIND='pod', FORWARD_JUDGE_KEY='0', KEEP_POD='1' if keep_pod else '0')
    command = ['bash', 'ops/drive_runpod.sh', 'example.invalid', '22', 'ops/remote_runpod_smoke.sh', str(tmp_path/'results'), '1']
    if close_stdout:
        command = ['bash', '-c', 'exec "$@" >&-', 'driver']+command
    result = subprocess.run(command, env=env, capture_output=True, text=True)
    return result.returncode, calls.read_text().splitlines(), result


def test_the_driver_bounds_billing_before_it_copies_anything(tmp_path):
    code, calls, result = drive(tmp_path)
    assert code == 0, result.stdout+result.stderr
    arm = next(i for i, call in enumerate(calls) if '--arm-only 900' in call)
    assert arm < next(i for i, call in enumerate(calls) if call.startswith('rsync')), 'the repo was copied to a pod with no removal timer'
    assert any('--remove-now' in call for call in calls) and (tmp_path/'removed').exists()
    assert 'pod removed: it no longer answers' in (tmp_path/'results'/'driver.log').read_text()


def test_results_that_were_not_copied_are_not_deleted_with_the_pod(tmp_path):
    code, calls, result = drive(tmp_path, pull_fails=True)
    assert code == 96, result.stdout
    assert not any('--remove-now' in call for call in calls), 'the pod holding the only copy of the results was removed'
    assert 'a removal timer is running' in result.stdout
    # And if no timer is running there either, that is an alarm, not a reassurance.
    code, calls, result = drive(tmp_path, pull_fails=True, timer_running=False)
    assert code == 97 and 'NO removal timer' in result.stdout


def test_a_kept_pod_without_a_timer_is_an_error(tmp_path):
    assert drive(tmp_path, keep_pod=True)[0] == 0
    code, calls, result = drive(tmp_path, keep_pod=True, timer_running=False)
    assert code == 97 and not any('--remove-now' in call for call in calls)
    # The check must not be satisfiable by the shell that runs it: its pattern cannot match its own command line.
    check = next(call for call in calls if 'pgrep' in call)
    assert "'[r]unpodctl remove pod'" in check


def test_cleanup_survives_a_closed_terminal(tmp_path):
    """With stdout gone every `echo` fails; under `set -e` that used to end the cleanup before the copy and the removal."""
    code, calls, result = drive(tmp_path, close_stdout=True)
    assert any('--remove-now' in call for call in calls), 'the pod was left running because a message could not be printed'
    assert any(call.startswith('rsync') and ':out/' in call for call in calls)
    assert 'pod removed' in (tmp_path/'results'/'driver.log').read_text()


SETSID = '''#!/usr/bin/env python3
import os, sys
os.setsid()  # a real new session and process group, as util-linux setsid gives the job on the pod
os.execvp(sys.argv[1], sys.argv[1:])
'''


def test_a_terminated_runner_takes_its_job_with_it(tmp_path):
    path = write_stubs(tmp_path, timeout='shift 3\nexec "$@"\n', setsid=SETSID)
    job = tmp_path/'job.sh'
    job.write_text('sleep 300 & wait\n')
    pids = tmp_path/'pids'
    env = dict(os.environ, PATH=path, REMOTE_JOB_OUT=str(tmp_path/'out'), REMOTE_JOB_PIDS=str(pids))
    runner = subprocess.Popen(['bash', 'ops/remote_job.sh', str(job), '600'], env=env)
    try:
        for _ in range(100):
            if pids.exists() and len(pids.read_text().split()) == 2:
                break
            time.sleep(0.05)
        group = int(pids.read_text().split()[1])
        for _ in range(100):  # the pid is recorded before the job's `setsid` has run
            if os.getpgid(group) == group:
                break
            time.sleep(0.05)
        assert os.getpgid(group) == group != os.getpgid(runner.pid), 'the job is not in its own process group'
        runner.send_signal(signal.SIGTERM)
        assert runner.wait(timeout=10) == 143
        assert (tmp_path/'out'/'EXIT_CODE').read_text().strip() == '143'
        time.sleep(0.3)
        assert subprocess.run(['pgrep', '-g', str(group)], capture_output=True).stdout == b'', 'the job outlived its runner'
    finally:
        if runner.poll() is None:
            runner.kill()
        subprocess.run(['pkill', '-9', '-g', str(group)] if 'group' in locals() else ['true'])


def test_a_timer_that_does_not_start_leaves_the_earlier_one_running(tmp_path):
    ran = tmp_path/'job-ran'
    path = write_stubs(tmp_path, timeout='shift 3\nexec "$@"\n', setsid='exec "$@"\n', runpodctl='exit 0\n', nohup='exit 1\n')
    earlier = subprocess.Popen(['bash', '-c', "sleep 300; runpodctl remove pod 'podz'"])
    job = tmp_path/'job.sh'
    job.write_text(f'touch {ran}\n')
    env = dict(os.environ, PATH=path, REMOTE_JOB_OUT=str(tmp_path/'out'), REMOTE_JOB_PIDS=str(tmp_path/'pids'), RUNPOD_POD_ID='podz', RUNPOD_API_KEY='k',
               REQUIRE_AUTOSTOP='1')
    try:
        result = subprocess.run(['bash', 'ops/remote_job.sh', str(job), '10'], env=env, capture_output=True, text=True)
        assert result.returncode == 3 and 'did not start' in result.stdout, result.stdout+result.stderr
        assert earlier.poll() is None, 'the only working timer was stopped before its replacement was proven'
        assert not ran.exists(), 'the job started with no new timer'
    finally:
        earlier.kill()


def test_probe_rules_are_read_from_a_real_summary():
    row = outcomes.score('unused', dict(id='t', group='g', tier='easy', true=True, statement='True'), '', 'baseline', truncated=True)
    probe = dict(outcomes.summarize([row, dict(row, truncated=False)]), tasks_planned=2, tasks_completed=1)
    limit, check, line = probe_rules.decide(probe)
    assert limit == 4096 and check == 0 and 'tasks 1 of 2' in line  # half the answers were cut off: the limit goes up
    with pytest.raises(ValueError, match='undetermined'):
        probe_rules.decide(dict(outcomes.summarize([]), tasks_planned=0, tasks_completed=0))
