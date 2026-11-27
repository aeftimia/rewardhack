"""Tests for what the second review round (2026-11-25) found in the pod scripts. Stubs only; nothing leaves this machine."""
import os
import signal
import subprocess
import time

import pytest
from conftest import LAUNCH_RECEIPT_ANSWER, SETSID, TIMEOUT_STUB, driver_script

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
    tmp_path.mkdir(parents=True, exist_ok=True)
    calls = tmp_path/'calls'
    calls.write_text('')
    ssh = f'''echo "ssh $*" >> {calls}
[[ -e {tmp_path}/removed ]] && exit 255
''' + LAUNCH_RECEIPT_ANSWER + f'''for arg in "$@"; do
  if [[ $arg == *"cat EXIT_CODE"* ]]; then echo 0; fi
  if [[ $arg == *pgrep* ]]; then exit {0 if timer_running else 1}; fi
  if [[ $arg == *--remove-now* ]]; then touch {tmp_path}/removed; fi
done
exit 0
'''
    rsync = f'''echo "rsync $*" >> {calls}
for arg in "$@"; do if [[ $arg == *:./ ]]; then exit {1 if pull_fails else 0}; fi; done
exit 0
'''
    env = dict(os.environ, PATH=write_stubs(tmp_path, ssh=ssh, rsync=rsync, sleep='exit 0\n'), JUDGE_KIND='pod', FORWARD_JUDGE_KEY='0', KEEP_POD='1' if keep_pod else '0',
               LEAN_ISOLATION='none')
    command = ['bash', driver_script(tmp_path), 'example.invalid', '22', 'ops/remote_runpod_smoke.sh', str(tmp_path/'results'), '240']
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
    code, calls, result = drive(tmp_path/'no-timer', pull_fails=True, timer_running=False)
    assert code == 97 and 'NO removal timer' in result.stdout


def test_a_kept_pod_without_a_timer_is_an_error(tmp_path):
    assert drive(tmp_path, keep_pod=True)[0] == 0
    code, calls, result = drive(tmp_path/'no-timer', keep_pod=True, timer_running=False)
    assert code == 97 and not any('--remove-now' in call for call in calls)
    # The check must not be satisfiable by the shell that runs it: its pattern cannot match its own command line.
    check = next(call for call in calls if 'pgrep' in call)
    assert "'[r]unpodctl remove pod'" in check


def test_cleanup_survives_a_closed_terminal(tmp_path):
    """With stdout gone every `echo` fails; under `set -e` that used to end the cleanup before the copy and the removal."""
    code, calls, result = drive(tmp_path, close_stdout=True)
    assert any('--remove-now' in call for call in calls), 'the pod was left running because a message could not be printed'
    assert any(call.startswith('rsync') and ':./' in call for call in calls)
    assert 'pod removed' in (tmp_path/'results'/'driver.log').read_text()


def test_a_terminated_runner_takes_its_job_with_it(tmp_path):
    path = write_stubs(tmp_path, timeout=TIMEOUT_STUB, setsid=SETSID)
    job = tmp_path/'job.sh'
    job.write_text(f'touch {tmp_path}/job-started\nsleep 300 & wait\n')
    pids = tmp_path/'pids'
    env = dict(os.environ, PATH=path, REMOTE_JOB_OUT=str(tmp_path/'out'), REMOTE_JOB_PIDS=str(pids), REQUIRE_AUTOSTOP='0')
    runner = subprocess.Popen(['bash', 'ops/remote_job.sh', str(job), '600'], env=env)
    try:
        # Wait for job readiness with room for contention from parallel workers.
        startup_end = time.monotonic() + 30
        while not (tmp_path/'job-started').exists() and runner.poll() is None and time.monotonic() < startup_end:
            time.sleep(0.05)
        assert (tmp_path/'job-started').exists(), 'runner never started its job'
        group = int(pids.read_text().strip())
        assert os.getsid(runner.pid) == group == runner.pid, 'the runner does not own the job session'
        runner.send_signal(signal.SIGTERM)
        assert runner.wait(timeout=10) == 143
        assert (tmp_path/'out'/'EXIT_CODE').read_text().strip() == '143'
        assert not pids.exists(), 'the pids file outlived the job it names'
        # A KILL to the whole group cannot be ignored, but the processes take a moment to go; a fixed 0.3 s failed 1 in 10
        # under the suite's load (26 Nov). Up to 5 s, then the group must be empty.
        for _ in range(50):
            if subprocess.run(['pgrep', '-g', str(group)], capture_output=True).stdout == b'':
                break
            time.sleep(0.1)
        assert subprocess.run(['pgrep', '-g', str(group)], capture_output=True).stdout == b'', 'the job outlived its runner'
    finally:
        if runner.poll() is None:
            runner.kill()
        subprocess.run(['pkill', '-9', '-g', str(group)] if 'group' in locals() else ['true'])


def test_a_timer_that_does_not_start_leaves_the_earlier_one_running(tmp_path, pod_id):
    ran = tmp_path/'job-ran'
    path = write_stubs(tmp_path, timeout=TIMEOUT_STUB, setsid=SETSID, runpodctl='exit 0\n', nohup='exit 1\n')
    pod = pod_id('podz')  # this test's own: parallel tests cannot touch its timers
    earlier = subprocess.Popen(['bash', '-c', f"sleep 300; runpodctl remove pod '{pod}'"])
    job = tmp_path/'job.sh'
    job.write_text(f'touch {ran}\n')
    env = dict(os.environ, PATH=path, REMOTE_JOB_OUT=str(tmp_path/'out'), REMOTE_JOB_PIDS=str(tmp_path/'pids'), RUNPOD_POD_ID=pod, RUNPOD_API_KEY='k',
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
    # Half the answers were cut off: the limit goes up. Two decided answers are no evidence of an undecided rate above
    # 5% either way, and the verifier check now needs such evidence (review round 16).
    assert limit == 4096 and check == 0 and 'tasks 1 of 2' in line
    decided = dict(row, truncated=False, format_valid=True)
    probe = dict(outcomes.summarize([decided]*48), tasks_planned=12, tasks_completed=12)
    assert probe_rules.decide(probe)[:2] == (2048, 0)  # none of 48 cut off or undecided: P(rate > 5%) = 0.03
    with pytest.raises(ValueError, match='undetermined'):
        probe_rules.decide(dict(outcomes.summarize([]), tasks_planned=0, tasks_completed=0))


def refused(tmp_path, results, **env):
    """The driver's own refusals, which must come before it contacts the pod at all."""
    calls = tmp_path/'calls'
    calls.write_text('')
    path = write_stubs(tmp_path, ssh=f'echo ssh >> {calls}\n', rsync=f'echo rsync >> {calls}\n', sleep='exit 0\n')
    base = dict(os.environ, PATH=path, JUDGE_KIND='pod', FORWARD_JUDGE_KEY='0', LEAN_ISOLATION='none')
    result = subprocess.run(['bash', driver_script(tmp_path), 'example.invalid', '22', 'ops/remote_runpod_smoke.sh', results, '240'],
                            env=dict(base, **env), capture_output=True, text=True)
    return result.returncode, result.stdout, calls.read_text()


def test_the_driver_refuses_what_would_only_fail_on_a_billing_pod(tmp_path):
    cases = [(str(tmp_path/'results with space'), {}, 'must not contain spaces'),
             (str(tmp_path/'r'), dict(SMOKE_SKIP_PROBE='1'), 'needs SMOKE_TOKEN_LIMIT'),
             (str(tmp_path/'r'), dict(AUTOSTOP_GRACE_SECONDS='36000'), 'at most 600')]
    for results, env, message in cases:
        code, output, calls = refused(tmp_path, results, **env)
        assert code == 2 and message in output, (env, output)
        assert calls == '', 'the driver contacted the pod before refusing'


def test_the_copy_to_the_pod_leaves_local_settings_and_tools_behind():
    line = next(line for line in open('ops/drive_runpod.sh') if line.startswith('rsync -az') and 'pilot-repo' in line)
    for kept_out in ('.git', '.tools', '.claude', 'spending'):
        assert f'--exclude {kept_out} ' in line, kept_out


def test_the_driver_can_check_a_command_without_a_pod(tmp_path):
    """`DRIVE_CHECK_ONLY=1` runs the refusals and nothing else, so they can be made before anything is deployed."""
    code, output, calls = refused(tmp_path, str(tmp_path/'r'), DRIVE_CHECK_ONLY='1')
    assert code == 0 and output.strip() == 'check passed' and calls == ''
    assert not (tmp_path/'r').exists(), 'a check created the results directory'
    code, output, calls = refused(tmp_path, str(tmp_path/'r'), DRIVE_CHECK_ONLY='1', SMOKE_SKIP_PROBE='1')
    assert code == 2 and 'check passed' not in output


def test_each_job_script_states_its_deadline():
    import re
    for script, minutes in (('ops/remote_runpod_smoke.sh', 60),):
        stated = re.search(r'^# deadline: (\d+)$', open(script).read(), re.M)
        assert stated and int(stated.group(1)) == minutes, script


def test_a_pod_judge_run_needs_its_longer_deadline(tmp_path):
    calls = tmp_path/'calls'
    calls.write_text('')
    path = write_stubs(tmp_path, ssh=f'echo ssh >> {calls}\n', rsync=f'echo rsync >> {calls}\n', sleep='exit 0\n')
    env = dict(os.environ, PATH=path, JUDGE_KIND='pod', FORWARD_JUDGE_KEY='0', LEAN_ISOLATION='none', DRIVE_CHECK_ONLY='1')
    def check(minutes):
        return subprocess.run(['bash', driver_script(tmp_path), 'example.invalid', '22', 'ops/remote_runpod_smoke.sh', str(tmp_path/'r'), minutes],
                              env=env, capture_output=True, text=True)
    short = check('170')
    assert short.returncode == 2 and 'at least 240' in short.stdout and calls.read_text() == ''
    assert check('240').returncode == 0
