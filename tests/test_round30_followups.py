"""Review round 30's minor follow-ups (notes section 77): a probe stage's failure says what its summary says, and a
model download is retried once within its limit. The gate's worst case is in test_gates.py, the runner's pids file in
the relaunch and termination tests."""
import json
import os
import subprocess
import sys
from pathlib import Path

from conftest import TIMEOUT_STUB, driver_script
from rewardhack import probe_rules

ROOT = Path(__file__).resolve().parents[1]


def test_a_failed_stage_says_whether_it_was_stopped_at_its_limit_and_what_it_kept(tmp_path):
    summary = tmp_path/'summary.json'
    assert 'ended before writing one' in probe_rules.stage(summary)
    summary.write_text(json.dumps(dict(status='running', argv=[])))
    assert 'rows written is unknown' in probe_rules.stage(summary)
    summary.write_text(json.dumps(dict(status='failed', rows_written=240, error='HardDeadline: hard deadline')))
    assert probe_rules.stage(summary) == 'status=failed rows_written=240 stopped at its --max-seconds: HardDeadline: hard deadline'
    summary.write_text(json.dumps(dict(status='failed', rows_written=0, error='RuntimeError: CUDA out of memory')))
    assert 'ended by an error: RuntimeError' in probe_rules.stage(summary)
    summary.write_text(json.dumps(dict(status='completed', rows_written=320)))
    assert probe_rules.stage(summary).endswith('no error recorded')
    done = subprocess.run([sys.executable, '-m', 'rewardhack.probe_rules', 'stage', str(summary)], capture_output=True, text=True, check=True,
                          env=dict(os.environ, PYTHONPATH=str(ROOT/'src')))
    assert done.stdout.strip() == probe_rules.stage(summary)


def fetch(tmp_path, hf_body, wait):
    """fetch_model from ops/pod_lib.sh with a stub `hf`; returns (exit status, the job's log lines)."""
    hf = tmp_path/'hf'
    hf.write_text('#!/bin/bash\n' + hf_body)
    hf.chmod(0o755)
    (tmp_path/'out').mkdir(exist_ok=True)
    script = f'source ops/pod_lib.sh; log() {{ echo "$*" >> {tmp_path}/steps.log; }}; HF={hf}; DOWNLOAD_WAIT={wait} fetch_model org/model rev'
    done = subprocess.run(['bash', '-c', script], cwd=ROOT, env=dict(os.environ, HOME=str(tmp_path)), capture_output=True, text=True)
    steps = (tmp_path/'steps.log').read_text().splitlines() if (tmp_path/'steps.log').exists() else []
    return done.returncode, steps


def test_a_failed_download_is_retried_once_within_its_limit(tmp_path):
    count = tmp_path/'attempts'
    flaky = f'echo attempt >> {count}\n[[ $(wc -l < {count}) -ge 2 ]] || {{ echo "File reconstruction error"; exit 1; }}\n'
    code, steps = fetch(tmp_path, flaky, 60)
    assert code == 0 and len(count.read_text().splitlines()) == 2
    assert len(steps) == 1 and steps[0].startswith('retry download org/model')
    assert 'File reconstruction error' in (tmp_path/'out'/'download-model.log').read_text()  # the first attempt's log is kept
    count.unlink()
    code, _ = fetch(tmp_path, f'echo attempt >> {count}\nexit 1\n', 60)
    assert code == 1 and len(count.read_text().splitlines()) == 2  # once, not more


def test_a_download_that_used_its_whole_limit_is_not_retried(tmp_path):
    count = tmp_path/'attempts'
    code, steps = fetch(tmp_path, f'echo attempt >> {count}\nsleep 30\n', 5)
    assert code != 0 and len(count.read_text().splitlines()) == 1
    assert steps == ['download org/model used its whole limit of 5 s (see ' + str(tmp_path) + '/out/download-model.log)']


def test_a_timer_that_dies_after_a_slow_start_is_not_taken_for_a_running_one(tmp_path, pod_id):
    """The 26 Nov flake: `arm` took a timer for running if its process was alive one second after starting. Under the full
    suite's load the stub that fails at once was still starting then, and the job started with no timer. A timer that
    lives 1.5 s and never runs the removal loop must be refused however long it takes to start (review round 33)."""
    bin_dir = tmp_path/'bin'
    bin_dir.mkdir()
    stubs = {'timeout': TIMEOUT_STUB, 'setsid': 'exec "$@"\n', 'runpodctl': 'exit 0\n', 'nohup': '/bin/sleep 1.5\nexit 1\n'}
    for name, body in stubs.items():
        (bin_dir/name).write_text('#!/bin/bash\n' + body)
        (bin_dir/name).chmod(0o755)
    ran = tmp_path/'ran'
    job = tmp_path/'job.sh'
    job.write_text(f'touch {ran}\n')
    env = dict(os.environ, PATH=f"{bin_dir}:{os.environ['PATH']}", REMOTE_JOB_OUT=str(tmp_path/'out'), REMOTE_JOB_PIDS=str(tmp_path/'pids'),
               RUNPOD_POD_ID=pod_id('podslow'), RUNPOD_API_KEY='k', REQUIRE_AUTOSTOP='1')
    result = subprocess.run(['bash', 'ops/remote_job.sh', str(job), '10'], cwd=ROOT, env=env, capture_output=True, text=True, timeout=60)
    assert result.returncode == 3 and 'did not start' in result.stdout, result.stdout + result.stderr
    assert not ran.exists(), 'the job started with no working timer'


def test_the_driver_refuses_at_once_without_a_timeout_command(tmp_path):
    """Every call that could hold a billing pod runs under `timeout`; without one each would fail as "not found" and the
    pod would be given up on with no removal timer (review round 36). Nothing but `dirname` is on this PATH."""
    bin_dir = tmp_path/'bin'
    bin_dir.mkdir()
    (bin_dir/'dirname').symlink_to(subprocess.run(['/usr/bin/which', 'dirname'], capture_output=True, text=True, check=True).stdout.strip())
    done = subprocess.run(['/bin/bash', 'ops/drive_runpod.sh', 'example.invalid', '22', 'ops/remote_runpod_smoke.sh', 'results/x', '60'],
                          cwd=ROOT, env=dict(PATH=str(bin_dir), HOME=str(tmp_path)), capture_output=True, text=True, timeout=30)
    assert done.returncode == 2 and 'needs a timeout command' in done.stdout, done.stdout + done.stderr


def arm_fails(tmp_path, out_on_pod, fails='--arm-only', rsync=1, keep='1'):
    """The driver against a pod that answers at once and whose ssh call containing `fails` fails (by default the arm
    call); `out_on_pod`: whether a job ever ran there; `rsync`: the exit status of every copy. Returns the driver's
    output and the ssh calls it made, in order."""
    calls = tmp_path/'calls'
    bin_dir = tmp_path/'bin'
    bin_dir.mkdir(parents=True)
    ssh = (f'echo "$*" >> {calls}\n[[ "$*" != *{fails}* ]] || exit 255\n'
           f'if [[ "$*" == *out/remote.log* ]]; then echo "Welcome to the pod"; echo {out_on_pod}; fi\n')  # a banner, then the answer to JOB_TRACES
    stubs = {'ssh': ssh, 'rsync': f'exit {rsync}\n', 'sleep': 'exit 0\n', 'timeout': TIMEOUT_STUB}
    for name, body in stubs.items():
        (bin_dir/name).write_text('#!/bin/bash\n' + body)
        (bin_dir/name).chmod(0o755)
    env = dict(os.environ, PATH=f"{bin_dir}:{os.environ['PATH']}", JUDGE_KIND='pod', FORWARD_JUDGE_KEY='0', LEAN_ISOLATION='none', KEEP_POD=keep)
    done = subprocess.run(['bash', driver_script(tmp_path), 'example.invalid', '22', 'ops/remote_runpod_smoke.sh', str(tmp_path/'results'), '240'],
                          env=env, capture_output=True, text=True, timeout=120)
    assert done.returncode != 0, done.stdout + done.stderr
    return done.stdout, calls.read_text().splitlines()


def test_a_wait_that_connected_makes_no_further_unretried_try_and_a_failed_arm_removes_only_an_empty_pod(tmp_path):
    """Round 37: after the wait loop connected, one more connection was made with no retry, and its failure gave up a pod
    with no removal timer. Round 39: a pod this run armed no timer on started no job here, so a fresh one is removed even
    with KEEP_POD=1. Round 40: a kept pod (a relaunch) may hold an earlier job's out/ and timer, and is not removed."""
    output, sent = arm_fails(tmp_path/'fresh', 'empty')
    before_arm = sent[:next(index for index, line in enumerate(sent) if '--arm-only' in line)]
    assert len(before_arm) == 1 and before_arm[0].endswith(' true'), sent
    assert any('--remove-now' in line for line in sent), sent
    assert 'nothing to copy' in output and 'FINAL COPY FAILED' not in output, output  # no results were there to lose
    output, sent = arm_fails(tmp_path/'kept', 'held')
    assert not any('--remove-now' in line for line in sent) and 'removal timer is running' in output, sent
    # This run launched no job, so the earlier job's results are not copied into its folder (counted twice) (round 43).
    assert 'not copied into this run' in output and 'copying results' not in output, output


def test_a_failed_arm_call_on_a_fresh_pod_leaves_no_trace_of_a_job(tmp_path):
    """Round 41: the runner creates out/ before arming, so `out/` was no sign of an earlier job and a fresh pod whose arm
    call failed was left billing with no timer. The driver's own check (JOB_TRACES), run where a real failed
    --arm-only ran, answers empty; a launched job's remote.log or a relaunch's out.prev.* answers held."""
    import re
    traces = re.search(r"^JOB_TRACES='([^']*)'$", (ROOT/'ops/drive_runpod.sh').read_text(), re.M).group(1)
    home = tmp_path/'home'
    home.mkdir()
    bin_dir = tmp_path/'bin'
    bin_dir.mkdir()
    (bin_dir/'runpodctl').write_text('#!/bin/bash\nexit 1\n')  # its key refused: this pod cannot remove itself
    (bin_dir/'runpodctl').chmod(0o755)
    env = dict(os.environ, PATH=f"{bin_dir}:{os.environ['PATH']}", HOME=str(home), RUNPOD_POD_ID='podfresh', RUNPOD_API_KEY='k')
    env.pop('REMOTE_JOB_OUT', None)
    with open(ROOT/'ops/remote_job.sh') as script:
        armed = subprocess.run(['bash', '-s', '--', '--arm-only', '900'], stdin=script, cwd=home, env=env, capture_output=True, text=True, timeout=60)
    assert armed.returncode == 3 and (home/'out').is_dir()  # failed, and out/ is there all the same

    def ask():
        return subprocess.run(['sh', '-c', traces], cwd=home, capture_output=True, text=True, check=True).stdout.strip()
    assert ask() == 'empty'
    (home/'out'/'remote.log').write_text('')
    assert ask() == 'held'
    (home/'out'/'remote.log').unlink()
    (home/'out.prev.1764100000').mkdir()
    assert ask() == 'held'


def test_a_run_that_armed_but_did_not_launch_copies_nothing_of_an_earlier_job(tmp_path):
    """Round 43/44: after arming, a failed sync (no launch sent) or a failed launch call (sent, unconfirmed) must not copy
    out/ into this run's folder, where an earlier job's rows would be counted twice; the pod keeps its timer, and is not
    removed with KEEP_POD=0, since what it holds was not copied (review round 45)."""
    output, sent = arm_fails(tmp_path/'sync', 'held', fails='--never--', rsync=1, keep='0')  # the sync is the first rsync
    assert "holds an earlier job's results" in output and 'copying results' not in output, output
    assert not any('--remove-now' in line for line in sent), sent
    output, sent = arm_fails(tmp_path/'launch', 'held', fails='nohup', rsync=0, keep='0')
    assert 'whether this run' in output and 'copying results' not in output, output
    assert not any('--remove-now' in line for line in sent), sent
