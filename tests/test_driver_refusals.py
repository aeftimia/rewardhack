"""What the pod driver and runner refuse before anything is paid for: each case once cost a pod boot, or left billing unbounded."""
import os
import subprocess

import pytest
from conftest import driver_script

DROPPED = ('LEAN_ISOLATION', 'REQUIRE_AUTOSTOP', 'AUTOSTOP_GRACE_SECONDS', 'KEEP_POD')


def clean_env(tmp_path, **settings):
    """This machine's environment without the settings under test or any SMOKE_* one, with ssh, rsync and sleep stubbed and every ssh call recorded."""
    bin_dir = tmp_path/'bin'
    bin_dir.mkdir()
    stubs = {'ssh': f'echo "$*" >> {tmp_path}/ssh-calls\nexit 255\n', 'rsync': 'exit 1\n', 'sleep': 'exit 0\n',
             'timeout': 'shift 3\nexec "$@"\n', 'setsid': 'exec "$@"\n'}  # GNU timeout and setsid are not on macOS
    for name, body in stubs.items():
        (bin_dir/name).write_text('#!/bin/bash\n'+body)
        (bin_dir/name).chmod(0o755)
    kept = {k: v for k, v in os.environ.items() if k not in DROPPED and not k.startswith('SMOKE_')}
    return dict(kept, PATH=f"{bin_dir}:{os.environ['PATH']}", **{'JUDGE_KIND': 'pod', 'FORWARD_JUDGE_KEY': '0', **settings})  # a case may name its own judge


def drive(tmp_path, remote='ops/remote_runpod_smoke.sh', **settings):
    result = subprocess.run(['bash', driver_script(tmp_path), 'example.invalid', '22', remote, str(tmp_path/'results'), '240'],  # 240: the least the smoke script accepts with the judge served on the pod
                            env=clean_env(tmp_path, **settings), capture_output=True, text=True)
    return result.returncode, result.stdout+result.stderr, (tmp_path/'ssh-calls').exists()


@pytest.mark.parametrize('remote, settings, said', [
    ('ops/remote_runpod_smoke.sh', {}, 'LEAN_ISOLATION must be set'),
    ('ops/remote_runpod_smoke.sh', dict(LEAN_ISOLATION='none', SMOKE_TRAIN_STEPS='4'), 'SMOKE_TRAIN_STEPS is set, but'),
    ('ops/remote_runpod_genbench.sh', dict(SMOKE_STEPS='4'), 'reads no SMOKE_* setting'),
    ('ops/remote_runpod_smoke.sh', dict(LEAN_ISOLATION='none', AUTOSTOP_GRACE_SECONDS='0'), 'AUTOSTOP_GRACE_SECONDS must be'),
    ('ops/remote_runpod_smoke.sh', dict(LEAN_ISOLATION='none', AUTOSTOP_GRACE_SECONDS='179'), 'at least 180'),
    ('ops/remote_runpod_smoke.sh', dict(LEAN_ISOLATION='none', SMOKE_VLLM_ROLLOUTS='1', SMOKE_SKIP_PROBE='1', SMOKE_SECOND_RUN='0'), 'vLLM pilot needs API judge'),
    # Part 1 is the whole job: a probe, a further run or the preflight length and time limit are refused (26 Nov).
    ('ops/remote_runpod_smoke.sh', dict(LEAN_ISOLATION='none', SMOKE_PART_ONE='1', SMOKE_SKIP_PROBE='1', SMOKE_STEPS='60',
                                        SMOKE_TRAIN_SECONDS='6000'), 'SMOKE_PART_ONE=1 needs'),
    ('ops/remote_runpod_smoke.sh', dict(LEAN_ISOLATION='none', SMOKE_PART_ONE='1', SMOKE_SECOND_RUN='0', SMOKE_STEPS='60',
                                        SMOKE_TRAIN_SECONDS='6000'), 'SMOKE_PART_ONE=1 needs'),
    ('ops/remote_runpod_smoke.sh', dict(LEAN_ISOLATION='none', SMOKE_PART_ONE='1', SMOKE_SKIP_PROBE='1', SMOKE_SECOND_RUN='0',
                                        SMOKE_TRAIN_SECONDS='6000'), 'SMOKE_PART_ONE=1 needs'),
    ('ops/remote_runpod_smoke.sh', dict(LEAN_ISOLATION='none', SMOKE_PART_ONE='1', SMOKE_SKIP_PROBE='1', SMOKE_SECOND_RUN='0',
                                        SMOKE_STEPS='60', SMOKE_TRAIN_SECONDS='6000', SMOKE_MEMORY_LIMIT='4096'), 'SMOKE_PART_ONE=1 needs'),
    ('ops/remote_runpod_smoke.sh', dict(LEAN_ISOLATION='none', SMOKE_PART_ONE='2'), 'SMOKE_PART_ONE must be 0 or 1'),
    # The after-evaluation's reserve must be stated, and must leave room in each run's limit (review round 2, 26 Nov).
    ('ops/remote_runpod_smoke.sh', dict(LEAN_ISOLATION='none', SMOKE_PART_ONE='1', SMOKE_SKIP_PROBE='1', SMOKE_SECOND_RUN='0',
                                        SMOKE_STEPS='60', SMOKE_TRAIN_SECONDS='6000'), 'SMOKE_FINAL_EVAL_SECONDS set'),
    ('ops/remote_runpod_smoke.sh', dict(LEAN_ISOLATION='none', SMOKE_PART_ONE='1', SMOKE_SKIP_PROBE='1', SMOKE_SECOND_RUN='0',
                                        SMOKE_STEPS='60', SMOKE_TRAIN_SECONDS='6000', SMOKE_FINAL_EVAL_SECONDS='6000', JUDGE_KIND='api', FORWARD_JUDGE_KEY='1', OPENAI_API_KEY='placeholder-not-a-key'), 'below SMOKE_TRAIN_SECONDS'),
    # clean_env sets JUDGE_KIND=pod: Part 1's setup allowance is the API judge's.
    ('ops/remote_runpod_smoke.sh', dict(LEAN_ISOLATION='none', SMOKE_PART_ONE='1', SMOKE_SKIP_PROBE='1', SMOKE_SECOND_RUN='0',
                                        SMOKE_STEPS='60', SMOKE_FINAL_EVAL_SECONDS='600', SMOKE_TRAIN_SECONDS='6000'), 'is sized for JUDGE_KIND=api'),
    ('ops/remote_runpod_smoke.sh', dict(JUDGE_KIND='api', FORWARD_JUDGE_KEY='1', OPENAI_API_KEY='placeholder-not-a-key', LEAN_ISOLATION='none', SMOKE_PART_ONE='1', SMOKE_SKIP_PROBE='1', SMOKE_SECOND_RUN='0',
                                        SMOKE_STEPS='60', SMOKE_FINAL_EVAL_SECONDS='600', SMOKE_TRAIN_SECONDS='90m'), 'as whole numbers'),
    # Two runs of 110 minutes after 30 minutes of setup need 250 minutes; the driver is given 240.
    ('ops/remote_runpod_smoke.sh', dict(JUDGE_KIND='api', FORWARD_JUDGE_KEY='1', OPENAI_API_KEY='placeholder-not-a-key', LEAN_ISOLATION='none', SMOKE_PART_ONE='1', SMOKE_SKIP_PROBE='1', SMOKE_SECOND_RUN='0',
                                        SMOKE_STEPS='60', SMOKE_FINAL_EVAL_SECONDS='600', SMOKE_TRAIN_SECONDS='6600'), 'needs a deadline of at least 250 minutes'),
])
def test_the_driver_refuses_before_it_reaches_for_the_pod(tmp_path, remote, settings, said):
    code, output, reached = drive(tmp_path, remote, **settings)
    assert code == 2 and said in output and not reached


def test_a_setting_the_script_reads_is_forwarded(tmp_path):
    code, output, reached = drive(tmp_path, LEAN_ISOLATION='none', SMOKE_STEPS='4', AUTOSTOP_GRACE_SECONDS='180')
    assert code != 2 and 'does not read' not in output and reached  # it goes on to the pod, which this stub never lets answer


def test_the_runner_needs_the_billing_cap_decided_by_name(tmp_path):
    job = tmp_path/'job.sh'
    job.write_text('exit 0\n')
    env = dict(clean_env(tmp_path), REMOTE_JOB_OUT=str(tmp_path/'out'), REMOTE_JOB_PIDS=str(tmp_path/'pids'))
    refused = subprocess.run(['bash', 'ops/remote_job.sh', str(job), '10'], env=env, capture_output=True, text=True)
    assert refused.returncode == 2 and 'REQUIRE_AUTOSTOP must be set' in refused.stdout+refused.stderr
    ran = subprocess.run(['bash', 'ops/remote_job.sh', str(job), '10'], env=dict(env, REQUIRE_AUTOSTOP='0'), capture_output=True, text=True)
    assert ran.returncode == 0 and 'NO removal timer' in ran.stdout+(tmp_path/'out'/'autostop.log').read_text()


def test_a_complete_part_one_launch_passes_every_check(tmp_path):
    """Only refusals were tested once, and a name the driver could not forward (SMOKE_PART1: a digit) refused the real
    launch for the wrong reason. The planned settings must reach the pod."""
    code, output, reached = drive(tmp_path, JUDGE_KIND='api', FORWARD_JUDGE_KEY='1', OPENAI_API_KEY='placeholder-not-a-key',
                                  LEAN_ISOLATION='none', SMOKE_PART_ONE='1', SMOKE_SKIP_PROBE='1', SMOKE_SECOND_RUN='0',
                                  SMOKE_STEPS='60', SMOKE_FINAL_EVAL_SECONDS='600', SMOKE_TRAIN_SECONDS='6000', SMOKE_TOKEN_LIMIT='2048')
    assert code != 2 and 'invalid' not in output and 'does not read' not in output and reached
