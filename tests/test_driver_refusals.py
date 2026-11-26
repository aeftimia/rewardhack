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
    ('ops/remote_runpod_smoke.sh', dict(LEAN_ISOLATION='none', SMOKE_PART_ONE='4'), "SMOKE_PART_ONE must index a job in ops/pod_pins.env's PART_ONE_RUNS (0 to 3)"),
    # The gpt-oss-120b probe follows a Part 1 job's training, on its vLLM path (owner, 26 Nov; notes section 77).
    ('ops/remote_runpod_smoke.sh', dict(LEAN_ISOLATION='none', SMOKE_PROBE_LARGE='1'), 'SMOKE_PROBE_LARGE=1 runs after a Part 1 job'),
    ('ops/remote_runpod_smoke.sh', dict(LEAN_ISOLATION='none', SMOKE_PROBE_LARGE='2'), 'SMOKE_PROBE_LARGE must be 0 or 1'),
    # A kept pod's grace must hold the checkpoint copy, the next launch's sizing and gate, and the relaunch: up to 30 minutes
    # there, 10 otherwise (review round 7, 26 Nov).
    ('ops/remote_runpod_smoke.sh', dict(LEAN_ISOLATION='none', KEEP_POD='1', AUTOSTOP_GRACE_SECONDS='1801'), 'at most 1800'),
    ('ops/remote_runpod_smoke.sh', dict(LEAN_ISOLATION='none', AUTOSTOP_GRACE_SECONDS='1800'), 'at most 600'),
    ('ops/remote_runpod_smoke.sh', dict(LEAN_ISOLATION='none', KEEP_POD='yes'), 'KEEP_POD must be 0 or 1'),
    # Part 1 runs on vLLM rollouts and evaluations; without them it would silently use the trainer's sampler (round 16).
    ('ops/remote_runpod_smoke.sh', dict(LEAN_ISOLATION='none', SMOKE_PART_ONE='1', SMOKE_SKIP_PROBE='1', SMOKE_SECOND_RUN='0',
                                        SMOKE_STEPS='60', SMOKE_TRAIN_SECONDS='6000', SMOKE_FINAL_EVAL_SECONDS='600'), 'SMOKE_VLLM_ROLLOUTS=1'),
    # The vLLM server's context is sized to SMOKE_TOKEN_LIMIT; a memory step's longer answers could never be sampled (round 7).
    ('ops/remote_runpod_smoke.sh', dict(JUDGE_KIND='api', LEAN_ISOLATION='none', SMOKE_VLLM_ROLLOUTS='1', SMOKE_SKIP_PROBE='1',
                                        SMOKE_SECOND_RUN='0', SMOKE_MEMORY_LIMIT='4096'), 'not with SMOKE_VLLM_ROLLOUTS=1'),
    # The after-evaluation's reserve must be stated, and must leave room in each run's limit (review round 2, 26 Nov).
    ('ops/remote_runpod_smoke.sh', dict(LEAN_ISOLATION='none', SMOKE_PART_ONE='1', SMOKE_SKIP_PROBE='1', SMOKE_SECOND_RUN='0',
                                        SMOKE_STEPS='60', SMOKE_TRAIN_SECONDS='6000'), 'SMOKE_FINAL_EVAL_SECONDS set'),
    ('ops/remote_runpod_smoke.sh', dict(LEAN_ISOLATION='none', SMOKE_PART_ONE='1', SMOKE_VLLM_ROLLOUTS='1', SMOKE_SKIP_PROBE='1', SMOKE_SECOND_RUN='0',
                                        SMOKE_STEPS='60', SMOKE_TRAIN_SECONDS='6000', SMOKE_FINAL_EVAL_SECONDS='6000', JUDGE_KIND='api', FORWARD_JUDGE_KEY='1', OPENAI_API_KEY='placeholder-not-a-key'), 'below SMOKE_TRAIN_SECONDS'),
    # clean_env sets JUDGE_KIND=pod: Part 1 runs on vLLM rollouts, which need the API judge.
    ('ops/remote_runpod_smoke.sh', dict(LEAN_ISOLATION='none', SMOKE_PART_ONE='1', SMOKE_VLLM_ROLLOUTS='1', SMOKE_SKIP_PROBE='1', SMOKE_SECOND_RUN='0',
                                        SMOKE_STEPS='60', SMOKE_FINAL_EVAL_SECONDS='600', SMOKE_TRAIN_SECONDS='6000'), 'vLLM pilot needs API judge'),
    ('ops/remote_runpod_smoke.sh', dict(JUDGE_KIND='api', FORWARD_JUDGE_KEY='1', OPENAI_API_KEY='placeholder-not-a-key', LEAN_ISOLATION='none', SMOKE_PART_ONE='1', SMOKE_VLLM_ROLLOUTS='1', SMOKE_SKIP_PROBE='1', SMOKE_SECOND_RUN='0',
                                        SMOKE_STEPS='60', SMOKE_FINAL_EVAL_SECONDS='600', SMOKE_TRAIN_SECONDS='90m'), 'as whole numbers'),
    # Two runs of 110 minutes, each with a 13.5-minute scoring tail, after 30 minutes of setup need 277 minutes; the driver is given 240.
    ('ops/remote_runpod_smoke.sh', dict(JUDGE_KIND='api', FORWARD_JUDGE_KEY='1', OPENAI_API_KEY='placeholder-not-a-key', LEAN_ISOLATION='none', SMOKE_PART_ONE='1', SMOKE_VLLM_ROLLOUTS='1', SMOKE_SKIP_PROBE='1', SMOKE_SECOND_RUN='0',
                                        SMOKE_STEPS='60', SMOKE_FINAL_EVAL_SECONDS='600', SMOKE_TRAIN_SECONDS='6600'), 'needs a deadline of at least 277 minutes'),
    # One run of 160 minutes with its tail (173.5) after 30 minutes of setup, and the gpt-oss-120b probe's 2 x 1 + 15 + 30 + 2 x 5
    # minutes: 261 minutes; the driver is given 240. Without the probe the same run fits (the next test).
    ('ops/remote_runpod_smoke.sh', dict(JUDGE_KIND='api', FORWARD_JUDGE_KEY='1', OPENAI_API_KEY='placeholder-not-a-key', LEAN_ISOLATION='none', SMOKE_PART_ONE='2', SMOKE_PROBE_LARGE='1', SMOKE_VLLM_ROLLOUTS='1', SMOKE_SKIP_PROBE='1', SMOKE_SECOND_RUN='0',
                                        SMOKE_STEPS='60', SMOKE_FINAL_EVAL_SECONDS='600', SMOKE_TRAIN_SECONDS='9600'), 'SMOKE_PART_ONE=2 needs a deadline of at least 261 minutes'),
    # The probe-only job (no runs) exists for a relaunch of the probe; alone it would pay for setup and nothing else.
    ('ops/remote_runpod_smoke.sh', dict(LEAN_ISOLATION='none', SMOKE_PART_ONE='3'), 'it is only for SMOKE_PROBE_LARGE=1'),
    # Leading zeros: 00 once passed as a Part 1 job with no runs, 08 died on an octal error (review round 25).
    ('ops/remote_runpod_smoke.sh', dict(LEAN_ISOLATION='none', SMOKE_PART_ONE='00'), 'SMOKE_PART_ONE must index a job'),
    ('ops/remote_runpod_smoke.sh', dict(LEAN_ISOLATION='none', SMOKE_PART_ONE='08'), 'SMOKE_PART_ONE must index a job'),
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
                                  SMOKE_STEPS='60', SMOKE_FINAL_EVAL_SECONDS='600', SMOKE_TRAIN_SECONDS='5400', SMOKE_TOKEN_LIMIT='2048', SMOKE_VLLM_ROLLOUTS='1')  # 30 + 2 x (90 + 13.5) = 237 of 240 minutes
    assert code != 2 and 'invalid' not in output and 'does not read' not in output and reached


@pytest.mark.parametrize('extra', [{}, {'SMOKE_PROBE_LARGE': '1'}, {'SMOKE_PROBE_LARGE': '1', 'SMOKE_PART_ONE': '3'}])
def test_a_single_run_part_one_launch_passes_every_check(tmp_path, extra):
    """The λ=0.5 job (SMOKE_PART_ONE=2) is one run, so it fits a deadline the pair would not: 30 + 153.5 = 183.5 of 240
    minutes; with the gpt-oss-120b probe, 30 + (8340 + 810 + 3420) s = 239.5. Both settings reach the pod, as does the
    probe-only job (SMOKE_PART_ONE=3) with the probe."""
    settings = dict(JUDGE_KIND='api', FORWARD_JUDGE_KEY='1', OPENAI_API_KEY='placeholder-not-a-key',
                    LEAN_ISOLATION='none', SMOKE_PART_ONE='2', SMOKE_SKIP_PROBE='1', SMOKE_SECOND_RUN='0', SMOKE_STEPS='60',
                    SMOKE_FINAL_EVAL_SECONDS='600', SMOKE_TRAIN_SECONDS='8340', SMOKE_TOKEN_LIMIT='2048', SMOKE_VLLM_ROLLOUTS='1')
    code, output, reached = drive(tmp_path, **(settings | extra))
    assert code != 2 and 'invalid' not in output and 'does not read' not in output and reached


def test_a_pod_that_never_answered_is_not_reported_as_removed(tmp_path):
    """A pod whose SSH never came up never got a removal timer: it is still billing, so the driver must say so and exit
    97, not report it as gone on its timer (review round 4, 26 Nov). The stub ssh here never answers."""
    code, output, reached = drive(tmp_path, LEAN_ISOLATION='none', SMOKE_STEPS='4')
    assert reached and code == 97
    assert 'never accepted SSH' in output and 'removed itself' not in output
