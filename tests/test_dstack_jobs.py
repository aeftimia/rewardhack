"""The dstack prototype of the pod job path, as far as it can be tested without a provider: no dstack, no RunPod, no pod."""
import json
import os
import re
import subprocess
from pathlib import Path

import pytest
import yaml

from rewardhack import dstack_jobs, gates
from test_gates import repo, run, settle  # noqa: F401 (repo is a fixture)

PROJECT = Path(__file__).resolve().parent.parent
SMOKE = dict(LEAN_ISOLATION='none', FORWARD_JUDGE_KEY='1', JUDGE_MAX_USD='1')


def record(root, deadline=170, basis='2x measured 85 min', hourly=2.70):
    (root/gates.RECORD).parent.mkdir(parents=True, exist_ok=True)
    (root/gates.RECORD).write_text(json.dumps(dict(basis=basis, hourly_usd=hourly, deadline_minutes=deadline)))


@pytest.mark.parametrize('job', sorted(dstack_jobs.JOBS))
@pytest.mark.parametrize('results, volume', [('volume', 'rh-results'), ('hold', None)])
def test_each_configuration_parses_as_the_yaml_dstack_reads(tmp_path, job, results, volume):
    run(tmp_path, 'init', '-q')
    run(tmp_path, '-c', 'user.email=t@example.com', '-c', 'user.name=T', 'commit', '-q', '--allow-empty', '-m', 'x')
    record(tmp_path)
    config, text = dstack_jobs.configuration(tmp_path, job, 170, SMOKE, results, volume)
    assert yaml.safe_load(text) == config
    assert config['type'] == 'task' and config['backends'] == ['runpod'] and config['retry'] is False
    assert config['spot_policy'] == 'on-demand' and config['max_price'] == 2.70 and config['python'] == '3.12'
    assert config['commands'] == [f"bash ops/dstack_job.sh {dstack_jobs.JOBS[job]['remote']} {170*60}"]
    assert ('volumes' in config) == (results == 'volume')
    assert 'OPENAI_API_KEY' in config['env']  # by name only: dstack takes the value from the launcher's environment
    assert not [line for line in config['env'] if line.startswith('OPENAI_API_KEY=')]


@pytest.mark.parametrize('deadline, basis, grace, results', [
    (170, '2x measured 85 min', 600, 'volume'), (65, '3x estimated 13 min', 180, 'hold'), (240, '2x measured 2 h', 600, 'hold'),
])
def test_max_duration_is_the_deadline_its_basis_sized_plus_stated_allowances(tmp_path, deadline, basis, grace, results):
    run(tmp_path, 'init', '-q')
    run(tmp_path, '-c', 'user.email=t@example.com', '-c', 'user.name=T', 'commit', '-q', '--allow-empty', '-m', 'x')
    record(tmp_path, deadline, basis)
    settings = dict(SMOKE, AUTOSTOP_GRACE_SECONDS=str(grace))
    config, text = dstack_jobs.configuration(tmp_path, 'smoke', deadline, settings, results, 'rh' if results == 'volume' else None)
    minutes = int(re.fullmatch(r'(\d+)m', config['max_duration']).group(1))
    hold = {'volume': 0, 'hold': grace}[results]
    assert minutes == deadline + dstack_jobs.AFTER_DEADLINE_MINUTES + -(-hold // 60)
    assert minutes - dstack_jobs.AFTER_DEADLINE_MINUTES - -(-hold // 60) >= gates.basis_minutes(basis)
    assert basis in text  # the basis is stated beside the number
    # After the job starts the pod's own timer (deadline + 30 s + grace) and dstack's stop are both inside the worst case.
    worst = dstack_jobs.worst_case_minutes(deadline, grace, hold)
    assert worst >= dstack_jobs.PROVISIONING_MINUTES + deadline + (30 + grace) / 60
    assert worst >= dstack_jobs.PROVISIONING_MINUTES + minutes + dstack_jobs.STOP_MINUTES + dstack_jobs.SERVER_BACKSTOP_MINUTES


def test_the_defaults_written_into_the_task_are_the_bash_drivers():
    """Read from the line where the driver hands each setting to the pod: `NAME=${NAME:-value}`."""
    driver = (PROJECT/'ops'/'drive_runpod.sh').read_text()
    for name, value in dstack_jobs.DRIVER_DEFAULTS.items():
        assert re.findall(rf'\b{name}=\${{{name}:-([^}}]*)}}', driver) == [value], name


def test_settings_the_pod_would_not_use_are_refused():
    with pytest.raises(ValueError, match='KEEP_POD'):
        dstack_jobs.driver_env(dict(KEEP_POD='1'))
    with pytest.raises(ValueError, match='no dstack job'):
        dstack_jobs.task(PROJECT, 'other', 60, {}, 2.7, 'hold', None)
    with pytest.raises(ValueError, match='volume name'):
        dstack_jobs.task(PROJECT, 'smoke', 60, {}, 2.7, 'volume', None)


def check(root, environ=None, **settings):
    return dstack_jobs.launch_check(root, 'smoke', 60, settings, 'volume', 'rh', environ if environ is not None else {})[0]


def test_the_launcher_refuses_without_a_launch_record(repo):
    refused = '\n'.join(check(repo))
    assert 'no launch record' in refused and 'no remote-tracking branch' in refused


def pushed(repo, tmp_path):
    run(tmp_path, 'init', '-q', '--bare', 'origin.git')
    run(repo, 'remote', 'add', 'origin', str(tmp_path/'origin.git'))
    run(repo, 'push', '-q', 'origin', 'main')


def test_the_launcher_refuses_a_launch_the_record_does_not_cover(repo, tmp_path):
    pushed(repo, tmp_path)
    assert settle(repo, settings=dict(LEAN_ISOLATION='none'))[1] == []
    assert 'not the ones that were checked' in '\n'.join(check(repo, LEAN_ISOLATION='none', SMOKE_STEPS='4'))
    assert '60 min deadline, not 61' in '\n'.join(dstack_jobs.launch_check(repo, 'smoke', 61, dict(LEAN_ISOLATION='none'), 'volume', 'rh', {})[0])
    gates.mark_launched(repo)
    assert 'already started a launch' in '\n'.join(check(repo, LEAN_ISOLATION='none'))


def test_the_launcher_refuses_a_commit_dstack_cannot_clone(repo, tmp_path):
    pushed(repo, tmp_path)
    run(repo, 'commit', '-q', '--allow-empty', '-m', 'not pushed')
    assert settle(repo, settings=dict(LEAN_ISOLATION='none'))[1] == []
    refused = check(repo, LEAN_ISOLATION='none')
    assert len(refused) == 1 and 'no remote-tracking branch' in refused[0]


def test_dstacks_worst_case_must_be_inside_the_budget_checks(repo, tmp_path):
    """gates.py counts the bash driver's overhead (15 min provisional timer plus the grace) and not remote_job.sh's 30 s kill-after,
    so as it stands every dstack launch is refused by one minute; with a record that counts it, the launch goes through."""
    pushed(repo, tmp_path)
    assert settle(repo, settings=dict(LEAN_ISOLATION='none'))[1] == []
    assert 'under dstack this launch can bill 86 min, and the budget check counted 85' in check(repo, LEAN_ISOLATION='none')[0]
    written = json.loads((repo/gates.RECORD).read_text())
    (repo/gates.RECORD).write_text(json.dumps(dict(written, worst_case_minutes=86)))
    problems, config, text = dstack_jobs.launch_check(repo, 'smoke', 60, dict(LEAN_ISOLATION='none'), 'volume', 'rh', {})
    assert problems == [] and config['max_duration'] == '62m' and yaml.safe_load(text) == config


def test_the_judge_key_must_be_in_the_launchers_environment_when_it_is_forwarded(repo, tmp_path):
    pushed(repo, tmp_path)
    settings = dict(LEAN_ISOLATION='none', FORWARD_JUDGE_KEY='1')
    assert settle(repo, settings=settings, other_usd=1.0)[1] == []
    written = json.loads((repo/gates.RECORD).read_text())
    (repo/gates.RECORD).write_text(json.dumps(dict(written, worst_case_minutes=86)))
    assert 'OPENAI_API_KEY is not set' in '\n'.join(check(repo, **settings))
    assert check(repo, dict(OPENAI_API_KEY='placeholder'), **settings) == []


# --- ops/dstack_job.sh, with remote_job.sh replaced by a stub that records how it was called ------------------------------------

def pod(tmp_path, **overrides):
    tree = tmp_path/'checkout'
    (tree/'ops').mkdir(parents=True)
    (tree/'ops'/'dstack_job.sh').write_text((PROJECT/'ops'/'dstack_job.sh').read_text())
    (tree/'ops'/'remote_runpod_smoke.sh').write_text('exit 0\n')
    (tree/'ops'/'remote_job.sh').write_text(
        f'echo "$* REQUIRE_AUTOSTOP=${{REQUIRE_AUTOSTOP:-unset}} KEY=${{OPENAI_API_KEY:-unset}}" >> {tmp_path}/calls\n'
        '[[ $1 == --arm-only ]] || exit 7\n')
    home = tmp_path/'home'
    home.mkdir()
    env = dict(PATH=os.environ['PATH'], HOME=str(home), JUDGE_KIND='api', LEAN_ISOLATION='none', JUDGE_MAX_USD='1', AUTOSTOP_GRACE_SECONDS='600',
               RESULTS_MODE='hold', RESULTS_HOLD_SECONDS='0', DSTACK_REPO_DIR=str(tree), DSTACK_RUN_NAME='rh-smoke-abc', FORWARD_JUDGE_KEY='1',
               OPENAI_API_KEY='placeholder-key')
    env.update(overrides)
    done = subprocess.run(['bash', 'ops/dstack_job.sh', 'ops/remote_runpod_smoke.sh', '3600'], cwd=tree, env=env, capture_output=True, text=True)
    calls = (tmp_path/'calls').read_text().splitlines() if (tmp_path/'calls').exists() else []
    return done, calls, home


def test_the_pod_command_arms_removal_then_starts_the_runner_with_the_cap_required(tmp_path):
    done, calls, home = pod(tmp_path)
    assert done.returncode == 7  # the runner's exit status is the job's
    assert calls == ['--arm-only 900 REQUIRE_AUTOSTOP=unset KEY=unset',
                     'ops/remote_runpod_smoke.sh 3600 REQUIRE_AUTOSTOP=1 KEY=unset']  # the key reaches neither through the environment
    assert (home/'.judge_key').read_text() == 'placeholder-key' and (home/'.judge_key').stat().st_mode & 0o777 == 0o600
    assert (home/'pilot-repo').resolve() == (tmp_path/'checkout').resolve() and (home/'out'/'remote.log').exists()


@pytest.mark.parametrize('unset', ['LEAN_ISOLATION', 'AUTOSTOP_GRACE_SECONDS', 'RESULTS_MODE', 'DSTACK_REPO_DIR'])
def test_the_pod_command_starts_nothing_without_every_setting_named(tmp_path, unset):
    done, calls, home = pod(tmp_path, **{unset: ''})
    assert done.returncode == 2 and f'{unset} is not set' in done.stdout and calls == []


def test_the_pod_command_refuses_an_unknown_results_mode_and_a_missing_key(tmp_path):
    done, calls, home = pod(tmp_path/'a', RESULTS_MODE='disk')
    assert done.returncode == 2 and 'RESULTS_MODE must be volume or hold' in done.stdout and calls == []
    done, calls, home = pod(tmp_path/'b', OPENAI_API_KEY='')
    assert done.returncode == 2 and 'carries no OPENAI_API_KEY' in done.stdout and calls == ['--arm-only 900 REQUIRE_AUTOSTOP=unset KEY=unset']
