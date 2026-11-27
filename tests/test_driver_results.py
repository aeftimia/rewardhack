"""Local result ownership must be settled before consuming a launch or contacting a pod."""
import os
import subprocess
from pathlib import Path

import pytest

from conftest import driver_script


def command(tmp_path, destination):
    driver = driver_script(tmp_path)
    gate_calls = tmp_path/'gate-calls'
    Path(driver).with_name('gate.sh').write_text(
        '#!/bin/bash\nprintf "launch\\n" >> "$GATE_CALLS"\nexit 2\n')
    env = dict(os.environ, JUDGE_KIND='pod', FORWARD_JUDGE_KEY='0', LEAN_ISOLATION='none',
               GATE_CALLS=str(gate_calls), DRIVE_CHECK_ONLY='0')
    return ['bash', driver, 'example.invalid', '22', 'ops/remote_runpod_smoke.sh', str(destination), '240'], env, gate_calls


@pytest.mark.parametrize('kind', ['previous-run', 'empty-directory', 'file', 'broken-link'])
def test_existing_results_are_refused_before_the_launch_gate(tmp_path, kind):
    destination = tmp_path/'results'
    if kind in ('previous-run', 'empty-directory'):
        destination.mkdir()
        if kind == 'previous-run':
            (destination/'launch-commit.txt').write_text('earlier commit\n')
            (destination/'rows.jsonl').write_text('earlier rows\n')
    elif kind == 'file':
        destination.write_text('occupied\n')
    else:
        destination.symlink_to(tmp_path/'missing')
    args, env, gate_calls = command(tmp_path, destination)
    result = subprocess.run(args, env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 2 and 'must be new' in result.stdout
    assert not gate_calls.exists()
    if kind == 'previous-run':
        assert (destination/'launch-commit.txt').read_text() == 'earlier commit\n'
        assert (destination/'rows.jsonl').read_text() == 'earlier rows\n'


def test_only_one_driver_can_reserve_the_results_directory(tmp_path):
    destination = tmp_path/'nested'/'results'
    args, env, gate_calls = command(tmp_path, destination)
    processes = [subprocess.Popen(args, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                 for _ in range(2)]
    for process in processes:
        process.communicate(timeout=30)
        assert process.returncode == 2
    assert gate_calls.read_text().splitlines() == ['launch']
    assert destination.is_dir() and list(destination.iterdir()) == []


def test_check_only_does_not_reserve_results(tmp_path):
    destination = tmp_path/'results'
    args, env, gate_calls = command(tmp_path, destination)
    result = subprocess.run(args, env=dict(env, DRIVE_CHECK_ONLY='1'), capture_output=True, text=True, timeout=30)
    assert result.returncode == 0 and result.stdout.strip() == 'check passed'
    assert not destination.exists() and not gate_calls.exists()
