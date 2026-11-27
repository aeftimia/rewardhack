"""The bounded copy phase refuses missing or unsuccessful predecessor output."""
import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('status', ['missing', 'failed', 'success'])
def test_archive_window_requires_completed_source_and_waits_for_receipt(tmp_path, status):
    out = tmp_path/'out'
    out.mkdir()
    (out/'LAUNCH_ID').write_text('new-launch')
    previous = tmp_path/'out.prev.new-launch'
    previous.mkdir()
    if status != 'missing':
        (previous/'DONE').touch()
        (previous/'EXIT_CODE').write_text('0' if status == 'success' else '1')
    bin_dir = tmp_path/'bin'
    bin_dir.mkdir()
    sleep = bin_dir/'sleep'
    sleep.write_text('#!/bin/sh\ntouch "$HOME/out/ARCHIVE_COPY_COMPLETE"\n')
    sleep.chmod(0o755)
    result = subprocess.run(['bash', str(ROOT/'ops/remote_archive_window.sh')],
                            env=dict(os.environ, HOME=str(tmp_path), PATH=f'{bin_dir}:{os.environ["PATH"]}'),
                            capture_output=True, text=True, timeout=10)
    if status == 'success':
        assert result.returncode == 0, result.stderr
        assert (out/'archive-source.txt').read_text().strip() == str(previous)
        assert 'local copy confirmed complete' in result.stdout
    else:
        assert result.returncode == 2
        assert 'successfully completed predecessor' in result.stdout
        assert not (out/'archive-source.txt').exists()
