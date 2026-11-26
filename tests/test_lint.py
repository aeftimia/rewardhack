"""Deterministic first pass of the no-silent-fallbacks rules: ruff on the Python (rules in pyproject.toml) and shellcheck on
the shell scripts in ops/. Run inside the test suite, so the launch gate's test run includes it. Both tools are pinned in
the `dev` extra; a missing tool fails here rather than passing unchecked."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHELL_SCRIPTS = sorted([*(ROOT/'ops').glob('*.sh'), *(ROOT/'ops'/'githooks').iterdir()])


def run(command):
    return subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=120)


def test_ruff_finds_nothing():
    done = run([sys.executable, '-m', 'ruff', 'check', '--no-cache', '.'])
    assert done.returncode == 0, done.stdout + done.stderr


def test_shellcheck_finds_nothing():
    shellcheck = Path(sys.executable).parent/'shellcheck'
    assert shellcheck.exists(), f'no {shellcheck}: install the dev extra (pip install -e .[dev])'
    done = run([str(shellcheck), '--format=gcc', '-x', *map(str, SHELL_SCRIPTS)])  # -x: follow ops/pod_pins.env, sourced from the root
    assert done.returncode == 0, done.stdout + done.stderr
