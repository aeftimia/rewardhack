"""Every verifier-backed CLI accepts the same retry budget before starting work."""
import runpy
import sys
from pathlib import Path
from types import ModuleType

import pytest

from rewardhack.outcomes import RETRY_SECONDS

ROOT = Path(__file__).resolve().parents[1]
ENTRYPOINTS = (
    ('run.py', 'rewardhack.hf_grpo', ['probe']),
    ('local_mlx.py', 'rewardhack.mlx_rl', ['rl']),
    ('probe_vllm.py', 'rewardhack.vllm_probe', []),
)


def invoke(monkeypatch, tmp_path, entrypoint, extra):
    filename, module_name, positional = entrypoint
    calls = []
    module = ModuleType(module_name)
    module.run = calls.append
    # Parsing must not require MLX, transformers, GPUs, or model downloads.
    monkeypatch.setitem(sys.modules, module_name, module)
    monkeypatch.setattr(sys, 'argv', [filename, *positional, '--out', str(tmp_path/'unused'),
                                    '--lean', 'must-not-run', '--model', 'stub', '--revision', 'stub', *extra])
    return calls, lambda: runpy.run_path(str(ROOT/'entrypoints'/filename), run_name='__main__')


@pytest.mark.parametrize('entrypoint', ENTRYPOINTS)
@pytest.mark.parametrize('value', ['0', '60.5', '120'])
def test_retry_allowance_reaches_each_runner(monkeypatch, tmp_path, entrypoint, value):
    calls, main = invoke(monkeypatch, tmp_path, entrypoint, ['--verifier-retry-seconds', value])
    main()
    assert len(calls) == 1
    assert calls[0].verifier_retry_seconds == float(value)
    assert not (tmp_path/'unused').exists()


@pytest.mark.parametrize('entrypoint', ENTRYPOINTS)
def test_retry_allowance_default_is_shared(monkeypatch, tmp_path, entrypoint):
    calls, main = invoke(monkeypatch, tmp_path, entrypoint, [])
    main()
    assert calls[0].verifier_retry_seconds == RETRY_SECONDS


@pytest.mark.parametrize('entrypoint', ENTRYPOINTS)
@pytest.mark.parametrize('value', ['nan', 'inf', '-inf', '-1', 'not-a-number'])
def test_invalid_retry_allowance_refused_before_run(monkeypatch, tmp_path, capsys, entrypoint, value):
    calls, main = invoke(monkeypatch, tmp_path, entrypoint, [f'--verifier-retry-seconds={value}'])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert calls == []
    assert 'verifier retry seconds must be a finite, nonnegative number' in capsys.readouterr().err
    assert not (tmp_path/'unused').exists()
