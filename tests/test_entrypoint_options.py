"""Every option the library reads from an entrypoint's arguments is one the entrypoint defines: the library has no defaults of its own for them."""
import runpy
import sys
from pathlib import Path
from types import ModuleType

import pytest

from rewardhack import judge

ROOT = Path(__file__).resolve().parents[1]
API = ['--model', 'stub', '--max-usd', '0']
VLLM = ['--lean', 'none', '--model', 'stub', '--revision', 'stub']


def parsed(monkeypatch, tmp_path, filename, module, arguments):
    """The namespace `filename` hands to `rewardhack.<module>.run`."""
    seen = []
    stub = ModuleType('rewardhack.'+module)
    stub.run = seen.append
    monkeypatch.setitem(sys.modules, stub.__name__, stub)
    monkeypatch.setattr(sys, 'argv', [filename, '--out', str(tmp_path), *arguments])
    runpy.run_path(str(ROOT/'entrypoints'/filename), run_name='__main__')
    assert len(seen) == 1
    return seen[0]


@pytest.mark.parametrize('given,expected', [([], {}), (['--extra-body', '{"chat_template_kwargs": {"enable_thinking": false}}'], {'chat_template_kwargs': {'enable_thinking': False}})])
def test_the_api_probe_defines_the_request_body_extras(monkeypatch, tmp_path, given, expected):
    assert parsed(monkeypatch, tmp_path, 'probe_api.py', 'openai_probe', API+given).extra_body == expected


@pytest.mark.parametrize('given,expected', [([], 'train'), (['--part', 'all'], 'all')])
def test_the_vllm_probe_defines_the_task_part(monkeypatch, tmp_path, given, expected):
    assert parsed(monkeypatch, tmp_path, 'probe_vllm.py', 'vllm_probe', VLLM+given).part == expected


def test_judge_validation_builds_its_judge_for_prose_reports(monkeypatch, tmp_path):
    built = []

    class Built(Exception):
        pass

    def from_args(args):
        built.append(args)
        raise Built
    monkeypatch.setattr(judge, 'from_args', from_args)
    monkeypatch.setattr(judge, 'read_set', lambda folder: [])
    monkeypatch.setattr(sys, 'argv', ['validate_judge.py', '--out', str(tmp_path/'out'), '--judge', 'triage', '--read', str(tmp_path)])
    with pytest.raises(Built):
        runpy.run_path(str(ROOT/'entrypoints'/'validate_judge.py'), run_name='__main__')
    assert (built[0].report_mode, built[0].judge) == ('prose', 'triage')
