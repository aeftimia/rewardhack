"""The one-off length probe preserves completed answers when a later request fails."""
import json
import runpy
import sys
from pathlib import Path

import httpx
import pytest
from transformers import AutoTokenizer

from rewardhack import hf_grpo
from rewardhack.tasks import select_tasks
from rewardhack.vllm_rollouts import AdapterRollouts, SEED_STRIDE


@pytest.mark.parametrize('fail_later', [False, True])
def test_length_probe_uses_evaluation_draws_and_saves_partial_answers(tmp_path, monkeypatch, fail_later):
    class Tokenizer:
        eos_token_id = 1

        def apply_chat_template(self, messages, **kwargs):
            assert kwargs['enable_thinking'] is True
            return {'input_ids': [8]}

        def decode(self, ids, **kwargs):
            return 'saved completion'

    monkeypatch.setattr(AutoTokenizer, 'from_pretrained', lambda *a, **kw: Tokenizer())
    monkeypatch.setattr(hf_grpo, 'settle_template', lambda *a: None)
    monkeypatch.setattr(AdapterRollouts, 'preflight', lambda self: None)
    reply = {'choices': [{'token_ids': [7, 1], 'finish_reason': 'stop'}]}
    monkeypatch.setattr(AdapterRollouts, 'request', lambda *a, **kw: httpx.Response(200, json=reply))

    def sample(self, model, prompts, *, seed, record):
        assert model is None and len(prompts) == 24
        assert seed == hf_grpo.evaluation_seed(self.args) * SEED_STRIDE
        for index in range(24):
            if fail_later and index == 1:
                raise httpx.ReadTimeout('later request failed')
            self.request('POST', '/inference/v1/generate', body={'sampling_params': {'seed': seed+index}})
        return [[7, 1]] * 24, None, None

    monkeypatch.setattr(AdapterRollouts, 'sample', sample)
    out = tmp_path/'probe'
    monkeypatch.setattr(sys, 'argv', ['qwen_length_measure.py', '--out', str(out), '--model', 'test',
                                    '--revision', 'pinned', '--rollout-url', 'http://127.0.0.1:8011',
                                    '--max-completion-length', '16384'])
    script = Path(__file__).resolve().parents[1]/'entrypoints/qwen_length_measure.py'
    if fail_later:
        with pytest.raises(httpx.ReadTimeout):
            runpy.run_path(str(script), run_name='__main__')
    else:
        runpy.run_path(str(script), run_name='__main__')
    rows = [json.loads(line) for line in (out/'rows.jsonl').read_text().splitlines()]
    assert len(rows) == (1 if fail_later else 24)
    assert rows[0]['task_id'] == select_tasks('hard', 'all', 'evaluation', limit=12)[0]['id']
    assert all(not r['truncated'] and r['raw'] == 'saved completion' for r in rows)
    summary = json.loads((out/'summary.json').read_text())
    assert summary['status'] == ('failed' if fail_later else 'completed')
    assert len((out/'responses.jsonl').read_text().splitlines()) == len(rows)
