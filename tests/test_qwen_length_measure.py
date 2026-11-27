"""The one-off length probe preserves completed answers when a later request fails."""
import json
import runpy
import sys
from pathlib import Path

import httpx
import pytest
from transformers import AutoTokenizer

from rewardhack import hf_grpo
from rewardhack.tasks import select_tasks, stratified_limit
from rewardhack.vllm_rollouts import AdapterRollouts, SEED_STRIDE


@pytest.mark.parametrize('failure', [None, 'timeout', 'two choices'])
def test_length_probe_uses_evaluation_draws_and_saves_partial_answers(tmp_path, monkeypatch, failure):
    class Tokenizer:
        eos_token_id = 1

        def apply_chat_template(self, messages, **kwargs):
            assert kwargs['enable_thinking'] is True
            return {'input_ids': [8]}

        def decode(self, ids, **kwargs):
            return 'saved completion'

    monkeypatch.setattr(AutoTokenizer, 'from_pretrained', lambda *a, **kw: Tokenizer())
    monkeypatch.setattr(hf_grpo, 'settle_template', lambda *a: None)
    monkeypatch.setattr(AdapterRollouts, 'preflight', lambda self: setattr(self, 'max_model_len', 3))
    monkeypatch.setattr(AdapterRollouts, 'require_sleep', lambda *args: None)

    def request(self, method, path, *, body=None, timeout=120):
        if path != '/inference/v1/generate':
            return httpx.Response(200, json={})
        index = body['sampling_params']['seed'] - hf_grpo.evaluation_seed(self.args) * SEED_STRIDE
        assert 0 <= index < 24
        assert body['model'] == 'test@pinned'
        if index == 23 and failure == 'timeout':
            raise httpx.ReadTimeout('later request failed')
        tokens = [7, 1] if index % 2 == 0 else [7, 8]
        logs = [{'token_id': token, 'logprob': -0.25} for token in tokens]
        choice = dict(token_ids=tokens, finish_reason='length', logprobs={'content': logs})
        choices = [choice, choice] if index == 23 and failure == 'two choices' else [choice]
        return httpx.Response(200, json={'choices': choices})

    monkeypatch.setattr(AdapterRollouts, 'request', request)
    out = tmp_path/'probe'
    monkeypatch.setattr(sys, 'argv', ['qwen_length_measure.py', '--out', str(out), '--model', 'test',
                                    '--revision', 'pinned', '--rollout-url', 'http://127.0.0.1:8011',
                                    '--max-completion-length', '2'])
    script = Path(__file__).resolve().parents[1]/'entrypoints/qwen_length_measure.py'
    if failure:
        with pytest.raises(httpx.ReadTimeout if failure == 'timeout' else ValueError,
                           match='later request failed' if failure == 'timeout' else 'exactly one completion'):
            runpy.run_path(str(script), run_name='__main__')
    else:
        runpy.run_path(str(script), run_name='__main__')
    rows = [json.loads(line) for line in (out/'rows.jsonl').read_text().splitlines()]
    assert len(rows) == (23 if failure else 24)
    expected = stratified_limit(select_tasks('hard', 'all', 'evaluation'), 12)
    assert [row['id'] for row in json.loads((out/'tasks.json').read_text())] == [row['id'] for row in expected]
    assert {row['id'] for row in expected} == {
        f'{group}-{truth}' for group in ('boundary-0-1', 'boundary-1-1', 'nattrap-sub-sub',
                                       'residue-n^3+2*n+7-37', 'sat-8-1', 'satsmall-4-2')
        for truth in ('T', 'F')}
    expected_draws = [(task['id'], index) for task in expected for index in range(2)]
    assert sorted((r['task_id'], r['round']) for r in rows) == sorted(expected_draws[:len(rows)])
    assert all(r['truncated'] == (r['round'] == 1) and r['raw'] == 'saved completion' for r in rows)
    summary = json.loads((out/'summary.json').read_text())
    assert summary['status'] == ('failed' if failure else 'completed')
    assert summary['samples'] == len(expected_draws)
    assert len((out/'responses.jsonl').read_text().splitlines()) == (23 if failure == 'timeout' else 24)
