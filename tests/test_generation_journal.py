import json
from types import SimpleNamespace

import pytest

from rewardhack import generations, hf_grpo, jsonl, vllm_rollouts
from conftest import run_options
import test_training_integration
from test_training_integration import TASKS, tiny, train


def test_training_journal_joins_real_cpu_generations_to_each_reward(tmp_path, monkeypatch):
    # Different dataset groups can contain the very same proposition.
    paired = [dict(task, statement='same', true=True) for task in TASKS[:2]]
    monkeypatch.setattr(test_training_integration, 'TASKS', paired)
    run = train(tmp_path, monkeypatch, lambda i, raw, step: i, steps=2, run_id='example')
    generated = list(jsonl.read(run.out / 'training-generations.jsonl'))
    assert {row['step'] for row in generated} == {1, 2}
    assert {row['task_id'] for row in generated} == {task['id'] for task in paired}
    assert len({row['generation_id'] for row in generated}) == len(generated)
    for row, scored in zip(generated, run.rows, strict=True):
        assert (row['generation_id'], row['task_id']) == (scored['generation_id'], scored['task_id'])
        assert row['messages'][0]['role'] == 'system'
        assert row['prompt_token_ids'] and row['completion_token_ids']
        assert row['run_id'] == 'example'


@pytest.mark.parametrize('backend', ['vllm', 'transformers'])
def test_sampling_records_tokens_before_first_score(tmp_path, monkeypatch, backend):
    model, tokenizer = tiny('plain')
    class Server:
        def sample(self, adapter_model, prompts, *, seed, record):
            return [[3, 1] for _ in prompts], None, None

    args = run_options(out=tmp_path, lean='unused', arm='baseline', penalty=0, judge_fn=None,
                       report_mode='prose', verifier_retry_seconds=60, eval_samples=2, seed=7,
                       prompt_variant='neutral', thinking='unsupported', reasoning_effort=None,
                       max_completion_length=2)
    args.run_id = 'example'
    args.completed_updates = 5
    server = Server() if backend == 'vllm' else None
    calls = []
    def fail(*args, **kwargs):
        calls.append(len(list(jsonl.read(tmp_path / 'after-generations.jsonl'))))
        raise ValueError('scorer stopped')
    monkeypatch.setattr(hf_grpo, 'score', fail)
    with pytest.raises(ValueError, match='scorer stopped'):
        hf_grpo.evaluate(model, tokenizer, TASKS[:2], args, 'after', rollouts=server)
    rows = list(jsonl.read(tmp_path / 'after-generations.jsonl'))
    assert calls and all(n == len(rows) for n in calls)
    assert len(rows) == (4 if backend == 'vllm' else 2)
    for row in rows:
        tokens = tokenizer.apply_chat_template(row['messages'], tokenize=True, return_dict=True,
                                               add_generation_prompt=True)['input_ids']
        assert row['prompt_token_ids'] == tokens and row['step'] == 5
        assert row['output'] == tokenizer.decode(row['completion_token_ids'], skip_special_tokens=False)


def test_external_observer_preserves_historical_sample_and_paired_task_ambiguity(tmp_path, monkeypatch):
    args = SimpleNamespace(out=tmp_path, report_mode='prose', prompt_variant='refute',
                           eval_samples=1, thinking='unsupported', reasoning_effort='low')
    tasks = {'train':[dict(id='a', statement='True'), dict(id='b', statement='True')],
             'evaluation':[dict(id='c', statement='False')]}
    (tmp_path/'tasks.json').write_text(json.dumps(tasks))
    (tmp_path/'live.json').write_text(json.dumps({'step':7}))
    class Tokenizer:
        def apply_chat_template(self, chat, **kwargs):
            return {'input_ids': list(chat[-1]['content'].encode())}

        def decode(self, ids, **kwargs):
            return str(ids)

    tokenizer = Tokenizer()
    monkeypatch.setattr(hf_grpo, 'load_model', lambda args, summary: (object(), tokenizer))
    backend = object.__new__(vllm_rollouts.AdapterRollouts)
    backend.args = args
    answer = ([[5]], [[0.1]], None)
    monkeypatch.setattr(vllm_rollouts.AdapterRollouts, 'sample', lambda *a, **kw: answer)
    with monkeypatch.context() as capture:
        # Restore the original scorer and both wrapped functions after this test.
        for module, name in [(hf_grpo, 'score_concurrently'), (hf_grpo, 'load_model'), (vllm_rollouts.AdapterRollouts, 'sample')]:
            capture.setattr(module, name, getattr(module, name))
        generations.observe_archived_vllm('historic')
        hf_grpo.load_model(args, {})
        for phase, task, info in [('training', tasks['train'][1], {'step':7}), ('after', tasks['evaluation'][0], {'evaluation':'after'})]:
            chat = hf_grpo.prompt(task['statement'], args.prompt_variant, args.report_mode)
            tokens = tokenizer.apply_chat_template(chat)['input_ids']
            assert backend.sample(None, [tokens], seed=123, record=info) is answer
            scored, = hf_grpo.score_concurrently(lambda item: dict(task_id=item['id']), [task])
            row, = jsonl.read(tmp_path/f'{phase}-generations.jsonl')
            assert row['generation_id'] == scored['generation_id'] and row['step'] == 7
            assert row['candidate_task_ids'] == (['a', 'b'] if phase == 'training' else ['c'])
            assert row['prompt_token_ids'] == tokens and row['completion_token_ids'] == [5]
