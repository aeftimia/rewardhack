"""Record configured experiments' model requests and returned samples before scoring."""
import json
import os
import runpy
import sys
from types import SimpleNamespace
from rewardhack import jsonl


def exec_archived(source, run_id, arguments, *, override_limit):
    script = str(source / 'entrypoints/run.py')
    env = {**os.environ, 'PYTHONPATH': str(source / 'src'), 'REWARDHACK_OBSERVED_RUN': run_id}
    os.execve(sys.executable, [sys.executable, __file__, str(int(override_limit)), script, *arguments], env)


def validate_archived_resume(trainer, check, args):
    """Validate the predecessor using its original ceiling. Training receives the child arguments."""
    original, _, _ = trainer.resume_point(args)
    parent_limit = json.loads((original / trainer.RUN_SETTINGS).read_text())['max_completion_length']
    print(f'Explicit continuation token ceiling: {parent_limit} -> {args.max_completion_length}', flush=True)
    return check(SimpleNamespace(**{**vars(args), 'max_completion_length': parent_limit}))


def task_messages(render, task, args):
    return render(task['statement'], args.prompt_variant, args.report_mode)


def request_tokens(tokenizer, chat, template):
    return tokenizer.apply_chat_template(chat, tokenize=True, return_dict=True,
                                         add_generation_prompt=True, **template)['input_ids']


def identity(args, phase, step, task_id, sample_index):
    """Join a generated sequence to its scored row, including when scoring stops early."""
    if 'run_id' not in vars(args):
        return {}
    return {'run_id': args.run_id, 'generation_id': f'{args.run_id}/{phase}/{step}/{task_id}/{sample_index}'}


def record(args, tokenizer, phase, step, inputs, completions, *, candidate_task_ids=None):
    """Inputs are (task ID, sample index, messages, actual prompt tokens) in sampler order."""
    if 'run_id' not in vars(args):
        return
    identities = []
    with (args.out / f'{phase}-generations.jsonl').open('a') as sink:
        for (task_id, index, messages, prompt_ids), completion in zip(inputs, completions, strict=True):
            key = identity(args, phase, step, task_id, index)
            identities.append((task_id, key))
            jsonl.write(sink, dict(**key,
                                   phase=phase, step=step, task_id=task_id,
                                   sample_index=index, messages=messages, prompt_token_ids=prompt_ids,
                                   candidate_task_ids=[task_id] if candidate_task_ids is None else candidate_task_ids[tuple(prompt_ids)],
                                   completion_token_ids=completion, output=tokenizer.decode(completion, skip_special_tokens=False)))
    return identities


def observe_archived_vllm(run_id):
    """Observe requests on the archived trainer's vLLM boundary using its own tokenizer and prompt.

    Invoked in a separate interpreter whose rewardhack package is the exact historical source.
    These wrappers delegate sampling, model loading and scoring unchanged to that package.
    """
    from rewardhack import hf_grpo, vllm_rollouts
    load = hf_grpo.load_model
    sample = vllm_rollouts.AdapterRollouts.sample
    score_batch = hf_grpo.score_concurrently
    train_tasks = {}
    train_ids = {}
    evaluation = []
    tokenizer = None
    pending = iter(())

    def load_and_index(args, summary):
        nonlocal tokenizer
        model, tokenizer = load(args, summary)
        dataset = json.loads((args.out / 'tasks.json').read_text())
        def request(task):
            chat = task_messages(hf_grpo.prompt, task, args)
            return tuple(request_tokens(tokenizer, chat, hf_grpo.template_kwargs(args))), (task['id'], chat)
        for task in dataset['train']:
            key, payload = request(task)
            if key in train_tasks and train_tasks[key][1] != payload[1]:
                raise ValueError('distinct historical training messages tokenize identically')
            # The archived reward function looked up the last task for an identical statement.
            train_tasks[key] = payload
            train_ids.setdefault(key, []).append(task['id'])
        for task in dataset['evaluation']:
            evaluation.append(request(task))
        return model, tokenizer

    def sample_and_record(backend, adapter_model, prompts, *, seed, record):
        nonlocal pending
        evidence = SimpleNamespace(out=backend.args.out, run_id=run_id)
        if 'step' in record:
            phase, step = 'training', record['step']
            indices = range(len(prompts))
            tasks = [train_tasks[tuple(tokens)] for tokens in prompts]
            candidates = train_ids
        else:
            phase = record['evaluation']
            step = json.loads((evidence.out / 'live.json').read_text())['step'] if phase == 'after' else 0
            expected = [(key, payload, i) for key, payload in evaluation for i in range(backend.args.eval_samples)]
            if [tuple(tokens) for tokens in prompts] != [key for key, _, _ in expected]:
                raise ValueError('historical evaluation prompts differ from the recorded task selection')
            tasks = [payload for _, payload, _ in expected]
            indices = [i for _, _, i in expected]
            candidates = None
        # dry-ok: entrypoints/qwen_length_measure.py::<module> | This wrapper observes the archived trainer's actual invocation and must forward its arguments unchanged. test_external_observer_preserves_historical_sample_and_paired_task_ambiguity exercises both training and after evaluation. The standalone length probe measures only the pinned base model before training; passing its fixed arguments here would corrupt both archived stages.
        result = sample(backend, adapter_model, prompts, seed=seed, record=record)
        requests = ((task_id, i, chat, tokens) for (task_id, chat), i, tokens in zip(tasks, indices, prompts, strict=True))
        pending = iter(write_batch(evidence, tokenizer, phase, step, requests, result[0], candidate_task_ids=candidates))
        return result

    def score_and_link(function, items):
        def linked(pair):
            item, (task_id, key) = pair
            row = function(item)
            if row['task_id'] != task_id:
                raise ValueError('scored task differs from the recorded historical generation')
            return dict(row, **key)
        return score_batch(linked, [(item, next(pending)) for item in items])

    write_batch = record
    hf_grpo.load_model = load_and_index
    vllm_rollouts.AdapterRollouts.sample = sample_and_record
    hf_grpo.score_concurrently = score_and_link


if __name__ == '__main__':
    observe_archived_vllm(os.environ['REWARDHACK_OBSERVED_RUN'])
    if int(sys.argv[1]):
        from functools import partial
        from rewardhack import hf_grpo
        hf_grpo.check_resume = partial(validate_archived_resume, hf_grpo, hf_grpo.check_resume)
    del sys.argv[:2]
    runpy.run_path(sys.argv[0], run_name='__main__')
