"""Replay the current Qwen held-out evaluation prompts without loading a trainer.

Save every completed HTTP response immediately, including if another request fails.
Use the training rollout client's sampling and validation unchanged. Score locally.
"""
import argparse
import json
import threading
import time
from pathlib import Path

from transformers import AutoTokenizer

from rewardhack import jsonl, provenance
from rewardhack.hf_grpo import evaluation_seed, settle_template
from rewardhack.prompts import prompt
from rewardhack.tasks import select_tasks
from rewardhack.vllm_rollouts import AdapterRollouts, SEED_STRIDE


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--revision', required=True)
    parser.add_argument('--rollout-url', required=True)
    parser.add_argument('--max-completion-length', type=int, required=True)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    settle_template(tokenizer, None, 'on')
    tasks = select_tasks('hard', 'all', 'evaluation', limit=12)
    template = {'enable_thinking': True}
    prompts = [tokenizer.apply_chat_template(prompt(t['statement'], 'neutral', 'prose'), tokenize=True,
               return_dict=True, add_generation_prompt=True, **template)['input_ids'] for t in tasks]
    prompts = [p for p in prompts for _ in range(2)]
    seed = evaluation_seed(args) * SEED_STRIDE
    meta = provenance.meta(model=args.model, revision=args.revision, arm=None, penalty=None,
                           stage='untrained', report_mode='prose', taskset='hard', tasks='all',
                           part='evaluation', variant='neutral', template=template, run=args.out.name)
    (args.out/'tasks.json').write_text(json.dumps(tasks, indent=2)+'\n')
    lock = threading.Lock()

    class SavedResponses(AdapterRollouts):
        def request(self, method, path, *, body=None, timeout=120):
            response = super().request(method, path, body=body, timeout=timeout)
            if path == '/inference/v1/generate':
                index = body['sampling_params']['seed'] - seed
                reply = response.json()
                with lock, (args.out/'responses.jsonl').open('a') as sink:
                    jsonl.write(sink, dict(index=index, response=reply))
                choice = reply['choices'][0]
                row = dict(task_id=tasks[index//2]['id'], round=index % 2, arm='baseline', scored=False,
                           report_mode='prose', raw=tokenizer.decode(choice['token_ids'], skip_special_tokens=False),
                           tokens=len(choice['token_ids']), truncated=choice['finish_reason'] == 'length',
                           termination_schema='cutoff-v1', meta=meta)
                with lock, (args.out/'rows.jsonl').open('a') as sink:
                    jsonl.write(sink, row)
            return response

    started = time.monotonic()
    summary = dict(status='running', model=args.model, revision=args.revision, template=template,
                   max_tokens=args.max_completion_length, seed=seed, samples=24)
    (args.out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    backend = SavedResponses(args, tokenizer.eos_token_id)
    try:
        backend.request('POST', '/sleep?level=1')
        backend.preflight()
        ids, _, _ = backend.sample(None, prompts, seed=seed, record={'evaluation': 'before'})
        summary.update(status='completed', generated_tokens=sum(map(len, ids)))
    except BaseException as error:
        summary.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        backend.close()
        summary['elapsed_seconds'] = time.monotonic() - started
        (args.out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
