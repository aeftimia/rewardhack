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
from rewardhack.hf_grpo import evaluation_draws, settle_template, stage_tasks, template_kwargs
from rewardhack.outcomes import completion_truncated
from rewardhack.vllm_rollouts import AdapterRollouts


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--revision', required=True)
    parser.add_argument('--rollout-url', required=True)
    parser.add_argument('--max-completion-length', type=int, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.set_defaults(reasoning_effort=None, thinking='on', prompt_variant='neutral', report_mode='prose')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    settle_template(tokenizer, args.reasoning_effort, args.thinking)
    tasks = stage_tasks('hard', 'before', 12)
    template = template_kwargs(args)
    seed, draws = evaluation_draws(tokenizer, tasks, args, 2)
    prompts = [ids for _, _, ids in draws]
    meta = provenance.meta(model=args.model, revision=args.revision, arm=None, penalty=None,
                           stage='untrained', report_mode=args.report_mode, taskset='hard', tasks='all',
                           part='evaluation', variant=args.prompt_variant, template=template, run=args.out.name)
    (args.out/'tasks.json').write_text(json.dumps(tasks, indent=2)+'\n')
    lock = threading.Lock()

    class SavedResponses(AdapterRollouts):
        def parse_completion(self, index, reply, max_tokens):
            with lock, (args.out/'responses.jsonl').open('a') as sink:
                jsonl.write(sink, dict(index=index, response=reply))
            tokens, values = super().parse_completion(index, reply, max_tokens)
            task, round_index, _ = draws[index]
            row = dict(task_id=task['id'], round=round_index, arm='baseline', scored=False,
                       report_mode=args.report_mode, raw=tokenizer.decode(tokens, skip_special_tokens=False),
                       tokens=len(tokens), truncated=completion_truncated(tokens, self.stops, max_tokens),
                       termination_schema='cutoff-v1', meta=meta)
            with lock, (args.out/'rows.jsonl').open('a') as sink:
                jsonl.write(sink, row)
            return tokens, values

    started = time.monotonic()
    summary = dict(status='running', model=args.model, revision=args.revision, template=template,
                   max_tokens=args.max_completion_length, seed=seed, samples=len(draws))
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
