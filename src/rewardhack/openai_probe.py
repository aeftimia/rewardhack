"""Exploration probe of a hosted model through an OpenAI-compatible chat API: same prompt as every other
probe, raw outputs only, scored afterwards on the laptop with `entrypoints/rescore.py` (`--lean none` style rows).

Inference only. Costs are recorded from the API's own usage counters per request, so the ledger entry is
measured, not estimated. Requests run concurrently; a failed request is recorded with its error and does
not fabricate a row. Entrypoint: `entrypoints/probe_api.py`.
"""
import concurrent.futures
import json
import random
import time

from .model_io import Meter, call, token_counts
from .budget import nonnegative_finite

from .prompts import follow_on, prompt
from .tasks import select_tasks, taskset_rows


def unscored_row(task, index, result, turn, conversation, reasoning, report_mode='prose'):
    """A raw row for later rescoring; `turn` 2 rows belong to the same `conversation` as their turn 1.

    `tokens` and `reasoning_tokens` are None when the provider did not send that counter: unknown, not 0.
    """
    return dict(task_id=task['id'], arm='baseline', raw=result['raw'], round=index, scored=False, termination_schema='cutoff-v1', turn=turn, conversation=conversation,
                tokens=token_counts(result['usage'])['output'], reasoning_tokens=reasoning, truncated=result['finish'] == 'length',
                model=result['model'], report_mode=report_mode)


def error_row(task, index, result, turn, conversation, report_mode='prose'):
    return dict(task_id=task['id'], arm='baseline', round=index, turn=turn, conversation=conversation, error=result['error'], scored=False, report_mode=report_mode)


def run(args):
    report_mode = getattr(args, 'report_mode', 'prose')
    args.max_usd = nonnegative_finite(args.max_usd, 'API spend cap')
    meter = Meter(args.usd_per_m_input, args.usd_per_m_output)
    args.out.mkdir(parents=True, exist_ok=True)
    tasks = select_tasks(args.taskset, args.tasks, limit=args.limit)
    false_pool = [t for t in taskset_rows(args.taskset) if not t['true']]
    jobs = [(task, index) for index in range(args.samples) for task in tasks]
    started = time.perf_counter()
    rows = []
    stop = args.max_usd == 0

    def job(task, index):
        messages = prompt(task['statement'], args.variant, report_mode)
        first = call(args, messages)
        if not args.follow_on or 'error' in first:
            return task, index, first, None, None
        # Second turn: a different false statement, no evaluator code, same conversation.
        follow = random.Random(f"{task['id']}:{index}").choice([t for t in false_pool if t['group'] != task['group']])
        second = call(args, messages + [{'role': 'assistant', 'content': first['raw']}, follow_on(follow['statement'])])
        return task, index, first, follow, second

    def record(sink, task, index, result, turn, conversation):
        meter.totals['requests'] += 1
        if 'error' in result:
            meter.totals['errors'] += 1
            sink.write(json.dumps(error_row(task, index, result, turn, conversation, report_mode))+'\n')
            if 'usage' in result:
                meter.add(result['usage'])  # a reply that arrived in an unusable shape was still billed
            return False
        row = unscored_row(task, index, result, turn, conversation, meter.add(result['usage'])['reasoning'], report_mode)
        rows.append(row)
        sink.write(json.dumps(row, ensure_ascii=False)+'\n')
        return True

    with (args.out/'rows.jsonl').open('w') as sink, concurrent.futures.ThreadPoolExecutor(args.concurrency) as pool:
        pending = iter(jobs)
        futures = {pool.submit(job, *next(pending)) for _ in range(0 if stop else min(args.concurrency, len(jobs)))}
        while futures:
            done, futures = concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                task, index, first, follow, second = future.result()
                conversation = f"{task['id']}:{index}"
                if record(sink, task, index, first, 1, conversation) and second is not None:
                    record(sink, follow, index, second, 2, conversation)
                sink.flush()
                if meter.totals['usd'] >= args.max_usd and not stop:
                    stop = True
                    print(f"cost cap ${args.max_usd} reached at ${meter.totals['usd']:.2f}; no new requests", flush=True)
            if not stop:
                # range first: zip pulls left to right, so the other order loses one job per batch.
                for _, next_job in zip(range(len(done)), pending):
                    futures.add(pool.submit(job, *next_job))
    summary = dict(model=args.model, report_mode=report_mode, variant=args.variant, samples=len(rows), planned=len(jobs), reasoning_effort=args.reasoning_effort,
                   max_tokens=args.max_tokens, truncated=sum(r['truncated'] for r in rows), **meter.totals,
                   mean_output_tokens=meter.totals['output']/len(rows) if rows and meter.totals['output'] is not None else None, elapsed_seconds=time.perf_counter()-started,
                   purpose='exploration probe; inference only; scored later with entrypoints/rescore.py', cost_cap_hit=stop)
    (args.out/'summary.json').write_text(json.dumps(summary, indent=1)+'\n')
    print(json.dumps(summary, indent=1))
