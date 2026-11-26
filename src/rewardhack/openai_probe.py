"""Exploration probe of a hosted model through an OpenAI-compatible chat API: same prompt as every other
probe, raw outputs only, scored afterwards on the laptop with `entrypoints/rescore.py` (`--lean none` style rows).

Inference only. Costs are recorded from the API's own usage counters per request, so the ledger entry is
measured, not estimated. Requests run concurrently; a failed request is recorded with its error and does
not fabricate a row. Entrypoint: `entrypoints/probe_api.py`.
"""
import json
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from . import jsonl
from .judge import served_here
from .model_io import Meter, call, price, token_counts
from .budget import nonnegative_finite

REQUEST_SECONDS = 600  # per attempt: a long reasoning answer from a hosted model, with nothing rented waiting on it
REQUEST_RETRIES = 3    # after the first attempt

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
    args.max_usd = nonnegative_finite(args.max_usd, 'API spend cap')
    report_mode = args.report_mode
    meter = Meter(0.0, 0.0) if served_here(args.base_url) else Meter(*price(args.model))  # prices: model_io.PRICES
    args.out.mkdir(parents=True, exist_ok=True)
    tasks = select_tasks(args.taskset, args.tasks, limit=args.limit)
    false_pool = [t for t in taskset_rows(args.taskset) if not t['true']]
    jobs = [(task, index) for index in range(args.samples) for task in tasks]
    started = time.perf_counter()
    rows = []
    capped = threading.Event()  # the metered total reached the cap
    stop = threading.Event()    # read by every worker before each request: set at the cap, or when the run ends early
    if args.max_usd == 0:
        capped.set()
        stop.set()
    written = threading.Lock()  # one writer and one meter at a time

    def ask(messages):
        return call(args, messages, timeout=REQUEST_SECONDS, retries=REQUEST_RETRIES, extra_body={})

    def job(sink, task, index):
        if stop.is_set():
            return  # never asked: the cap was reached or the run is ending
        messages = prompt(task['statement'], args.variant, report_mode)
        first = ask(messages)
        follow = second = None
        if args.follow_on and 'error' not in first and not stop.is_set():
            # Second turn: a different false statement, no evaluator code, same conversation.
            follow = random.Random(f"{task['id']}:{index}").choice([t for t in false_pool if t['group'] != task['group']])
            second = ask(messages + [{'role': 'assistant', 'content': first['raw']}, follow_on(follow['statement'])])
        # Metered here, before this worker can take another job, so the cap is seen before any further request.
        with written:
            try:
                conversation = f"{task['id']}:{index}"
                if record(sink, task, index, first, 1, conversation) and second is not None:
                    record(sink, follow, index, second, 2, conversation)
                sink.flush()
            except BaseException:
                stop.set()  # an unmeterable reply ends the run: no other worker may send another request
                raise
            if meter.totals['usd'] >= args.max_usd and not capped.is_set():
                capped.set()
                stop.set()
                print(f"cost cap ${args.max_usd} reached at ${meter.totals['usd']:.2f}; no new requests", flush=True)

    def record(sink, task, index, result, turn, conversation):
        meter.totals['requests'] += 1
        if 'error' in result:
            meter.totals['errors'] += 1
            jsonl.write(sink, error_row(task, index, result, turn, conversation, report_mode), ensure_ascii=True)
            if 'usage' in result:
                meter.add(result['usage'])  # a reply that arrived in an unusable shape was still billed
            return False
        row = unscored_row(task, index, result, turn, conversation, meter.add(result['usage'])['reasoning'], report_mode)
        rows.append(row)
        jsonl.write(sink, row)
        return True

    with (args.out/'rows.jsonl').open('w') as sink, ThreadPoolExecutor(args.concurrency) as pool:
        try:
            for future in as_completed([pool.submit(job, sink, *planned) for planned in jobs]):
                future.result()  # a worker's error ends the run here
        finally:
            stop.set()  # an error, or the hard deadline in this thread, must not leave queued jobs to send requests
    summary = dict(model=args.model, report_mode=report_mode, variant=args.variant, samples=len(rows), planned=len(jobs), reasoning_effort=args.reasoning_effort,
                   max_tokens=args.max_tokens, truncated=sum(r['truncated'] for r in rows), **meter.totals,
                   mean_output_tokens=meter.totals['output']/len(rows) if rows and meter.totals['output'] is not None else None, elapsed_seconds=time.perf_counter()-started,
                   purpose='exploration probe; inference only; scored later with entrypoints/rescore.py', cost_cap_hit=capped.is_set())
    (args.out/'summary.json').write_text(json.dumps(summary, indent=1)+'\n')
    print(json.dumps(summary, indent=1))
