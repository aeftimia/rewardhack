"""Exploration probe of a hosted model through an OpenAI-compatible chat API: same prompt as every other
probe, raw outputs only, scored afterwards on the laptop with `entrypoints/rescore.py` (`--lean none` style rows).

Inference only. Costs are recorded from the API's own usage counters per request, so the ledger entry is
measured, not estimated. Requests run concurrently; a failed request is recorded with its error and does
not fabricate a row. Entrypoint: `entrypoints/probe_api.py`.
"""
import concurrent.futures
import json
import os
import random
import time

import requests

from .prompts import follow_on, prompt
from .tasks import select_tasks, taskset_rows


def call(args, messages):
    body = {'model': args.model, 'messages': messages, 'max_completion_tokens': args.max_tokens}
    if args.reasoning_effort:
        body['reasoning_effort'] = args.reasoning_effort
    if args.temperature is not None:
        body['temperature'] = args.temperature
    last = 'no attempt'
    for attempt in range(4):
        try:
            response = requests.post(args.base_url.rstrip('/')+'/chat/completions', json=body, timeout=600,
                                     headers={'Authorization': 'Bearer '+os.environ[args.key_env]})
            if response.status_code == 429 or response.status_code >= 500:
                time.sleep(5*(attempt+1))
                continue
            if response.status_code >= 400:
                return dict(error=f'HTTP {response.status_code}: {response.text[:300]}')
            data = response.json()
            choice = data['choices'][0]
            return dict(raw=choice['message']['content'] or '', finish=choice.get('finish_reason'), usage=data.get('usage', {}),
                        model=data.get('model'))
        except (requests.RequestException, KeyError, ValueError) as error:
            last = f'{type(error).__name__}: {error}'[:300]
            time.sleep(5*(attempt+1))
    return dict(error=last)


class Meter:
    """Running token and dollar totals from the API's usage counters."""

    def __init__(self, usd_per_m_input, usd_per_m_output):
        self.rates = (usd_per_m_input, usd_per_m_output)
        self.totals = dict(input=0, output=0, reasoning=0, usd=0.0, requests=0, errors=0)

    def add(self, usage):
        reasoning = usage.get('completion_tokens_details', {}).get('reasoning_tokens', 0)
        self.totals['input'] += usage.get('prompt_tokens', 0)
        self.totals['output'] += usage.get('completion_tokens', 0)
        self.totals['reasoning'] += reasoning
        self.totals['usd'] = self.totals['input']*self.rates[0]/1e6+self.totals['output']*self.rates[1]/1e6
        return reasoning


def unscored_row(task, index, result, turn, conversation, reasoning):
    """A raw row for later rescoring; `turn` 2 rows belong to the same `conversation` as their turn 1."""
    return dict(task_id=task['id'], arm='baseline', raw=result['raw'], round=index, scored=False, turn=turn, conversation=conversation,
                tokens=result['usage'].get('completion_tokens', 0), reasoning_tokens=reasoning, truncated=result['finish'] == 'length',
                model=result['model'])


def error_row(task, index, result, turn, conversation):
    return dict(task_id=task['id'], arm='baseline', round=index, turn=turn, conversation=conversation, error=result['error'], scored=False)


def run(args):
    args.out.mkdir(parents=True, exist_ok=True)
    tasks = select_tasks(args.taskset, args.tasks, limit=args.limit)
    false_pool = [t for t in taskset_rows(args.taskset) if not t['true']]
    jobs = [(task, index) for index in range(args.samples) for task in tasks]
    started = time.perf_counter()
    meter = Meter(args.usd_per_m_input, args.usd_per_m_output)
    rows = []
    stop = False

    def job(task, index):
        messages = prompt(task['statement'], args.variant)
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
            sink.write(json.dumps(error_row(task, index, result, turn, conversation))+'\n')
            return False
        row = unscored_row(task, index, result, turn, conversation, meter.add(result['usage']))
        rows.append(row)
        sink.write(json.dumps(row, ensure_ascii=False)+'\n')
        return True

    with (args.out/'rows.jsonl').open('w') as sink, concurrent.futures.ThreadPoolExecutor(args.concurrency) as pool:
        pending = iter(jobs)
        futures = {pool.submit(job, *next(pending)) for _ in range(min(args.concurrency, len(jobs)))}
        while futures:
            done, futures = concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                task, index, first, follow, second = future.result()
                conversation = f"{task['id']}:{index}"
                if record(sink, task, index, first, 1, conversation) and second is not None:
                    record(sink, follow, index, second, 2, conversation)
                sink.flush()
                if meter.totals['usd'] > args.max_usd and not stop:
                    stop = True
                    print(f"cost cap ${args.max_usd} reached at ${meter.totals['usd']:.2f}; no new requests", flush=True)
            if not stop:
                # range first: zip pulls left to right, so the other order loses one job per batch.
                for _, next_job in zip(range(len(done)), pending):
                    futures.add(pool.submit(job, *next_job))
    summary = dict(model=args.model, variant=args.variant, samples=len(rows), planned=len(jobs), reasoning_effort=args.reasoning_effort,
                   max_tokens=args.max_tokens, truncated=sum(r['truncated'] for r in rows), **meter.totals,
                   mean_output_tokens=meter.totals['output']/max(len(rows), 1), elapsed_seconds=time.perf_counter()-started,
                   purpose='exploration probe; inference only; scored later with entrypoints/rescore.py', cost_cap_hit=stop)
    (args.out/'summary.json').write_text(json.dumps(summary, indent=1)+'\n')
    print(json.dumps(summary, indent=1))
