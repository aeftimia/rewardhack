"""`openai_probe.run` over a thread pool: the cost cap stops new requests, an error stops queued ones, rows are written as
they finish, failed requests become error rows, and the summary keeps its fields."""
import json
import threading
import time
from types import SimpleNamespace

import pytest

from rewardhack import model_io, openai_probe

TASK = dict(id='t', group='g', statement='True', true=True)
SUMMARY_KEYS = {'model', 'report_mode', 'variant', 'samples', 'planned', 'reasoning_effort', 'max_tokens', 'truncated', 'input', 'output',
                'reasoning', 'usd', 'requests', 'errors', 'uncounted', 'mean_output_tokens', 'elapsed_seconds', 'purpose', 'cost_cap_hit'}


def probe(monkeypatch, tmp_path, reply, made=None, **changes):
    """Run the probe against `reply(number)` for the number-th request; returns the requests made (also kept in `made`)."""
    monkeypatch.setattr(openai_probe, 'select_tasks', lambda *a, **kw: [TASK])
    monkeypatch.setattr(openai_probe, 'taskset_rows', lambda *a: [TASK])
    made, lock = [] if made is None else made, threading.Lock()

    def call(args, messages, *, timeout, retries, extra_body):
        assert (timeout, retries, extra_body) == (openai_probe.REQUEST_SECONDS, openai_probe.REQUEST_RETRIES, {})
        with lock:
            made.append(messages)
            number = len(made)
        return reply(number)
    monkeypatch.setattr(openai_probe, 'call', call)
    args = dict(report_mode='prose', max_usd=1, usd_per_m_input=0, usd_per_m_output=0, out=tmp_path, taskset='v1', tasks='all', limit=None,
                samples=10, concurrency=2, follow_on=False, variant='neutral', model='m', reasoning_effort=None, max_tokens=8)
    openai_probe.run(SimpleNamespace(**dict(args, **changes)))
    return made


def rows(tmp_path):
    return [json.loads(line) for line in (tmp_path/'rows.jsonl').read_text().splitlines()]


def test_the_cost_cap_stops_new_requests(monkeypatch, tmp_path):
    dollar = dict(raw='{}', finish='stop', model='m', usage=dict(prompt_tokens=1_000_000, completion_tokens=0))  # $1 each at $1/M input
    made = probe(monkeypatch, tmp_path, lambda number: dollar, max_usd=2, usd_per_m_input=1, usd_per_m_output=1)
    summary = json.loads((tmp_path/'summary.json').read_text())
    # The second reply is metered before its worker takes another job; only the other worker's request in flight may still finish.
    assert 2 <= len(made) <= 2 + 1 and summary['cost_cap_hit'] and summary['planned'] == 10
    assert summary['requests'] == len(made) == len(rows(tmp_path)) and summary['usd'] == len(made)


def test_no_cap_reached_runs_every_job_with_unchanged_summary_fields(monkeypatch, tmp_path):
    made = probe(monkeypatch, tmp_path, lambda number: dict(raw='{}', finish='stop', model='m', usage=dict(prompt_tokens=1, completion_tokens=2)))
    summary = json.loads((tmp_path/'summary.json').read_text())
    assert len(made) == 10 and summary['samples'] == 10 and not summary['cost_cap_hit'] and set(summary) == SUMMARY_KEYS


def test_a_failed_request_is_an_error_row_and_a_billed_bad_reply_is_metered(monkeypatch, tmp_path):
    replies = {1: dict(error='HTTP 503: busy'), 2: dict(error='reply has the wrong shape; not asked again', usage=dict(prompt_tokens=3, completion_tokens=1))}
    probe(monkeypatch, tmp_path, lambda number: replies[number] if number in replies else dict(raw='{}', finish='stop', model='m', usage=dict(prompt_tokens=1, completion_tokens=2)),
          samples=4, concurrency=1)
    written = rows(tmp_path)
    errors = [row for row in written if 'error' in row]
    summary = json.loads((tmp_path/'summary.json').read_text())
    assert len(written) == 4 and [row['error'] for row in errors] == ['HTTP 503: busy', 'reply has the wrong shape; not asked again']
    assert summary['errors'] == 2 and summary['samples'] == 2 and summary['input'] == 3 + 2*1


def test_rows_are_written_as_they_finish(monkeypatch, tmp_path):
    def reply(number):
        if number == 2:  # the second request is answered only once the first row is on disk
            limit = time.monotonic() + 5
            while not (tmp_path/'rows.jsonl').read_text() and time.monotonic() < limit:
                time.sleep(0.01)
            assert (tmp_path/'rows.jsonl').read_text(), 'the first row was not written before the run ended'
        return dict(raw='{}', finish='stop', model='m', usage={})
    assert len(probe(monkeypatch, tmp_path, reply, samples=2)) == 2


def test_an_error_stops_the_queued_jobs(monkeypatch, tmp_path):
    made = []
    with pytest.raises(model_io.UnmeteredReply):
        probe(monkeypatch, tmp_path, lambda number: dict(raw='{}', usage=None, finish='stop', model='m'), made,
              samples=50, usd_per_m_input=0.2, usd_per_m_output=1.25)
    # The first reply stops the run; only requests already started by the two workers may still be sent.
    assert 1 <= len(made) <= 1 + 1 and not (tmp_path/'summary.json').exists()
