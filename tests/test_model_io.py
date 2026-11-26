"""`model_io.call` through the openai SDK against a fake server (`httpx.MockTransport`): what is sent, what is sent
again, and what a reply that arrived in the wrong shape becomes. No request leaves this process."""
import json
import time
from types import SimpleNamespace

import httpx
import openai._base_client
import pytest

from rewardhack import deadline, judge, model_io

REAL_SLEEP = time.sleep
USAGE = dict(prompt_tokens=4, completion_tokens=2)
ARGS = SimpleNamespace(model='m', max_tokens=8, reasoning_effort=None, temperature=None, base_url='http://127.0.0.1:1/v1', key_env='MODEL_IO_TEST_KEY')


def body(**changes):
    return dict(dict(choices=[dict(index=0, message=dict(role='assistant', content='text'), finish_reason='stop')], model='m', usage=USAGE), **changes)


def answer(status=200, content=None, headers=None, **changes):
    """A server reply: a JSON body unless `content` (bytes) is given."""
    return httpx.Response(status, content=content if content is not None else json.dumps(body(**changes)).encode(),
                          headers=dict({'content-type': 'application/json'}, **(headers or {})))


def no_waiting(seconds):
    pass


@pytest.fixture
def server(monkeypatch):
    """Serve canned replies (or raise canned errors) in order; returns the list of requests received."""
    monkeypatch.setenv('MODEL_IO_TEST_KEY', 'placeholder')
    monkeypatch.setattr(openai._base_client.time, 'sleep', no_waiting)  # the SDK's backoff between attempts
    received, real = [], model_io.OpenAI

    def serve(*replies):
        received.clear()
        def handler(request):
            received.append(request)
            reply = replies[len(received)-1]
            if isinstance(reply, BaseException):
                raise reply
            return reply

        class Fake(real):
            def __init__(self, **options):
                super().__init__(**options, http_client=httpx.Client(transport=httpx.MockTransport(handler)))
        monkeypatch.setattr(model_io, 'OpenAI', Fake)
        return received
    return serve


def ask(**options):
    return model_io.call(ARGS, [{'role': 'user', 'content': 'hi'}], **dict(dict(timeout=60, retries=3, extra_body={}), **options))


def test_a_whole_reply(server):
    received = server(answer())
    assert ask() == dict(raw='text', finish='stop', usage=USAGE, model='m') and len(received) == 1


def test_the_request_carries_every_option_and_the_per_attempt_timeout(server):
    received = server(answer())
    reasoning = SimpleNamespace(**dict(vars(ARGS), reasoning_effort='low', temperature=0))
    model_io.call(reasoning, [{'role': 'user', 'content': 'hi'}], timeout=60, retries=0,
                  extra_body={'response_format': judge.SCHEMA, 'chat_template_kwargs': {'enable_thinking': False}})
    request = received[0]
    sent = json.loads(request.content)
    assert str(request.url) == 'http://127.0.0.1:1/v1/chat/completions' and request.headers['authorization'] == 'Bearer placeholder'
    assert sent == dict(model='m', messages=[{'role': 'user', 'content': 'hi'}], max_completion_tokens=8, reasoning_effort='low', temperature=0,
                        response_format=judge.SCHEMA, chat_template_kwargs={'enable_thinking': False})
    assert request.extensions['timeout'] == dict(connect=60, read=60, write=60, pool=60)
    server(answer())
    ask()
    assert 'reasoning_effort' not in json.loads(received[0].content) and 'temperature' not in json.loads(received[0].content)


def test_the_judge_asks_with_its_short_timeout_and_its_schema(server):
    received = server(answer(choices=[dict(message=dict(content=json.dumps(dict(requested='a', reported='b', **dict.fromkeys(judge.CHECKS, False)))),
                                           finish_reason='stop')]))
    reader = judge.make_judge('api', model='m', base_url=ARGS.base_url, key_env=ARGS.key_env)
    assert reader(dict(statement='True', report='Proved it.'))['claim'] == 'OK'
    assert received[0].extensions['timeout']['read'] == judge.REQUEST_SECONDS == 60
    assert json.loads(received[0].content)['response_format'] == judge.SCHEMA


def test_a_missing_key_fails_before_any_request(server, monkeypatch):
    received = server()
    monkeypatch.delenv('MODEL_IO_TEST_KEY')
    with pytest.raises(KeyError, match='MODEL_IO_TEST_KEY'):
        ask()
    assert received == []


@pytest.mark.parametrize('failure', [answer(408), answer(409), answer(429), answer(500), answer(503), httpx.ConnectError('refused'), httpx.ReadTimeout('stalled')])
def test_unanswered_and_load_refused_requests_are_retried_the_given_number_of_times(server, failure):
    received = server(failure, answer())
    assert ask(retries=1)['raw'] == 'text' and len(received) == 2
    received = server(*[failure]*3)
    result = ask(retries=2)
    assert len(received) == 3 and set(result) == {'error'}  # nothing was served, so there is no usage to meter
    assert result['error'].startswith(('HTTP 408', 'HTTP 409', 'HTTP 429', 'HTTP 500', 'HTTP 503', 'APIConnectionError', 'APITimeoutError'))


@pytest.mark.parametrize('status', [400, 401, 403, 404, 422])
def test_any_other_error_status_is_an_error_at_once(server, status):
    received = server(answer(status, content=b'{"error": "no"}'), answer())
    assert ask() == dict(error=f'HTTP {status}: {{"error": "no"}}') and len(received) == 1


def test_the_hard_deadline_passes_through_the_retry_loop(server):
    received = server(deadline.HardDeadline('hard deadline'), answer())
    with pytest.raises(deadline.HardDeadline):
        ask()
    assert len(received) == 1, 'the deadline was taken for a failed connection and the request sent again'


def test_the_hard_deadline_alarm_ends_a_retry_wait(server, monkeypatch):
    """The real SIGALRM mechanism, firing while the SDK sleeps between attempts."""
    monkeypatch.setattr(openai._base_client.time, 'sleep', REAL_SLEEP)
    monkeypatch.setattr(openai.OpenAI, '_calculate_retry_timeout', lambda self, *a, **kw: 5.0)
    received = server(*[answer(503)]*4)
    started = time.monotonic()
    deadline.arm(0.2)
    try:
        with pytest.raises(deadline.HardDeadline):
            ask(retries=3)
    finally:
        deadline.disarm()
    assert len(received) == 1 and time.monotonic() - started < 2, 'the alarm did not end the wait between attempts'


@pytest.mark.parametrize('changes,named', [
    (dict(choices=[]), 'choices'), (dict(choices=None), 'choices'), (dict(choices=[dict(finish_reason='stop')]), 'choices.0.message'),
    (dict(choices=[dict(message={}, finish_reason='stop')]), 'choices.0.message.content'),
    (dict(choices=[dict(message=dict(content='text'))]), 'choices.0.finish_reason'),
    (dict(choices=[dict(message=dict(content='text'), finish_reason=None)]), 'choices.0.finish_reason'),
    (dict(choices=[dict(message=dict(content=[{'type': 'text', 'text': 'a'}]), finish_reason='stop')]), 'choices.0.message.content'),
    (dict(choices=[dict(message=dict(content=None), finish_reason='stop')]), 'not text')])
def test_a_reply_in_the_wrong_shape_is_an_error_that_carries_its_usage_and_is_not_asked_again(server, changes, named):
    received = server(*[answer(**changes)]*4)
    result = ask()
    assert len(received) == 1, 'a billed reply was requested again'
    assert named in result['error'] and 'raw' not in result
    assert result['usage'] == USAGE, 'the billed reply no longer reaches the meter'


def test_a_reply_that_is_not_a_json_object(server):
    received = server(answer(content=b'<html>'), answer(content=b'[1, 2]'))
    first, second = ask(), ask()
    assert 'not JSON' in first['error'] and first['usage'] is None
    assert 'not an object' in second['error'] and second['usage'] is None and len(received) == 2


def test_no_content_with_finish_length_is_an_empty_cut_off_answer(server):
    server(answer(choices=[dict(message=dict(content=None), finish_reason='length')]))
    assert ask() == dict(raw='', finish='length', usage=USAGE, model='m')


def test_a_judge_reply_in_the_wrong_shape_is_a_failed_reading_and_is_metered(server, monkeypatch):
    received = server(answer(choices=[dict(message=dict(content='{}'))]))
    # The stub server is on this machine, which is never metered: meter it as a paid endpoint at $1/M each way.
    monkeypatch.setattr(judge, 'served_here', lambda url: False)
    monkeypatch.setitem(model_io.PRICES, 'm', (1, 1))
    reader = judge.make_judge('api', model='m', base_url=ARGS.base_url, key_env=ARGS.key_env, max_usd=1)
    verdict = reader(dict(statement='True', report='Proved it.'))
    assert verdict['claim'] == 'UNPARSED' and 'finish_reason' in verdict['judge_error'] and len(received) == 1
    assert reader.meter.totals['errors'] == 1 and reader.meter.totals['usd'] == 6/1e6, 'the unusable reply was billed and not counted'
