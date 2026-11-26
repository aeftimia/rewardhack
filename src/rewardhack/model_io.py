"""Shared inference transport for probes and report readers; no experiment policy.

One OpenAI-compatible chat call through the `openai` SDK, which serves the OpenAI API and a vLLM server alike.
"""
import os

from openai import APIConnectionError, APIStatusError, OpenAI
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .budget import nonnegative_finite


def call(args, messages, *, timeout, retries, extra_body):
    """One chat completion: `{raw, finish, usage, model}`, or `{error}` (plus `usage` when a reply arrived).

    `timeout` is seconds per attempt and `retries` the attempts after the first: a caller with a rented GPU waiting on
    the answer passes short ones. `extra_body` is merged into the request (a response schema, server options).
    """
    key = os.environ[args.key_env]  # read before any request: a missing key is a setup error and is raised at once, never retried
    options = {'reasoning_effort': args.reasoning_effort} if args.reasoning_effort else {}
    if args.temperature is not None:
        options['temperature'] = args.temperature
    # The SDK's own retry rule: an unanswered request, a timeout, 408, 409, 429, 5xx, or what a server marks with
    # `x-should-retry`; any other status is an error at once. A `HardDeadline` (a BaseException) passes through its
    # retry loop, which catches `Exception`.
    with OpenAI(base_url=args.base_url, api_key=key, timeout=timeout, max_retries=retries) as client:
        try:
            response = client.chat.completions.with_raw_response.create(
                model=args.model, messages=messages, max_completion_tokens=args.max_tokens, extra_body=extra_body, **options)
        except APIStatusError as error:  # an error status, after any retries: nothing was served
            return dict(error=f'HTTP {error.status_code}: {error.response.text[:300]}')
        except APIConnectionError as error:  # no reply on any attempt (APITimeoutError is one)
            return dict(error=f'{type(error).__name__}: {error}'[:300])
        try:
            body = response.http_response.json()
        except ValueError as error:
            return dict(error=f'reply body is not JSON ({error}); not asked again'[:300], usage=None)
    return read_reply(body)


SHAPE = ConfigDict(strict=True, extra='ignore')  # the fields read here must have their types; a reply carries many more


class Message(BaseModel):
    model_config = SHAPE
    content: str | None


class Choice(BaseModel):
    model_config = SHAPE
    message: Message
    finish_reason: str  # without it nobody can say whether the answer was cut off


class Reply(BaseModel):
    model_config = SHAPE
    choices: list[Choice] = Field(min_length=1)
    model: str | None = None  # the name the provider resolved an alias to: a record, absent from some servers


def read_reply(body):
    """A served reply body as `call`'s result, or an error naming what it lacks.

    A reply that arrived was billed, so it is never asked for again: asking again buys the same wrong shape.
    Its error carries the reply's `usage` (None when the reply had none) so the caller's meter still counts it.
    """
    if not isinstance(body, dict):
        return dict(error=f'reply is a JSON {type(body).__name__}, not an object; not asked again', usage=None)
    usage = body['usage'] if 'usage' in body else None
    try:
        reply = Reply.model_validate(body)
    except ValidationError as error:
        wrong = '; '.join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in error.errors())
        return dict(error=f'reply has the wrong shape ({wrong}); not asked again'[:300], usage=usage)
    finish, content = reply.choices[0].finish_reason, reply.choices[0].message.content
    if content is None and finish == 'length':
        content = ''  # the token limit ran out before any answer text (all of it went on reasoning): a cut-off answer with nothing in it
    if content is None:
        return dict(error=f'reply content is None with finish_reason {finish!r}, not text; not asked again', usage=usage)
    return dict(raw=content, finish=finish, usage=usage, model=reply.model)


class UnmeteredReply(RuntimeError):
    """A reply at a price above zero whose token counters are absent: its cost is unknown, so no spend cap can hold."""


COUNTERS = {'input': 'prompt_tokens', 'output': 'completion_tokens'}


def token_counts(usage):
    """A reply's token counters under the meter's names. None means the provider did not send that counter: unknown, not zero."""
    if usage is not None and not isinstance(usage, dict):
        raise ValueError(f'usage is a {type(usage).__name__}, not an object')
    sent = usage if usage is not None else {}
    counts = {name: sent[key] if key in sent else None for name, key in COUNTERS.items()}
    details = sent['completion_tokens_details'] if 'completion_tokens_details' in sent else None
    # Many providers send no reasoning count; it has no price of its own (reasoning is billed inside the output tokens).
    counts['reasoning'] = details['reasoning_tokens'] if isinstance(details, dict) and 'reasoning_tokens' in details else None
    for name, count in counts.items():
        if count is not None and (type(count) is not int or count < 0):
            raise ValueError(f'usage counter for {name} tokens is {count!r}, not a count')
    return counts


class Meter:
    """Running token and dollar totals from the API's usage counters.

    A counter with a price above zero must be in every reply: `add` raises `UnmeteredReply` otherwise, which stops the
    run, because a priced run that meters $0 never reaches its cap. A counter with no price (a server on this machine)
    may be absent: its total then becomes None, unknown, for the rest of the run, and `uncounted` says in how many replies.
    """

    def __init__(self, usd_per_m_input, usd_per_m_output):
        self.rates = dict(input=nonnegative_finite(usd_per_m_input, 'input token price'),
                          output=nonnegative_finite(usd_per_m_output, 'output token price'))
        self.totals = dict(input=0, output=0, reasoning=0, usd=0.0, requests=0, errors=0, uncounted=dict(input=0, output=0, reasoning=0))
        # Replies are counted from concurrent scoring threads: whoever changes or reads `totals` holds this (judge.py).
        import threading
        self.lock = threading.Lock()

    def snapshot(self):
        """A copy of the totals as they stand, consistent with each other: threads may still be counting replies."""
        import copy
        with self.lock:
            return copy.deepcopy(self.totals)

    def add(self, usage):
        """Count one served reply and return its counters (`token_counts`)."""
        counts = token_counts(usage)
        absent = [COUNTERS[name] for name, rate in self.rates.items() if rate > 0 and counts[name] is None]
        if absent:
            raise UnmeteredReply(f"the reply carried no {' or '.join(absent)} in its usage, so its cost cannot be metered")
        for name, count in counts.items():
            if count is None:
                self.totals['uncounted'][name] += 1
                self.totals[name] = None
            elif self.totals[name] is not None:
                self.totals[name] += count
        self.totals['usd'] = sum(self.totals[name]*rate/1e6 for name, rate in self.rates.items() if rate > 0)
        return counts



def local_reader(model_id, revision, max_tokens):
    """Load an MLX model once and return a non-thinking prompt-to-text callable. The model reads one prompt at a time:
    the callable holds a lock, since rows are scored concurrently (outcomes.score_concurrently)."""
    import threading
    from mlx_lm import generate, load
    model, tokenizer = load(model_id, revision=revision)
    lock = threading.Lock()

    def ask(prompt):
        ids = tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}], add_generation_prompt=True,
                                             tokenize=False, enable_thinking=False)
        with lock:
            return generate(model, tokenizer, ids, max_tokens=max_tokens, verbose=False)
    return ask
