"""Shared inference transport for probes and report readers; no experiment policy."""
import os
import time

import requests

from .budget import nonnegative_finite


def call(args, messages, timeout=600):
    """`timeout` is per attempt, of four: a caller with a rented GPU waiting on the answer passes a short one."""
    body = {'model': args.model, 'messages': messages, 'max_completion_tokens': args.max_tokens}
    if args.reasoning_effort:
        body['reasoning_effort'] = args.reasoning_effort
    if args.temperature is not None:
        body['temperature'] = args.temperature
    body.update(getattr(args, 'extra_body', None) or {})
    key = os.environ[args.key_env]  # read before any request: a missing key is a setup error and is raised at once, never retried
    last = 'no attempt'
    for attempt in range(4):
        # Only a request that was not answered, or was refused for load, is sent again.
        try:
            response = requests.post(args.base_url.rstrip('/')+'/chat/completions', json=body, timeout=timeout,
                                     headers={'Authorization': 'Bearer '+key})
        except requests.RequestException as error:
            last = f'{type(error).__name__}: {error}'[:300]
            time.sleep(5*(attempt+1))
            continue
        if response.status_code == 429 or response.status_code >= 500:
            last = f'HTTP {response.status_code}: {response.text[:300]}'
            time.sleep(5*(attempt+1))
            continue
        if response.status_code >= 400:
            return dict(error=f'HTTP {response.status_code}: {response.text[:300]}')
        return read_reply(response)
    return dict(error=last)


def read_reply(response):
    """A served reply as `call`'s result, or an error naming the first thing it lacks.

    A reply that arrived was billed, so it is never asked for again: asking again buys the same wrong shape.
    Its error carries the reply's `usage` (None when the reply had none) so the caller's meter still counts it.
    A reply with no finish reason is an error too: without it nobody can say whether the answer was cut off.
    """
    try:
        data = response.json()
    except ValueError as error:
        return dict(error=f'reply body is not JSON ({error}); not asked again'[:300], usage=None)
    if not isinstance(data, dict):
        return dict(error=f'reply is a JSON {type(data).__name__}, not an object; not asked again', usage=None)
    usage = data['usage'] if 'usage' in data else None
    at, value = 'reply', data
    for step in ('choices', 0, 'message', 'content'):
        inside = step in value if isinstance(value, dict) else isinstance(value, list) and len(value) > step
        if not inside:
            return dict(error=f'{at} has no {step!r}; not asked again', usage=usage)
        at, value = f'{at}[{step!r}]', value[step]
    choice = data['choices'][0]
    if 'finish_reason' not in choice or not isinstance(choice['finish_reason'], str):
        return dict(error="reply['choices'][0] has no 'finish_reason', so a cut-off answer could not be told from a whole one; not asked again", usage=usage)
    finish, content = choice['finish_reason'], choice['message']['content']
    if content is None and finish == 'length':
        content = ''  # the token limit ran out before any answer text (all of it went on reasoning): a cut-off answer with nothing in it
    if not isinstance(content, str):
        return dict(error=f'reply content is {content!r} with finish_reason {finish!r}, not text; not asked again'[:300], usage=usage)
    return dict(raw=content, finish=finish, usage=usage, model=data.get('model'))


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
    """Load an MLX model once and return a non-thinking prompt-to-text callable."""
    from mlx_lm import generate, load
    model, tokenizer = load(model_id, revision=revision)

    def ask(prompt):
        ids = tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}], add_generation_prompt=True,
                                             tokenize=False, enable_thinking=False)
        return generate(model, tokenizer, ids, max_tokens=max_tokens, verbose=False)
    return ask
