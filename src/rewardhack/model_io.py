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
    last = 'no attempt'
    for attempt in range(4):
        try:
            response = requests.post(args.base_url.rstrip('/')+'/chat/completions', json=body, timeout=timeout,
                                     headers={'Authorization': 'Bearer '+os.environ[args.key_env]})
            if response.status_code == 429 or response.status_code >= 500:
                last = f'HTTP {response.status_code}: {response.text[:300]}'
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
        self.rates = (nonnegative_finite(usd_per_m_input, 'input token price'),
                      nonnegative_finite(usd_per_m_output, 'output token price'))
        self.totals = dict(input=0, output=0, reasoning=0, usd=0.0, requests=0, errors=0)

    def add(self, usage):
        reasoning = usage.get('completion_tokens_details', {}).get('reasoning_tokens', 0)
        self.totals['input'] += usage.get('prompt_tokens', 0)
        self.totals['output'] += usage.get('completion_tokens', 0)
        self.totals['reasoning'] += reasoning
        self.totals['usd'] = self.totals['input']*self.rates[0]/1e6+self.totals['output']*self.rates[1]/1e6
        return reasoning



def local_reader(model_id, revision, max_tokens):
    """Load an MLX model once and return a non-thinking prompt-to-text callable."""
    from mlx_lm import generate, load
    model, tokenizer = load(model_id, revision=revision)

    def ask(prompt):
        ids = tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}], add_generation_prompt=True,
                                             tokenize=False, enable_thinking=False)
        return generate(model, tokenizer, ids, max_tokens=max_tokens, verbose=False)
    return ask
