"""Greedy decoding for the adapter's startup control."""
from copy import copy
from functools import wraps


GPT_RELOAD_CUE = (
    "Copy the following line exactly, including capitalization and punctuation.\n"
    "The quick brown fox jumps over the lazy dog.\n"
    "Exact copy: The quick"
)


def greedy_reload(verify):
    """Run the trainer's assertions on the same server with a diagnostic-only client view."""
    @wraps(verify)
    def check(backend, prompt):
        probe = copy(backend)

        def request(method, path, **kwargs):
            if path == '/inference/v1/generate':
                body = kwargs['body']
                kwargs['body'] = {**body, 'sampling_params': {**body['sampling_params'], 'temperature': 0.0}}
            return backend.request(method, path, **kwargs)

        probe.request = request
        if backend.args.model == 'openai/gpt-oss-20b':
            # Open-ended reasoning can have nearly tied next tokens on unchanged
            # MXFP4 weights. Use the same short input for every diagnostic arm;
            # the real perturbations must still change its greedy continuation.
            reply = backend.request('POST', '/tokenize', body={
                'model': f'{backend.args.model}@{backend.args.revision}',
                'prompt': GPT_RELOAD_CUE, 'add_special_tokens': False,
            }).json()
            prompt = reply['tokens']
            if not prompt or not all(type(token) is int and token >= 0 for token in prompt):
                raise ValueError('invalid tokenized GPT reload control')
        print('Adapter reload control: greedy decoding (temperature=0)', flush=True)
        checks = verify(probe, prompt)
        backend.checked_reload = probe.checked_reload
        return checks

    check.greedy_reload_probe = True
    return check
