"""Greedy decoding for the adapter's startup control."""
from copy import copy
from functools import wraps


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
        print('Adapter reload control: greedy decoding (temperature=0)', flush=True)
        checks = verify(probe, prompt)
        backend.checked_reload = probe.checked_reload
        return checks

    check.greedy_reload_probe = True
    return check
