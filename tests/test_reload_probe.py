from types import SimpleNamespace

import pytest

from rewardhack.reload_probe import GPT_RELOAD_CUE, greedy_reload


@pytest.mark.parametrize('fail', [False, True])
def test_gpt_control_only_changes_diagnostic_input_and_temperature(fail):
    calls = []
    def request(method, path, *, body):
        calls.append((path, body))
        return SimpleNamespace(json=lambda:{'tokens':[3, 4]})
    backend = SimpleNamespace(args=SimpleNamespace(model='openai/gpt-oss-20b', revision='pinned'),
                              request=request, checked_reload=False)
    @greedy_reload
    def verify(probe, prompt):
        assert prompt == [3, 4]
        probe.request('POST', '/inference/v1/generate', body={'sampling_params':{'temperature':1.0}})
        if fail:
            raise RuntimeError('expert restoration failed')
        probe.checked_reload = True
    if fail:
        with pytest.raises(RuntimeError, match='expert restoration failed'):
            verify(backend, [9])
    else:
        verify(backend, [9])
    assert backend.checked_reload == (not fail)
    assert backend.request is request
    assert calls == [('/tokenize', {'model':'openai/gpt-oss-20b@pinned', 'prompt':GPT_RELOAD_CUE,
                                    'add_special_tokens':False}),
                     ('/inference/v1/generate', {'sampling_params':{'temperature':0.0}})]
