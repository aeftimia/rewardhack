"""External HTTP protocol failures and real TRL updates using replayed support."""
import json
import math
import time
from types import SimpleNamespace

import httpx
import pytest

from rewardhack.vllm_rollouts import AdapterRollouts, check_args


@pytest.fixture
def backend(tmp_path):
    args = SimpleNamespace(out=tmp_path, rollout_url='http://127.0.0.1:8011', model='model', revision='revision',
                           mode='train', quant='none', rollout_backend='vllm', seed=7, max_completion_length=2)
    backend = AdapterRollouts(args, 1)
    backend.max_model_len = 16  # what preflight reads from the server
    yield backend
    backend.close()


def connect(backend, handler):
    backend.client.close()
    backend.client = httpx.Client(base_url=backend.args.rollout_url, transport=httpx.MockTransport(handler))


def choice(token=3):
    return dict(token_ids=[token, 1], finish_reason='stop',
                logprobs={'content': [dict(token_id=t, logprob=-math.log(2)) for t in [token, 1]]})


def test_parallel_responses_preserve_group_order_and_unique_seeds(backend):
    seen = []
    def handler(request):
        body = json.loads(request.content)
        seen.append(body)
        index = body['sampling_params']['seed']-7
        time.sleep((3-index)*0.005)
        return httpx.Response(200, json={'choices': [choice(3+index % 2)]})
    connect(backend, handler)
    ids, logs, masks = backend.completions([[8], [8], [9], [9]], seed=7, max_tokens=2, model='m')
    assert ids == [[3, 1], [4, 1], [3, 1], [4, 1]]
    assert masks is None and len(logs) == 4
    assert all((b['sampling_params']['temperature'], b['sampling_params']['top_p'], b['sampling_params']['top_k']) == (1.0, 1.0, 0) for b in seen)
    assert sorted(b['sampling_params']['seed'] for b in seen) == [7, 8, 9, 10]
    assert all(b['sampling_params']['ignore_eos'] and b['sampling_params']['stop_token_ids'] == [1] for b in seen)
    assert all(b['sampling_params']['output_kind'] == 2 for b in seen)


@pytest.mark.parametrize('damage', ['misaligned', 'nonfinite', 'missing_stop', 'extra', 'http'])
def test_invalid_rollouts_fail_instead_of_training(backend, damage):
    row = choice()
    if damage == 'misaligned': row['logprobs']['content'][0]['token_id'] = 4
    if damage == 'nonfinite': row['logprobs']['content'][0]['logprob'] = None
    if damage == 'missing_stop': row['token_ids'][-1] = 2; row['logprobs']['content'][-1]['token_id'] = 2
    connect(backend, lambda request: httpx.Response(503 if damage == 'http' else 200,
        json={'choices': [row, row] if damage == 'extra' else [row]}))
    with pytest.raises((ValueError, KeyError, httpx.HTTPStatusError)):
        backend.completions([[8]], seed=7, max_tokens=2, model='m')


@pytest.mark.parametrize('wrong', ['version', 'model', 'logprobs', 'mask', 'context', 'awake'])
def test_preflight_checks_server_contract(backend, wrong):
    responses = {
        '/version': {'version': 'wrong' if wrong == 'version' else '0.34.0'},
        '/v1/models': {'data': [{'id': 'wrong' if wrong == 'model' else 'model@revision'}]},
        '/server_info': {'vllm_config': {'model_config': {'logprobs_mode': 'raw_logprobs' if wrong == 'logprobs' else 'processed_logprobs',
                                                       'return_sampling_mask': wrong == 'mask',
                                                       'max_model_len': 2 if wrong == 'context' else 16}}},
        '/is_sleeping': {'is_sleeping': wrong != 'awake'},
    }
    connect(backend, lambda request: httpx.Response(200, json=responses[request.url.path]))
    with pytest.raises((ValueError, RuntimeError)):
        backend.preflight()


def test_generation_failure_sleeps_server_and_propagates(backend, monkeypatch):
    import torch
    from safetensors.torch import save_file
    class Model:
        def save_pretrained(self, path, **kwargs):
            save_file({'lora_B': torch.ones(1)}, str(path/'adapter_model.safetensors'))
    trainer = SimpleNamespace(model=Model(), state=SimpleNamespace(global_step=0), tools=None,
                              accelerator=SimpleNamespace(num_processes=1))
    backend.checked_reload = True
    sleeping, calls = [True], []
    def handler(request):
        calls.append(request.url.path)
        if request.url.path == '/wake_up': sleeping[0] = False
        if request.url.path == '/sleep': sleeping[0] = True
        if request.url.path == '/inference/v1/generate': return httpx.Response(503)
        return httpx.Response(200, json={'is_sleeping': sleeping[0], 'success': True})
    connect(backend, handler)
    with pytest.raises(httpx.HTTPStatusError):
        backend.generate(trainer, [[8], [8]], 2)
    assert sleeping[0] and calls[-2:] == ['/sleep', '/is_sleeping']
    assert json.loads(backend.log.read_text())['status'] == 'failed'


def test_reload_requires_changed_weights_and_restoration(backend, monkeypatch):
    import torch
    from safetensors.torch import save_file, load_file
    save_file({'layer.lora_B.weight': torch.zeros(2)}, str(backend.path/'adapter_model.safetensors'))
    (backend.path/'adapter_config.json').write_text('{}')
    active = [backend.path]
    monkeypatch.setattr(backend, 'load', lambda path: active.__setitem__(0, path))
    def sample(*args, **kwargs):
        value = float(load_file(str(active[0]/'adapter_model.safetensors'))['layer.lora_B.weight'].sum())
        return [[3]], [[-1-value]], None
    monkeypatch.setattr(backend, 'completions', sample)
    backend.verify_reload([8])
    assert backend.checked_reload and active[0] == backend.path
    assert float(load_file(str(backend.path/'adapter_model.safetensors'))['layer.lora_B.weight'].sum()) == 0
    monkeypatch.setattr(backend, 'completions', lambda *a, **kw: ([[3]], [[-1]], None))
    with pytest.raises(RuntimeError, match='A/B/A'):
        backend.verify_reload([8])


def test_untruncated_rollouts_reach_real_trl_loss(tmp_path, monkeypatch):
    torch = pytest.importorskip('torch')
    from test_training_integration import train, REWARDS, cosine
    from trl import GRPOTrainer
    real = GRPOTrainer._get_per_token_logps_and_entropies
    seen = []
    def inspect(self, *args, **kwargs):
        assert kwargs['sampling_mask'] is None
        result = real(self, *args, **kwargs)
        seen.append(result[0].detach().clone())
        # The tiny model is nearly uniform over its 16-token vocabulary: log(1/16) = -2.77, where a two-token
        # support would give log(1/2). Sampling is untruncated, so the trainer scores against the whole vocabulary.
        assert bool((result[0] < -2).all())
        return result
    monkeypatch.setattr(GRPOTrainer, '_get_per_token_logps_and_entropies', inspect)
    class Rollouts:
        def generate(self, trainer, prompts, num_generations):
            assert trainer.eos_token_ids == [self.stops] == [1]
            return ([[3+i % 2, 1] for i in range(len(prompts))],
                    [[-math.log(16)]*2 for _ in prompts], None)
    def run(reward, steps=1):
        return train(tmp_path, monkeypatch, reward, steps=steps, rollouts=Rollouts(), rollout_backend='vllm')
    positive = run(REWARDS['first of the group'])
    negative = run(lambda i, raw, step: -float(i == 0))
    assert cosine(positive.change, negative.change) < -0.99
    assert float(positive.change.abs().max()) > 1e-4
    assert float(run(lambda i, raw, step: 1.0).change.abs().max()) < 1e-7
    assert len(run(REWARDS['first of the group'], steps=2).log) == 2
    assert len(seen) >= 10
    assert 'sampling_mask' not in positive.summary['rollout_backend']


def test_external_rollouts_refuse_unsupported_configuration(backend):
    check_args(backend.args)
    backend.args.quant = '4bit'
    with pytest.raises(ValueError, match='unquantized'):
        check_args(backend.args)


def test_every_batch_reloads_current_adapter_and_sleeps(backend):
    import torch
    from safetensors.torch import save_file, load_file
    value, loaded, sleeping, seeds = [0.0], [0.0], [True], []
    class Model:
        def save_pretrained(self, path, **kwargs):
            (path/'adapter_config.json').write_text('{}')
            save_file({'layer.lora_B.weight': torch.tensor([value[0]])}, str(path/'adapter_model.safetensors'))
    trainer = SimpleNamespace(model=Model(), state=SimpleNamespace(global_step=0), tools=None,
                              accelerator=SimpleNamespace(num_processes=1))
    def handler(request):
        path = request.url.path
        if path == '/wake_up': sleeping[0] = False
        if path == '/sleep': sleeping[0] = True
        if path == '/v1/load_lora_adapter':
            from pathlib import Path
            folder = Path(json.loads(request.content)['lora_path'])
            loaded[0] = float(load_file(str(folder/'adapter_model.safetensors'))['layer.lora_B.weight'].sum())
        if path == '/inference/v1/generate':
            assert not sleeping[0]
            params = json.loads(request.content)['sampling_params']
            assert (params['temperature'], params['top_p'], params['top_k']) == (1.0, 1.0, 0)
            if params['max_tokens'] == 1:
                row = dict(token_ids=[3], finish_reason='length',
                           logprobs={'content': [dict(token_id=3, logprob=-1-loaded[0])]})
            else:
                seeds.append(params['seed'])
                assert loaded[0] == pytest.approx(value[0])
                row = choice()
            return httpx.Response(200, json={'choices': [row]})
        return httpx.Response(200, json={'is_sleeping': sleeping[0], 'success': True})
    connect(backend, handler)
    for step in range(2):
        trainer.state.global_step = step
        value[0] = step*0.2
        ids, logs, masks = backend.generate(trainer, [[8], [8]], 2)
        assert sleeping[0] and ids == [[3, 1]]*2 and masks is None
    records = [json.loads(line) for line in backend.log.read_text().splitlines()]
    assert all(row['status'] == 'completed' for row in records)
    assert records[0]['adapter_sha256'] != records[1]['adapter_sha256']
    assert 'reload_check' in records[0] and 'reload_check' not in records[1]
    # Run seed 7 starts at 7 * SEED_STRIDE; step 0's two prompts take two seeds and step 1's the next two.
    assert sorted(seeds) == [7_000_000, 7_000_001, 7_000_002, 7_000_003]
    assert [row['step'] for row in records] == [1, 2]  # counted from 1, as the rollout rows and steps.jsonl are


def test_each_training_process_registers_its_own_adapter_name(tmp_path, backend):
    """Part 1 trains two arms in a row on one server. vLLM refuses a registered name unless the load is in place, and a
    new process's first load is not: the second run must not reuse the first run's name."""
    registered = set()

    def server(request):
        if request.url.path == '/v1/load_lora_adapter':
            body = json.loads(request.content)
            if body['lora_name'] in registered and not body['load_inplace']:
                return httpx.Response(400, json={'error': 'already been loaded'})
            registered.add(body['lora_name'])
            return httpx.Response(200, json={})
        return httpx.Response(200, json={'success': True})

    second = AdapterRollouts(backend.args, 1)
    for process in (backend, second):
        connect(process, server)
        process.load(tmp_path)
        process.load(tmp_path)  # later loads in the same process are in place
    assert backend.name != second.name and len(registered) == 2
    # Control: a second process reusing the first one's name is refused by this server, as by vLLM.
    second.name, second.loaded = backend.name, False
    with pytest.raises(httpx.HTTPStatusError):
        second.load(tmp_path)
    second.close()


def test_reload_checks_expert_tensors_on_their_own(backend, monkeypatch):
    # A server that applies the attention adapter but ignores the experts must not pass on the attention tensors.
    import torch
    from safetensors.torch import save_file, load_file
    save_file({'layer.self_attn.q_proj.lora_B.weight': torch.zeros(2), 'layer.mlp.experts.lora_B.weight': torch.zeros(2)},
              str(backend.path/'adapter_model.safetensors'))
    (backend.path/'adapter_config.json').write_text('{}')
    active = [backend.path]
    monkeypatch.setattr(backend, 'load', lambda path: active.__setitem__(0, path))
    def served(*names):
        def sample(*args, **kwargs):
            state = load_file(str(active[0]/'adapter_model.safetensors'))
            return [[3]], [[-1-sum(float(state[name].sum()) for name in names)]], None
        return sample
    monkeypatch.setattr(backend, 'completions', served('layer.self_attn.q_proj.lora_B.weight', 'layer.mlp.experts.lora_B.weight'))
    checks = backend.verify_reload([8])
    assert {kind: check['tensors'] for kind, check in checks.items()} == {'linear': 1, 'experts': 1}
    monkeypatch.setattr(backend, 'completions', served('layer.self_attn.q_proj.lora_B.weight'))
    with pytest.raises(RuntimeError, match='experts tensors'):
        backend.verify_reload([8])


def test_a_request_longer_than_the_server_context_is_refused_before_it_is_sent(backend):
    """A fixed 4,096-token server context refused every request once the answer limit was 4,096 (review round 6)."""
    connect(backend, lambda request: pytest.fail('an over-long request reached the server'))
    with pytest.raises(ValueError, match='exceed the vLLM context'):
        backend.completions([[8]*15], seed=7, max_tokens=2, model='m')


def test_sampling_without_an_adapter_asks_the_served_base_model_and_loads_nothing(backend):
    """The before-evaluation samples the base model through the same server (owner, 26 Nov)."""
    sleeping, seen = [True], []
    def handler(request):
        path = request.url.path
        if path == '/wake_up': sleeping[0] = False
        if path == '/sleep': sleeping[0] = True
        assert path != '/v1/load_lora_adapter', 'no adapter to load'
        if path == '/inference/v1/generate':
            assert not sleeping[0]
            seen.append(json.loads(request.content))
            return httpx.Response(200, json={'choices': [choice()]})
        return httpx.Response(200, json={'is_sleeping': sleeping[0], 'success': True})
    connect(backend, handler)
    ids, _, masks = backend.sample(None, [[8], [8], [9]], seed=100, record=dict(evaluation='before'))
    assert sleeping[0] and len(ids) == 3 and masks is None
    assert {body['model'] for body in seen} == {'model@revision'}
    assert sorted(body['sampling_params']['seed'] for body in seen) == [100, 101, 102]
    record = json.loads(backend.log.read_text().splitlines()[-1])
    assert record['evaluation'] == 'before' and record['adapter_sha256'] is None and record['status'] == 'completed'
