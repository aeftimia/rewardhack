"""LoRA-only rollouts from a sleeping local vLLM server in a separate environment.

TRL 1.19.0's native backend merges and transfers the entire model and imports vLLM
into the training environment. Here only generation is replaced; TRL still owns
log-probability recomputation, importance ratios, rewards, and the GRPO loss.
"""
import hashlib
import importlib.metadata
import json
import math
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlsplit

import httpx


def add_arguments(parser):
    parser.add_argument('--rollout-backend', choices=['transformers', 'vllm'], default='transformers')
    parser.add_argument('--rollout-url', default='http://127.0.0.1:8011')


def check_args(args):
    if args.rollout_backend == 'vllm':
        if args.mode != 'train' or args.quant != 'none':
            raise ValueError('vLLM rollouts require training with unquantized training weights')
        if not args.revision:
            raise ValueError('vLLM rollouts require a pinned model revision')
        url = urlsplit(args.rollout_url)
        if url.scheme != 'http' or url.hostname != '127.0.0.1' or not url.port or url.path or url.query or url.fragment or url.username:
            raise ValueError('the rollout server must be a local http://127.0.0.1:port sharing the adapter directory')
        if importlib.metadata.version('trl') != '1.19.0':
            raise ValueError('the external rollout integration requires TRL 1.19.0')


class AdapterRollouts:
    def __init__(self, args, stops):
        self.args, self.stops = args, stops
        # 3x estimated 40 s for an adapter operation; completion requests use
        # 3x the longest measured 106 s rollout, rounded up to 330 s.
        self.client = httpx.Client(base_url=args.rollout_url, timeout=120, trust_env=False)
        # One name per training process: runs share one server (Part 1 trains two arms in a row), and vLLM refuses a name
        # already registered unless the load is in place, which a new process's first load is not.
        self.name = f'training-adapter-{uuid.uuid4().hex[:12]}'
        self.loaded = False
        self.checked_reload = False
        self.path = args.out / 'adapter-rollout'
        self.path.mkdir(parents=True, exist_ok=True)
        self.log = args.out / 'rollout-backend.jsonl'

    def request(self, method, path, *, body=None, timeout=120):
        response = self.client.request(method, path, json=body, timeout=timeout)
        response.raise_for_status()
        return response

    def require_sleep(self, sleeping):
        actual = self.request('GET', '/is_sleeping').json()['is_sleeping']
        if actual is not sleeping:
            raise RuntimeError(f'vLLM sleep state is {actual!r}; expected {sleeping}')

    def preflight(self):
        version = self.request('GET', '/version').json()['version']
        if version != '0.34.0':
            raise ValueError(f'expected vLLM 0.34.0, got {version}')
        models = self.request('GET', '/v1/models').json()['data']
        expected = f'{self.args.model}@{self.args.revision}'
        if expected not in [model['id'] for model in models]:
            raise ValueError(f'rollout server does not advertise the pinned model {expected}')
        config = self.request('GET', '/server_info?config_format=json').json()['vllm_config']['model_config']
        if config['logprobs_mode'] != 'processed_logprobs' or not config['return_sampling_mask']:
            raise ValueError('vLLM requires processed_logprobs and return_sampling_mask')
        self.require_sleep(True)

    def load(self, path):
        self.request('POST', '/v1/load_lora_adapter', body={
            'lora_name': self.name, 'lora_path': str(path.resolve()), 'load_inplace': self.loaded})
        self.loaded = True
        if self.request('POST', '/reset_prefix_cache').json()['success'] is not True:
            raise RuntimeError('vLLM did not reset the prefix cache after adapter loading')

    def completions(self, prompts, *, seed, max_tokens):
        if not prompts:
            raise ValueError('rollout prompts must not be empty')
        def send(item):
            index, prompt = item
            params = dict(n=1, max_tokens=max_tokens, temperature=1.0,
                          top_p=0.95, top_k=20,
                          seed=seed+index, stop_token_ids=[self.stops], ignore_eos=True,
                          skip_special_tokens=False, logprobs=0, output_kind=2)
            body = dict(model=self.name, request_id=str(uuid.uuid4()), token_ids=prompt, sampling_params=params)
            reply = self.request('POST', '/inference/v1/generate', body=body, timeout=330).json()
            if len(reply['choices']) != 1:
                raise ValueError('vLLM must return exactly one completion per request')
            return reply['choices'][0]
        # executor.map preserves prompt/group order despite out-of-order replies.
        with ThreadPoolExecutor(max_workers=min(32, len(prompts))) as pool:
            choices = list(pool.map(send, enumerate(prompts)))
        ids, probabilities, masks = [], [], []
        for choice in choices:
            tokens = choice['token_ids']
            logs = choice['logprobs']['content']
            values = [entry['logprob'] for entry in logs]
            if (not tokens or len(tokens) > max_tokens or len(tokens) != len(values)
                    or [entry['token_id'] for entry in logs] != tokens
                    or not all(type(token) is int and token >= 0 for token in tokens)
                    or not all(isinstance(value, (int, float)) and math.isfinite(value) and value <= 0 for value in values)):
                raise ValueError('vLLM returned invalid or misaligned token IDs/log probabilities')
            if choice['finish_reason'] not in ('stop', 'length'):
                raise ValueError(f"unexpected finish reason {choice['finish_reason']!r}")
            if choice['finish_reason'] == 'stop' and tokens[-1] != self.stops:
                raise ValueError('vLLM stopped without returning a configured stop token')
            if choice['finish_reason'] == 'length' and len(tokens) != max_tokens:
                raise ValueError('vLLM reached a length limit before the requested completion budget')
            if self.stops in tokens[:-1]:
                raise ValueError('vLLM continued past the configured stop token')
            mask = choice['sampling_mask']
            if len(mask) != len(tokens) or any(
                    not 1 <= len(support) <= 20 or token not in support or len(set(support)) != len(support)
                    or any(type(candidate) is not int or candidate < 0 for candidate in support)
                    for token, support in zip(tokens, mask, strict=True)):
                raise ValueError('vLLM returned an invalid sampling mask')
            masks.append(mask)
            ids.append(tokens)
            probabilities.append(values)
        return ids, probabilities, masks

    def verify_reload(self, prompt):
        """A/B/A: changed adapter bytes must change fixed-seed output/logprob,
        and restoring A must restore it. The live training tensors are untouched.
        """
        from safetensors.torch import load_file, save_file
        import shutil
        probe_path = self.args.out / 'adapter-sync-check'
        probe_path.mkdir(exist_ok=True)
        shutil.copyfile(self.path/'adapter_config.json', probe_path/'adapter_config.json')
        state = load_file(str(self.path/'adapter_model.safetensors'))
        changed = 0
        for name, tensor in state.items():
            if 'lora_B' in name:
                tensor.fill_(0.1)
                changed += 1
        if not changed:
            raise ValueError('adapter has no LoRA B matrices for the synchronization check')
        save_file(state, str(probe_path/'adapter_model.safetensors'))
        def sample():
            # Replay requires temperature > 0 and top_k > 0 even for this probe.
            return self.completions([prompt], seed=0, max_tokens=1)
        before = sample()
        self.load(probe_path)
        changed_output = sample()
        self.load(self.path)
        restored = sample()
        def matches(left, right):
            return left[0] == right[0] and math.isclose(left[1][0][0], right[1][0][0], abs_tol=1e-4, rel_tol=0)
        if matches(before, changed_output) or not matches(before, restored):
            raise RuntimeError('adapter A/B/A reload did not change and then restore the sampled token/logprob')
        self.checked_reload = True
        return dict(before=before, changed=changed_output, restored=restored)

    def generate(self, trainer, prompts, num_generations):
        import torch
        if trainer.accelerator.num_processes != 1 or trainer.tools:
            raise ValueError('external adapter rollouts support one process and no tools')
        if len(prompts) % num_generations or any(
                group != prompts[start] for start in range(0, len(prompts), num_generations)
                for group in prompts[start:start+num_generations]):
            raise ValueError('TRL must supply contiguous repeated prompt groups')
        began = time.monotonic()
        trainer.model.save_pretrained(self.path, safe_serialization=True)
        digest = hashlib.sha256((self.path/'adapter_model.safetensors').read_bytes()).hexdigest()
        record = dict(step=trainer.state.global_step, adapter_sha256=digest,
                      seed=self.args.seed + trainer.state.global_step * len(prompts), status='failed')
        # No live activations exist between steps. Return cached blocks to CUDA
        # before waking the second process; its weights sleep during the update.
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        self.require_sleep(True)
        try:
            self.request('POST', '/wake_up')
            self.require_sleep(False)
            self.load(self.path)
            if not self.checked_reload:
                record['reload_check'] = self.verify_reload(prompts[0])
            record['sync_seconds'] = time.monotonic() - began
            ids, logprobs, masks = self.completions(prompts, seed=record['seed'], max_tokens=self.args.max_completion_length)
            record['status'] = 'completed'
        finally:
            # A failed sleep is fatal too: never proceed into backward with the
            # server awake. A cleanup exception retains the original as context.
            self.request('POST', '/sleep?level=1')
            self.require_sleep(True)
            record['seconds'] = time.monotonic() - began
            with self.log.open('a') as sink:
                sink.write(json.dumps(record, allow_nan=False) + '\n')
        return ids, logprobs, masks

    def close(self):
        self.client.close()
