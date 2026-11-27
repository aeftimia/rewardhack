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
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlsplit

import httpx

from .sampling import TEMPERATURE, TOP_K, TOP_P

SEED_STRIDE = 10**6
ROLLOUT_REFERENCE_SECONDS = 330  # 3x longest measured 106 s, rounded up
ROLLOUT_REFERENCE_TOKENS = 2048
ROLLOUT_REFERENCE_BATCH = 16
RELOAD_PROBE_TOKENS = 8


def completion_timeout(prompt_count, max_tokens):
    """Scale the short-answer timing bound for answer length and queued batches.

    The 24-answer Qwen evaluation at 8,192 tokens exceeded the old fixed 330 s
    while its server was still generating normally. Budget full waves of the
    reference batch, including requests waiting behind the first wave.
    """
    return (ROLLOUT_REFERENCE_SECONDS * max(1, math.ceil(max_tokens / ROLLOUT_REFERENCE_TOKENS))
            * max(1, math.ceil(prompt_count / ROLLOUT_REFERENCE_BATCH)))


def pinned_vllm():
    """The vLLM version this client is written against: the one pin, in requirements-vllm.txt at the repository root,
    which the pod installs from. Read when a server is checked (`preflight`), not on import: the file is beside the
    package only in a checkout (the pod's editable install), and other projects import this module from a wheel."""
    pins = [line.split('==')[1].strip() for line in (Path(__file__).resolve().parents[2]/'requirements-vllm.txt').read_text().splitlines()
            if line.startswith('vllm==')]
    if len(pins) != 1:
        raise ValueError(f'requirements-vllm.txt must pin vllm exactly once, found {pins}')
    return pins[0]


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
        # the answer-length and batch-size bound in completion_timeout.
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
        version, pinned = self.request('GET', '/version').json()['version'], pinned_vllm()
        if version != pinned:
            raise ValueError(f'expected vLLM {pinned} (requirements-vllm.txt), got {version}')
        models = self.request('GET', '/v1/models').json()['data']
        expected = f'{self.args.model}@{self.args.revision}'
        if expected not in [model['id'] for model in models]:
            raise ValueError(f'rollout server does not advertise the pinned model {expected}')
        config = self.request('GET', '/server_info?config_format=json').json()['vllm_config']['model_config']
        self.max_model_len = config['max_model_len']
        if self.max_model_len <= self.args.max_completion_length:
            raise ValueError(f'vLLM context {self.max_model_len} leaves no room for a prompt beside {self.args.max_completion_length} answer tokens')
        # Sampling is untruncated (sampling.py), so the trainer's full-vocabulary log-probabilities are the sampler's own
        # and no per-token candidate set is needed; a server that sends one was started for some other setting.
        if config['logprobs_mode'] != 'processed_logprobs' or config['return_sampling_mask']:
            raise ValueError('vLLM requires processed_logprobs and no sampling mask (sampling is untruncated)')
        self.require_sleep(True)

    def load(self, path):
        self.request('POST', '/v1/load_lora_adapter', body={
            'lora_name': self.name, 'lora_path': str(path.resolve()), 'load_inplace': self.loaded})
        self.loaded = True
        if self.request('POST', '/reset_prefix_cache').json()['success'] is not True:
            raise RuntimeError('vLLM did not reset the prefix cache after adapter loading')

    def completions(self, prompts, *, seed, max_tokens, model):
        if not prompts:
            raise ValueError('rollout prompts must not be empty')
        longest = max(map(len, prompts))
        if longest + max_tokens > self.max_model_len:
            raise ValueError(f'a {longest}-token prompt and {max_tokens} answer tokens exceed the vLLM context of {self.max_model_len}')
        request_timeout = completion_timeout(len(prompts), max_tokens)
        def send(item):
            index, prompt = item
            params = dict(n=1, max_tokens=max_tokens, temperature=TEMPERATURE,
                          top_p=TOP_P, top_k=TOP_K,
                          seed=seed+index, stop_token_ids=[self.stops], ignore_eos=True,
                          skip_special_tokens=False, logprobs=0, output_kind=2)
            body = dict(model=model, request_id=str(uuid.uuid4()), token_ids=prompt, sampling_params=params)
            reply = self.request('POST', '/inference/v1/generate', body=body, timeout=request_timeout).json()
            if len(reply['choices']) != 1:
                raise ValueError('vLLM must return exactly one completion per request')
            return reply['choices'][0]
        # executor.map preserves prompt/group order despite out-of-order replies.
        with ThreadPoolExecutor(max_workers=min(32, len(prompts))) as pool:
            choices = list(pool.map(send, enumerate(prompts)))
        ids, probabilities = [], []
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
            ids.append(tokens)
            probabilities.append(values)
        # No sampling mask: TRL then normalises the trainer's log-probabilities over the whole vocabulary, as sampled.
        return ids, probabilities, None

    def verify_reload(self, prompt):
        """A/B/A, once per kind of adapted weight (attention and other linear layers; the MoE experts): changing only
        that kind's LoRA B bytes must change the fixed-seed token sequence, and restoring A must restore it. Checked
        separately so that a server that ignores the expert tensors cannot pass on the attention ones. The live
        training tensors are untouched.
        """
        from safetensors.torch import load_file, save_file
        import shutil
        probe_path = self.args.out / 'adapter-sync-check'
        probe_path.mkdir(exist_ok=True)
        shutil.copyfile(self.path/'adapter_config.json', probe_path/'adapter_config.json')
        names = load_file(str(self.path/'adapter_model.safetensors')).keys()
        kinds = {kind: [name for name in names if 'lora_B' in name and ('.experts.' in name) == experts]
                 for kind, experts in {'linear': False, 'experts': True}.items()}
        if not kinds['linear']:
            raise ValueError('adapter has no LoRA B matrices for the synchronization check')
        def sample():
            return self.completions([prompt], seed=0, max_tokens=RELOAD_PROBE_TOKENS, model=self.name)
        def matches(left, right):
            return left[0] == right[0]
        def record_check(row):
            with (self.args.out / 'adapter-reload-checks.jsonl').open('a') as sink:
                sink.write(json.dumps(dict(prompt_ids=prompt, max_tokens=RELOAD_PROBE_TOKENS,
                                           **row), allow_nan=False) + '\n')
        # Establish repeatability separately from replacement. Reset the cache
        # identically before every observation; never infer a reload defect from
        # a comparison that already drifts with unchanged weights.
        def reset_and_sample():
            if self.request('POST', '/reset_prefix_cache').json()['success'] is not True:
                raise RuntimeError('vLLM did not reset the diagnostic prefix cache')
            return sample()
        before = reset_and_sample()
        for stage in ('unchanged-repeat', 'identical-reload'):
            if stage == 'identical-reload':
                self.load(self.path)
            after = reset_and_sample()
            matched = matches(before, after)
            record_check(dict(stage=stage, before=before, after=after, matched=matched))
            if not matched:
                raise RuntimeError(f'adapter repeatability check failed at {stage}')
        checks = {}
        for kind, changed in kinds.items():
            if not changed:
                continue
            state = load_file(str(self.path/'adapter_model.safetensors'))
            for name in changed:
                state[name].fill_(0.1)
            save_file(state, str(probe_path/'adapter_model.safetensors'))
            before = reset_and_sample()
            self.load(probe_path)
            try:
                changed_output = reset_and_sample()
            finally:
                self.load(self.path)
            restored = reset_and_sample()
            changed_ok, restored_ok = not matches(before, changed_output), matches(before, restored)
            checks[kind] = dict(tensors=len(changed), before=before, changed=changed_output, restored=restored,
                                changed_ok=changed_ok, restored_ok=restored_ok)
            # Persist before rejecting: a failed check must retain the evidence
            # distinguishing insensitive perturbations from failed restoration.
            record_check(dict(kind=kind, **checks[kind]))
            if not changed_ok or not restored_ok:
                raise RuntimeError(f'adapter A/B/A reload of the {kind} tensors failed: '
                                   f'perturbation_changed={changed_ok}, original_restored={restored_ok}')
        self.checked_reload = True
        return checks

    def generate(self, trainer, prompts, num_generations):
        if trainer.accelerator.num_processes != 1 or trainer.tools:
            raise ValueError('external adapter rollouts support one process and no tools')
        if len(prompts) % num_generations or any(
                group != prompts[start] for start in range(0, len(prompts), num_generations)
                for group in prompts[start:start+num_generations]):
            raise ValueError('TRL must supply contiguous repeated prompt groups')
        # `step` counts from 1, as the rollout rows and steps.jsonl do. Each run's seeds start SEED_STRIDE apart, so runs whose
        # --seed differs by one do not share request seeds a slot apart (a run uses far fewer than SEED_STRIDE requests).
        return self.sample(trainer.model, prompts, seed=self.args.seed * SEED_STRIDE + trainer.state.global_step * len(prompts),
                           record=dict(step=trainer.state.global_step + 1))

    def sample(self, adapter_model, prompts, *, seed, record):
        """Wake the server, load `adapter_model`'s current adapter (None: sample the served base model), sample each
        prompt once with seeds from `seed`, and put the server back to sleep whatever happens. Training rollouts and the
        before- and after-evaluations all come through here; each call is a line of rollout-backend.jsonl."""
        import torch
        began = time.monotonic()
        record = dict(record, seed=seed, adapter_sha256=None, status='failed')
        if adapter_model is not None:
            adapter_model.save_pretrained(self.path, safe_serialization=True)
            record['adapter_sha256'] = hashlib.sha256((self.path/'adapter_model.safetensors').read_bytes()).hexdigest()
        # No live activations exist between steps. Return cached blocks to CUDA
        # before waking the second process; its weights sleep during the update.
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        self.require_sleep(True)
        try:
            self.request('POST', '/wake_up')
            self.require_sleep(False)
            if adapter_model is not None:
                self.load(self.path)
                if not self.checked_reload:
                    record['reload_check'] = self.verify_reload(prompts[0])
            record['sync_seconds'] = time.monotonic() - began
            served = self.name if adapter_model is not None else f'{self.args.model}@{self.args.revision}'
            ids, logprobs, masks = self.completions(prompts, seed=seed, max_tokens=self.args.max_completion_length, model=served)
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
