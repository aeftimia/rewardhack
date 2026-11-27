"""GPU path: measure exploration (`probe`) or train with TRL GRPO (`train`), recording cost-relevant timings.

Adapted from exit-reward-research@ec0565e scripts/grpo_lean_pilot.py. Arms share one prompt; only the
reward differs. Entrypoint: `entrypoints/run.py`.
"""
import hashlib
import importlib.metadata
import json
import math
import resource
import subprocess
import sys
import time
from pathlib import Path

from . import deadline, jsonl, provenance
from ._hashing import file_sha256, source_sha256
from .gpu import GpuSampler
from .outcomes import accumulate, compare, judge_advice, judge_failing, verifier_failing, NO_VARIATION, VERIFIER_UNDECIDED, SATURATED, saturated, RetryBudget, ScoringStopped, verifier_advice, completion_truncated, reward_groups, score, score_concurrently, stop_ids, summarize, tally, training_rewards, training_keep
from .prompts import prompt
from .rates import observed_ratio
from .sampling import TEMPERATURE, TOP_K, TOP_P
from .tasks import select_tasks, stratified_limit
from .verifier import controls


def stage_tasks(taskset, stage, limit):
    """Select the ordered task groups for a trainer stage or its inference replay."""
    return stratified_limit(select_tasks(taskset, 'all', provenance.STAGES[stage]['part']), limit)


class VerifierMaskMixin:
    """Mask undecided rollouts after TRL normalization, including the independent KL loss.

    The reward callback sets a local mask in the same order as its completions.
    Keeping it with each generated batch also handles TRL's buffered minibatches.
    """
    def _generate_and_score_completions(self, inputs):
        self._verifier_keep = None
        batch = super()._generate_and_score_completions(inputs)
        import torch
        keep = self._verifier_keep
        if keep is None or len(keep) != len(batch['advantages']):
            raise ValueError('Verifier mask must match the generated local completions')
        keep = torch.tensor(keep, dtype=torch.bool, device=batch['advantages'].device)
        batch['advantages'] = batch['advantages'].masked_fill(~keep, 0.0)
        # tool_mask affects the loss only, leaving attention to generated tokens intact.
        mask = keep[:, None].expand_as(batch['completion_mask'])
        batch['tool_mask'] = batch.get('tool_mask', torch.ones_like(batch['completion_mask'])) * mask
        # The 'dapo' loss divides by the tokens of the whole step (`num_items_in_batch`, counted by TRL before this mask).
        # Recount without the dropped rows, so the kept rows train exactly as if the dropped ones had never been sampled,
        # whichever micro-batch they fall in.
        batch['num_items_in_batch'] = self.accelerator.gather((batch['completion_mask'] * batch['tool_mask']).sum()).sum()
        return batch

    def _compute_loss(self, model, inputs):
        loss = super()._compute_loss(model, inputs)
        import torch
        if not bool(torch.isfinite(loss)):
            # Here, before the optimiser step, so the adapter is still the last good one. The step callback is too late and too
            # early at once: it runs after the update and before the step's loss is logged.
            raise FloatingPointError(f'the training loss is {float(loss)} at step {self.state.global_step+1}')
        return loss


EVALUATION_SEED_OFFSET = 31415


def evaluation_seed(args):
    return EVALUATION_SEED_OFFSET + args.seed


# What a training trajectory depends on. A run resumed from a checkpoint must have the same values, or it would continue
# a different experiment from the saved optimizer state. `--steps` is left out: raising it is how a run is extended.
# The judge decides the report_correct reward, the verifier's retry allowance decides which slow proofs are scored, the
# attention and expert implementations change the numbers, and `adapter` is where a branch run started.
RESUME_SETTINGS = ('model', 'revision', 'arm', 'penalty', 'report_mode', 'taskset', 'train_limit', 'prompt_variant', 'reasoning_effort',
                   'thinking', 'mxfp4', 'quant', 'attn_implementation', 'experts_implementation', 'lora_rank', 'learning_rate',
                   'lr_schedule', 'scale_rewards', 'max_grad_norm', 'router_aux_loss_coef', 'beta', 'num_generations',
                   'prompts_per_step', 'max_completion_length', 'seed', 'save_every', 'stop_no_variation', 'stop_saturated', 'rollout_backend',
                   'judge', 'judge_model', 'judge_revision', 'judge_url', 'judge_extra', 'verifier_retry_seconds', 'adapter',
                   # The resumed run carries the original's before-evaluation over, so its after-evaluation must match it.
                   'eval_limit', 'eval_samples')
RUN_SETTINGS = 'run-settings.json'


# The libraries whose code decides the update, the task order and the checkpoint format.
TRAINING_LIBRARIES = ('torch', 'transformers', 'trl', 'peft', 'accelerate', 'datasets')


def run_settings(args):
    """The settings as JSON values (a path is saved as its text), with the code and the training libraries' versions: a
    resume after a change to scoring or training would continue the saved optimizer state on a different objective."""
    libraries = {name: importlib.metadata.version(name) for name in TRAINING_LIBRARIES}
    return json.loads(json.dumps(dict({key: vars(args)[key] for key in RESUME_SETTINGS}, code_sha256=source_sha256(), libraries=libraries,
                                      steps=args.steps), default=str))


# What a checkpoint must hold to continue a run: the adapter, the optimizer, the scheduler and the random state.
CHECKPOINT_FILES = ('adapter_model.safetensors', 'adapter_config.json', 'optimizer.pt', 'scheduler.pt', 'rng_state.pth')
# What a resume reads from the run folder beside the checkpoint (check_resume, original_records); ops/run_archive.py fetches
# exactly these and the checkpoint for a pod.
ROLLOUTS, STEPS = 'training-rollouts.jsonl', 'steps.jsonl'
RESUME_RUN_FILES = (RUN_SETTINGS, ROLLOUTS, STEPS, 'summary.json', 'before-evaluation.jsonl')


def resume_point(args):
    """Where a resume starts: the run that wrote the checkpoint (<run>/trainer/checkpoint-N), the steps it had finished,
    and its whole passes over the tasks (trainer_state's `epoch` is exact at a pass's end)."""
    state = json.loads((args.resume_from/'trainer_state.json').read_text())
    return args.resume_from.parent.parent, state['global_step'], math.floor(state['epoch'])


def original_records(args):
    """The original run's rollouts and step records up to its checkpoint, read no further than needed: a run killed from
    outside may end in a half-written line, past its checkpoint. Rows of a failed step's partial batch are left out."""
    original, done, _ = resume_point(args)
    wanted = done*args.num_generations*args.prompts_per_step
    rows, lines = [], []
    # Each loop stops on the last line it needs, before parsing the next one.
    for row in jsonl.read(original/ROLLOUTS):
        if 'partial_batch' not in row:
            rows.append(row)
        if len(rows) == wanted:
            break
    for line in jsonl.read(original/STEPS):
        lines.append(line)
        if line['step'] == done:
            break
    if len(rows) != wanted or [line['step'] for line in lines] != list(range(1, done+1)):
        raise ValueError(f'{original} does not hold the records of the {done} steps before its checkpoint')
    return rows, lines


def check_resume(args):
    """Refuse a resume that would not continue the run that wrote the checkpoint (<run>/trainer/checkpoint-N).

    Called before the model is loaded, so a wrong resume costs nothing on a pod. `run.py` allows --resume-from only in train mode."""
    if not (args.resume_from/'trainer_state.json').is_file():
        raise ValueError(f'--resume-from {args.resume_from} is not a trainer checkpoint: it has no trainer_state.json')
    # transformers resumes without a word from a checkpoint missing its optimizer or scheduler (a fresh one) or its random
    # state (a warning): a copy caught mid-rotation would continue as a different run.
    missing = [name for name in CHECKPOINT_FILES if not (args.resume_from/name).is_file()]
    if missing:
        raise ValueError(f"--resume-from {args.resume_from} is incomplete: no {', '.join(missing)}")
    original, done, _ = resume_point(args)
    if original.resolve() == args.out.resolve():
        raise ValueError(f'--out is the run that wrote the checkpoint; a resumed run writes a new folder, or the records of {original} would be overwritten')
    if not (original/RUN_SETTINGS).is_file():
        raise ValueError(f'{original} has no {RUN_SETTINGS}: it was written before runs could be resumed')
    saved = json.loads((original/RUN_SETTINGS).read_text())
    # `run` writes summary.json as it starts and again as it ends; one killed from outside keeps the first, without an
    # early stop, which only a run that ended itself can record.
    outcome = json.loads((original/'summary.json').read_text())
    if 'early_stop' in outcome and outcome['early_stop'] in (NO_VARIATION, VERIFIER_UNDECIDED, SATURATED):
        # Its replayed groups have no variation (or too little), so a resume would pay for setup and one step and then
        # stop again.
        raise ValueError(f"{original} stopped for {outcome['early_stop']!r}; resuming it would stop again after one step")
    if saved['adapter'] is not None:
        raise ValueError(f"{original} started from --adapter {saved['adapter']}; resuming it would measure its before-evaluation on the base model")
    now = run_settings(args)
    # The code first: a change to what is checked (the library list, the settings) is itself a change to the code.
    code_then, code_now = saved.pop('code_sha256'), now.pop('code_sha256')
    if code_then != code_now:
        changed = sorted(name for name in set(code_then) | set(code_now) if code_then.get(name) != code_now.get(name))
        raise ValueError(f"--resume-from was written by other code; changed since: {', '.join(changed)}")
    libraries_then, libraries_now = saved.pop('libraries'), now.pop('libraries')
    if libraries_then != libraries_now:
        changed = sorted(f'{name} {libraries_then[name]} -> {libraries_now[name]}' for name in libraries_now if libraries_then[name] != libraries_now[name])
        raise ValueError(f"--resume-from was written with other training libraries: {', '.join(changed)}")
    steps_then = saved.pop('steps')  # raising --steps is how a run is extended, so it is not a setting that must match
    now.pop('steps')
    differ = {key: (saved[key], value) for key, value in now.items() if saved[key] != value}
    if differ:
        raise ValueError('--resume-from was written by a run with other settings (saved, now): '
                         + ', '.join(f'{key} {old!r} -> {new!r}' for key, (old, new) in differ.items()))
    if args.lr_schedule == 'linear' and args.steps != steps_then:
        # Its rate decays over --steps: another total would not match an uninterrupted run. The same total resumes fine.
        raise ValueError(f'a run with --lr-schedule linear cannot be extended: its rate decays to zero over its {steps_then} steps')
    if args.steps <= done:
        raise ValueError(f'--steps {args.steps} is the new total and must exceed the {done} steps the checkpoint has done')
    original_records(args)  # the records the resumed run will replay, read now rather than after a paid model load
    if not (original/'before-evaluation.jsonl').is_file():
        raise ValueError(f'{original} has no before-evaluation to carry over')


PROMPT_DATE = '2026-11-24'  # the day of the first unhinted probe; every prompt since carries it
MODES = {'train': dict(training=True, controls=False, probe=False),
         'controls': dict(training=False, controls=True, probe=False),
         'probe': dict(training=False, controls=False, probe=True)}


def settle_template(tokenizer, reasoning_effort, thinking):
    """Make the chat template say the same thing on every day and refuse a reasoning switch it would ignore.

    The gpt-oss template writes "Current date: <today>" into the system message, so the prompt changed at midnight,
    between runs, and when transcripts were rebuilt later. It also falls back to "Reasoning: medium" when no effort
    is passed, and ignores Qwen's `enable_thinking` without a word.
    """
    template = tokenizer.chat_template
    if not isinstance(template, str) or not template:
        raise ValueError('a nonempty chat template is required to reproduce the prompt')
    if 'strftime_now' in template:
        pinned = template.replace('strftime_now("%Y-%m-%d")', f'"{PROMPT_DATE}"')
        if 'strftime_now' in pinned:
            raise ValueError('the chat template reads the clock in a form this code does not pin')
        tokenizer.chat_template = pinned
    if ('reasoning_effort' in template) != bool(reasoning_effort):
        raise ValueError('--reasoning-effort must be given exactly when the model\'s chat template takes one (gpt-oss); '
                         'unset, that template silently uses medium')
    if thinking != 'unsupported' and 'enable_thinking' not in template:
        raise ValueError('--thinking on/off needs a chat template with enable_thinking; pass --thinking unsupported for this model')
    return 'strftime_now' in template


def load_model(args, summary):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    training = MODES[args.mode]['training']
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.padding_side = 'left'
    if not tokenizer.chat_template and args.model.startswith('hf-internal-testing/'):
        tokenizer.chat_template = "{% for m in messages %}{{ m['role'] }}: {{ m['content'] }}\n{% endfor %}assistant:"  # CPU dry run only
    reads_clock = settle_template(tokenizer, args.reasoning_effort, args.thinking)
    rendered = tokenizer.apply_chat_template(prompt('True', args.prompt_variant, args.report_mode), tokenize=False, add_generation_prompt=True, **template_kwargs(args))
    summary['prompt'] = {'date_in_template': PROMPT_DATE if reads_clock else None,
                         'sha256_of_prompt_for_statement_True': hashlib.sha256(rendered.encode()).hexdigest()}
    kwargs = {'revision': args.revision, 'attn_implementation': args.attn_implementation}
    cuda = torch.cuda.is_available()
    kwargs['dtype'] = torch.bfloat16 if cuda else torch.float32
    quantization = {'none': None, '8bit': dict(load_in_8bit=True),
                    '4bit': dict(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                                 bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)}[args.quant]
    if quantization is not None:
        kwargs['quantization_config'] = BitsAndBytesConfig(**quantization)
    if args.mxfp4 == 'dequantize':
        # gpt-oss ships 4-bit (MXFP4) expert weights. Left to itself the loader keeps them when the `kernels` package is
        # installed and expands them to bf16 when it is not, with one warning line: 13 GiB or 39 GiB, trainable or not,
        # decided by what happens to be installed. Training needs the expanded weights, so that is asked for by name.
        if quantization is not None:
            raise ValueError('--mxfp4 dequantize cannot be combined with --quant')
        from transformers import Mxfp4Config
        kwargs['quantization_config'] = Mxfp4Config(dequantize=True)
    if cuda:
        kwargs['device_map'] = 'auto'
    t0 = time.perf_counter()
    # No retry with another attention path: an unsupported one raises here (gpt-oss in transformers 5.17 has
    # only `eager`), and the caller names the right one. Eager attention costs memory, so it must be a decision.
    model = AutoModelForCausalLM.from_pretrained(args.model, **kwargs)
    summary['attn_implementation'] = args.attn_implementation
    if args.experts_implementation:
        model.config._experts_implementation = args.experts_implementation  # read at every forward of a mixture-of-experts layer
    # What was actually loaded, so a library default that changed memory or speed can be seen in the summary afterwards.
    if hasattr(model, 'is_quantized'):
        quantized = model.is_quantized
        if type(quantized) is not bool:
            raise TypeError('the model quantization flag must be Boolean')
    else:
        # Unquantized transformers models omit this flag. Establish that state
        # from both the requested settings and the loaded model's metadata.
        if quantization is not None or hasattr(model, 'hf_quantizer') or model.config.to_dict().get('quantization_config') is not None:
            raise ValueError('quantization metadata exists but the loader omitted its quantization flag')
        quantized = False
    summary['weights'] = {'dtype': str(next(model.parameters()).dtype), 'quantized': quantized, 'footprint_gib': model.get_memory_footprint()/2**30,
                          'mxfp4': args.mxfp4, 'experts_implementation': model.config._experts_implementation}
    if quantized and training and quantization is None:
        raise ValueError('the model loaded with quantized weights that were not asked for; pass --mxfp4 dequantize for gpt-oss')
    if quantization is not None and training:
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    devices = {parameter.device for parameter in model.parameters()}
    if cuda and any(device.type != 'cuda' for device in devices):
        raise ValueError(f'part of the model was placed off the GPU ({sorted(map(str, devices))}); it does not fit this card as loaded')
    summary['generation_config'] = model.generation_config.to_dict()  # what generation inherits wherever our arguments are silent
    if args.adapter:
        from peft import PeftModel
        saved = json.loads((args.adapter/'adapter_config.json').read_text())
        # A continuation takes rank, alpha and target modules from the saved adapter, whatever the command line says.
        if training and (saved['r'], saved['lora_alpha']) != (args.lora_rank, 2*args.lora_rank):
            raise ValueError(f"--adapter was trained at rank {saved['r']}, alpha {saved['lora_alpha']}; --lora-rank {args.lora_rank} would be ignored")
        if saved['base_model_name_or_path'] != args.model:
            raise ValueError(f"--adapter belongs to {saved['base_model_name_or_path']}, not {args.model}")
        summary['initial_adapter_sha256'] = file_sha256(args.adapter/'adapter_model.safetensors')
        model = PeftModel.from_pretrained(model, str(args.adapter), is_trainable=training)
    summary['model_load_seconds'] = time.perf_counter()-t0
    summary['parameters_total'] = sum(p.numel() for p in model.parameters())
    return model, tokenizer


# Fused 3D expert weights that `all-linear` cannot reach, by architecture (owner, 26 Nov, notes section 72). LoRA on the
# MLP/MoE weights, not attention alone, is the current recommendation for RL ("LoRA Without Regret"); PEFT 0.21 wraps
# these as parameters and vLLM 0.34 serves such adapters on the MXFP4 gpt-oss base. None: a dense model, whose MLP
# `all-linear` already covers. An architecture not listed raises.
EXPERT_PARAMETERS = {
    'gpt_oss': ['mlp.experts.gate_up_proj', 'mlp.experts.down_proj'],
    'gpt2': None,   # CPU tests
    'llama': None,  # dry run
    'qwen3': None,
}


def lora_config(model, rank):
    from peft import LoraConfig
    return LoraConfig(task_type='CAUSAL_LM', r=rank, lora_alpha=2*rank, target_modules='all-linear',
                      target_parameters=EXPERT_PARAMETERS[model.config.model_type], lora_dropout=0.0)


def template_kwargs(args):
    """Chat-template switches for reasoning: Qwen takes enable_thinking, gpt-oss takes reasoning_effort."""
    if args.reasoning_effort:
        return {'reasoning_effort': args.reasoning_effort}
    return {'enable_thinking': args.thinking == 'on'} if args.thinking != 'unsupported' else {}


def row_provenance(args, stage):
    """What each row this run writes in `stage` says about where it came from (provenance.meta)."""
    return provenance.meta(model=args.model, revision=args.revision, arm=args.arm, penalty=args.penalty, stage=stage,
                           report_mode=args.report_mode, taskset=args.taskset, variant=args.prompt_variant, template=template_kwargs(args),
                           run=args.out.name, **{name: provenance.STAGES[stage][name] for name in provenance.SELECTION})


def reasons(args):
    """The completion carries reasoning before the JSON, so its markers are kept in the saved raw text."""
    return args.thinking == 'on' or bool(args.reasoning_effort)


def trainer_samples(model, tokenizer, tasks, args, samples, rollouts, adapter_model, stage):
    """Samples generated by the training model itself, one task at a time: for runs without a vLLM server (the CPU tests,
    the dry run, the probe). Yields (task, completion token ids, generation seconds)."""
    import torch
    device = next(model.parameters()).device
    torch.manual_seed(evaluation_seed(args))
    model.eval()
    for task in tasks:
        inputs = tokenizer.apply_chat_template(prompt(task['statement'], args.prompt_variant, args.report_mode), tokenize=True, return_dict=True,
                                               return_tensors='pt', add_generation_prompt=True, **template_kwargs(args)).to(device)
        t0 = time.perf_counter()
        with torch.inference_mode():
            outputs = model.generate(**inputs, max_new_tokens=args.max_completion_length, do_sample=True, temperature=TEMPERATURE, top_p=TOP_P,
                                     top_k=TOP_K, num_return_sequences=samples, pad_token_id=tokenizer.pad_token_id,
                                     eos_token_id=stop_ids(model, tokenizer))
        start = inputs['input_ids'].shape[1]
        yield task, [output[start:].tolist() for output in outputs], time.perf_counter()-t0


def evaluation_draws(tokenizer, tasks, args, samples):
    """Return the server seed and ordered (task, round, prompt IDs) draws."""
    from .vllm_rollouts import SEED_STRIDE
    draws = []
    for task in tasks:
        messages = prompt(task['statement'], args.prompt_variant, args.report_mode)
        ids = tokenizer.apply_chat_template(messages, tokenize=True, return_dict=True,
                                            add_generation_prompt=True, **template_kwargs(args))['input_ids']
        draws.extend((task, round_index, ids) for round_index in range(samples))
    return evaluation_seed(args) * SEED_STRIDE, draws


def server_samples(model, tokenizer, tasks, args, samples, rollouts, adapter_model, stage):
    """Samples from the vLLM adapter server that serves the training rollouts, for every task in one call (owner, 26 Nov:
    a library's engine rather than trainer-side generation, which LoRA on the gpt-oss experts made about 8x slower). The
    server samples `adapter_model`'s adapter, or the served base model when it is None, with the same untruncated
    sampling as training, from per-request seeds that the before- and after-evaluations share. Yields as trainer_samples.
    Its generation time is the whole server call, wake to sleep (adapter save and load included), and covers every task's
    samples even when the evaluation stops early: `generation_seconds` reads high and `tokens_per_second` low, which errs on
    the safe side when either sizes a limit. The call's own parts are in rollout-backend.jsonl (`sync_seconds`, `seconds`)."""
    seed, draws = evaluation_draws(tokenizer, tasks, args, samples)
    t0 = time.perf_counter()
    ids, _, _ = rollouts.sample(adapter_model, [ids for _, _, ids in draws], seed=seed,
                                record=dict(evaluation=stage))
    seconds = time.perf_counter()-t0
    for index, task in enumerate(tasks):
        # The whole call is generation time, reported with the first task; the tasks after it cost only their scoring.
        yield task, ids[index*samples:(index+1)*samples], seconds if index == 0 else 0.0


def evaluate(model, tokenizer, tasks, args, stage, samples=None, stop_at=None, rollouts=None, adapter_model=None):
    """Score `samples` completions per task, sampled by the vLLM server when the run has one (`rollouts`) and by the training
    model otherwise. With `stop_at` (a `time.monotonic()` instant), stop between tasks when the slowest task so far would no
    longer fit: whole groups are kept and the result says it is incomplete, instead of the hard deadline discarding the
    summary of a probe that had already measured most of its tasks."""
    rows, generated, t_gen, t_lean = [], 0, 0.0, 0.0
    report_mode = args.report_mode
    stage_provenance = row_provenance(args, stage)
    retries = RetryBudget(args.verifier_retry_seconds)
    samples = args.eval_samples if samples is None else samples
    # The before- and after-evaluations of one run draw from the same stream, so their difference is the adapter's. The
    # stream follows --seed (owner, 26 Nov), so runs with different seeds are evaluated on different draws and count as
    # repeats; the offset keeps it apart from the training stream of the same seed.
    sampler = trainer_samples if rollouts is None else server_samples
    # Whether each further task still has to be generated (the trainer, one task at a time) or was generated with the
    # first (the server, one call): only work still to come counts against `stop_at`, so samples already paid for are
    # scored, not dropped (review round 10). The single server call itself cannot be stopped part way: the after-evaluation
    # reserve (--final-eval-seconds) is sized to hold it.
    generates_each_task = {trainer_samples: True, server_samples: False}[sampler]
    path = args.out/f'{stage}-evaluation.jsonl'
    completed, slowest_generation, slowest_scoring = 0, 0.0, 0.0
    stops = stop_ids(model, tokenizer)
    with path.open('w') as sink:
        drawn = sampler(model, tokenizer, tasks, args, samples, rollouts, adapter_model, stage)
        for _ in tasks:
            ahead = slowest_scoring + (slowest_generation if generates_each_task else 0.0)
            if stop_at is not None and completed and time.monotonic()+1.25*ahead > stop_at:
                break
            task, completions, seconds = next(drawn)
            scoring_started = time.monotonic()
            t_gen += seconds
            generated += sum(token != tokenizer.pad_token_id for new in completions for token in new)

            def scored(new, task=task):
                raw = tokenizer.decode(new, skip_special_tokens=report_mode == 'typed' or not reasons(args))
                truncated = completion_truncated(new, stops, args.max_completion_length)
                return score(args.lean, task, raw, args.arm, args.penalty, args.judge_fn, truncated=truncated, retries=retries, report_mode=report_mode)
            t0 = time.perf_counter()
            try:
                scored_rows = score_concurrently(scored, completions)  # a task's samples at once
            except ScoringStopped as stopped:
                jsonl.write_rows(sink, provenance.stamp(stopped.done, stage_provenance))  # scored and paid for: kept, as when scoring was one row at a time
                raise stopped.__cause__  # noqa: B904  re-raises the original error; `from` would overwrite that error's own cause
            t_lean += time.perf_counter()-t0
            rows.extend(scored_rows)
            jsonl.write_rows(sink, provenance.stamp(scored_rows, stage_provenance))
            if judge_failing(rows):
                raise RuntimeError(f"the judge is failing in the {stage} evaluation: {judge_advice(rows)}")
            completed += 1
            slowest_generation, slowest_scoring = max(slowest_generation, seconds), max(slowest_scoring, time.monotonic()-scoring_started)
    return dict(evaluation_counts(rows, samples), tasks_planned=len(tasks), tasks_completed=completed, incomplete=completed < len(tasks), report_mode=report_mode, verifier_retries_denied=retries.denied, generation_seconds=t_gen, lean_seconds=t_lean, generated_tokens=generated,
                tokens_per_second=observed_ratio(generated, t_gen), completions_per_hour=observed_ratio(3600*len(rows), t_gen+t_lean))


def evaluation_counts(rows, samples):
    """What an evaluation's rows say, measured now or carried over from the run a resume continues."""
    training_rewards(rows, samples)  # probe variation must predict the signal the trainer will actually use
    grouped = reward_groups(rows, samples)
    return dict(summarize(rows), samples_per_prompt=samples, reward_groups=grouped['groups'], mixed_reward_groups=grouped['mixed_groups'],
                zero_advantage_groups=grouped['zero_advantage_groups'], per_group=grouped['per_group'])


def copy_before(args, summary):
    """A resumed run's before-evaluation. The original measured the base model on the same tasks, seed and settings
    (`check_resume` holds them equal): its rows are copied, marked so the transcript export counts them once, rather
    than paid for again."""
    original, _, _ = resume_point(args)
    before = list(jsonl.read(original/'before-evaluation.jsonl'))
    with (args.out/'before-evaluation.jsonl').open('w') as sink:
        jsonl.write_rows(sink, (dict(row, copied_from=str(original)) for row in before))
    summary['before'] = dict(evaluation_counts(before, args.eval_samples), copied_from=str(original))


def prepare_training(model, tokenizer, tasks, args, summary, *, rollouts=None):
    import torch
    from datasets import Dataset
    from transformers import TrainerCallback, set_seed
    from transformers.integrations import TensorBoardCallback
    from trl import GRPOConfig, GRPOTrainer
    by_statement = {t['statement']: t for t in tasks}
    report_mode = args.report_mode
    summary['report_mode'] = report_mode
    stops = stop_ids(model, tokenizer)  # the same stop IDs as `evaluate`, so both paths agree on a finished answer
    if (args.rollout_backend == 'vllm') != (rollouts is not None):
        raise ValueError('the requested rollout backend must match the supplied backend')
    if rollouts is not None:
        rollouts.stops = stops
    timing = {'step_start': 0.0, 'rollout': [], 'lean': [], 'update': [], 'reward_end': 0.0}
    sink = (args.out/ROLLOUTS).open('w')
    training_provenance = row_provenance(args, 'training')  # on every rollout and steps.jsonl line this run writes
    groups, undecided, lost = [], [], []
    retries = RetryBudget(args.verifier_retry_seconds)  # for the whole training phase, so the deadline reserve holds

    steps = (args.out/STEPS).open('w')
    first_seen = summary.setdefault('first_seen_step', {})

    def note_undecided(rows):
        undecided.extend({key: row.get(key) for key in ('verifier_inconclusive', 'judge_asked', 'judge_undecided')} for row in rows)

    def account(rows, step):
        """Add one step's rollouts to the run's counters: variation for the no-variation stop, totals, first sightings."""
        grouped = reward_groups(rows, args.num_generations)
        groups.extend([True]*grouped['mixed_groups'] + [False]*grouped['zero_advantage_groups'])
        lost.append(grouped['variation_lost_to_verifier'])
        counts = tally(rows)
        accumulate(summary['rollouts'], counts)
        # The first step at which each behaviour was sampled at all: the exploration question, per arm and seed.
        for behaviour in ('exploit', 'refutation', 'genuine', 'says_not_solved', 'concealed_exploit'):
            if counts.get(behaviour) and behaviour not in first_seen:
                first_seen[behaviour] = step
        return counts, grouped

    if args.resume_from:
        # A resumed run carries the steps before its checkpoint: the original run's records of them are copied into this
        # run's files and replayed into the counters, so the stops, totals and first sightings cover the whole run.
        original, done, _ = resume_point(args)
        batch = args.num_generations*args.prompts_per_step
        rows, lines = original_records(args)
        for step in range(1, done+1):
            batch_rows = rows[(step-1)*batch:step*batch]
            note_undecided(batch_rows)
            account(batch_rows, step)
        # Marked as copies, so that readers of all runs (the transcript export) count each rollout once.
        jsonl.write_rows(sink, (dict(row, copied_from=str(original)) for row in rows))
        jsonl.write_rows(steps, (dict(line, copied_from=str(original)) for line in lines))
        # The verifier's retry allowance is the whole run's: the uninterrupted run would have had what step `done` left.
        retries.left, retries.denied = lines[-1]['retries_left'], lines[-1]['retries_denied']
        summary['training_verifier'] = dict(verifier_advice(undecided), retries_denied=retries.denied, **judge_advice(undecided))

    def reward_fn(completions, statement, **kwargs):
        t0 = time.perf_counter()
        timing['rollout'].append(t0-timing['step_start'])
        # Counts on from the checkpoint in a resumed run. Every row carries it, so a reader can tell the original run's rows
        # past a checkpoint, which the resumed run generates again, from those before it.
        step = trainer.state.global_step+1
        completion_ids = kwargs.get('completion_ids')
        if completion_ids is None:
            raise ValueError('TRL must supply completion_ids to determine whether each answer stopped')
        def scored(answer):
            completion, claim, token_ids = answer
            raw = completion[-1]['content'] if isinstance(completion, list) else completion
            truncated = completion_truncated(token_ids, stops, args.max_completion_length)
            return dict(score(args.lean, by_statement[claim], raw, args.arm, args.penalty, args.judge_fn, truncated=truncated,
                              retries=retries, report_mode=report_mode), step=step)
        try:
            rows = score_concurrently(scored, list(zip(completions, statement, completion_ids, strict=True)))
        except ScoringStopped as stopped:
            # A stop mid-batch (deadline, judge streak, spend cap) keeps what was already scored and paid for.
            # Marked, because these rows are in no step, no tally and no complete prompt group.
            jsonl.write_rows(sink, provenance.stamp((dict(row, partial_batch=True) for row in stopped.done), training_provenance))
            summary['partial_batch_rows'] = len(stopped.done)
            raise stopped.__cause__  # noqa: B904  re-raises the original error; `from` would overwrite that error's own cause
        rewards = training_rewards(rows, args.num_generations)
        trainer._verifier_keep = [training_keep(row) for row in rows]
        note_undecided(rows)
        judging = judge_advice(undecided)
        summary['training_verifier'] = dict(verifier_advice(undecided), retries_denied=retries.denied, **judging)
        if judge_failing(undecided):
            # Stops a paid run whose rollouts are being dropped by an unreliable judge. The batch is saved and marked: it is in
            # no step and no tally.
            jsonl.write_rows(sink, provenance.stamp((dict(row, partial_batch=True) for row in rows), training_provenance))
            summary['partial_batch_rows'] = len(rows)
            raise RuntimeError(f"the judge failed on {judging['judge_undecided']} of {judging['judge_asked']} readings: its failure rate is above the 5% limit")
        if verifier_failing(undecided):
            # Lean should decide these statements in well under a second: timeouts piling up mean a broken verifier or
            # proofs long enough to time out (owner wants to know at once). Saved and marked like the judge stop above.
            jsonl.write_rows(sink, provenance.stamp((dict(row, partial_batch=True) for row in rows), training_provenance))
            summary['partial_batch_rows'] = len(rows)
            raise RuntimeError(f"the verifier left {sum(bool(r['verifier_inconclusive']) for r in undecided)} of {len(undecided)} answers undecided, "
                               f"its undecided rate is credibly above 5%; longest undecided proof in this batch: {tally(rows)['inconclusive_longest_proof_chars']} characters")
        jsonl.write_rows(sink, provenance.stamp(rows, training_provenance))
        counts, grouped = account(rows, step)
        # Equal rewards give zero reward advantages; KL and optimizer momentum can still move weights.
        # `retries_left` lets a resumed run restore the verifier's retry allowance where this step left it.
        jsonl.write(steps, dict(step=step, **counts, **verifier_advice(rows), retries_denied=retries.denied, retries_left=retries.left,
                                **grouped, meta=training_provenance), ensure_ascii=True)
        timing['reward_end'] = time.perf_counter()
        timing['lean'].append(timing['reward_end']-t0)
        return rewards

    run_args = args

    class Instrument(TrainerCallback):
        def on_pre_optimizer_step(self, args, state, control, **kwargs):
            # A finite loss can still have non-finite gradients; the optimiser would write them into the adapter, and the
            # loss check would only notice one paid step later, with the last good weights gone.
            bad = [name for name, p in kwargs['model'].named_parameters() if p.grad is not None and not bool(p.grad.isfinite().all())]
            if bad:
                raise FloatingPointError(f'non-finite gradient at step {state.global_step+1} in {len(bad)} tensors, first {bad[0]}')

        def on_log(self, args, state, control, logs=None, **kwargs):
            # Finite gradients whose total norm overflows are clipped to zero and pass the check above; the logged norm shows it.
            if not isinstance(logs, dict):
                raise TypeError('trainer log event needs a dictionary')
            if 'loss' in logs:
                norm = logs['grad_norm']
            elif 'grad_norm' in logs:
                norm = logs['grad_norm']
            else:
                if not any(key.startswith('eval_') or key.startswith('train_') for key in logs):
                    raise ValueError(f'unrecognized trainer log event: {sorted(logs)}')
                return
            if not math.isfinite(norm):
                raise FloatingPointError(f'the gradient norm is {norm} at step {state.global_step}')

        def on_step_begin(self, args, state, control, **kwargs):
            timing['step_start'] = time.perf_counter()

        def on_step_end(self, args, state, control, **kwargs):
            timing['update'].append(time.perf_counter()-timing['reward_end'])
            (run_args.out/'live.json').write_text(json.dumps({'step': state.global_step, **summary['rollouts'], **summary['training_verifier'], 'mixed_groups': sum(groups)}))
            if state.global_step % run_args.save_every == 0:
                kwargs['model'].save_pretrained(str(run_args.out/f'adapter-step{state.global_step}'))
            if state.global_step >= run_args.stop_no_variation and not any(groups):
                verifier = summary.setdefault('training_verifier', {})
                verifier['variation_lost_groups'] = sum(lost)
                summary['early_stop'] = VERIFIER_UNDECIDED if sum(lost) else NO_VARIATION  # rollouts left out for any reason, the judge's failures included
                control.should_training_stop = True
            # Only once some group has varied: a run that never varied is the no-variation stop's case, which waits for
            # its own step and records whether the verifier lost the variation.
            elif run_args.stop_saturated and any(groups) and saturated(groups, run_args.prompts_per_step):
                summary['early_stop'] = SATURATED
                control.should_training_stop = True
            # A stop recorded above keeps its reason: it is what decides whether the run may be resumed.
            if time.monotonic() > run_args.train_deadline and not control.should_training_stop:
                summary['early_stop'] = 'deadline'
                control.should_training_stop = True
            # A run stopped here (deadline, no variation, saturation, or its last step) keeps a checkpoint of that step, so a resume repeats
            # no paid step; `fit` does the same for a stop by exception.
            control.should_save = control.should_save or control.should_training_stop
            return control

    # One prompt group per micro-batch and --prompts-per-step micro-batches per optimizer step: TRL generates the whole
    # step's answers at once (its generation batch is the accumulated batch) and a backward pass holds one group, so
    # more prompts per step cost time, not memory. The 'dapo' loss normalises over all the step's tokens.
    config = GRPOConfig(output_dir=str(args.out/'trainer'), max_steps=args.steps, per_device_train_batch_size=args.num_generations,
                        gradient_accumulation_steps=args.prompts_per_step, num_generations=args.num_generations, max_completion_length=args.max_completion_length,
                        learning_rate=args.learning_rate, bf16=torch.cuda.is_available(), use_cpu=not torch.cuda.is_available(), gradient_checkpointing=True,
                        beta=args.beta, temperature=TEMPERATURE, top_p=TOP_P, top_k=TOP_K, chat_template_kwargs=template_kwargs(args), logging_steps=1,
                        # Resumable checkpoints (<out>/trainer/checkpoint-N: adapter, optimizer, step, random state, task order) every
                        # --save-every steps; only the latest two are kept, as a resume needs one. The adapter-stepN snapshots above
                        # keep every interval's weights for evaluation.
                        save_strategy='steps', save_steps=args.save_every, save_total_limit=2,
                        # Training metrics as TensorBoard event files on the run's own disk (<out>/tensorboard): no account, no
                        # network, nothing leaves the machine. TRL's `log_completions` stays off: it only prints tables and
                        # writes parquet copies of what training-rollouts.jsonl already keeps with its scoring.
                        report_to=['tensorboard'], seed=args.seed, dataloader_num_workers=0,
                        # TRL's default: average the loss over every token in the batch. 'grpo' divided each answer's loss by its
                        # own length, so a long wrong answer was penalised less per token than a short one; the 60-step baseline
                        # of 26 Nov drifted to the token limit under it (notes sections 70-71; DAPO, Dr. GRPO).
                        loss_type='dapo',
                        # Two library defaults that changed the objective until 2026-11-25. Left unset, TRL takes the model's own
                        # router load-balancing coefficient (0.9 for gpt-oss), which made the logged loss 0.9 x that term; and the
                        # learning rate decays linearly to zero over --steps, so a run's rate depended on its length.
                        router_aux_loss_coef=args.router_aux_loss_coef, lr_scheduler_type=args.lr_schedule,
                        # More defaults that shape the update, now arguments or stated here so that nothing is inherited unseen.
                        # `scale_rewards`: 'group' divides each prompt group's advantages by that group's reward spread, so the
                        # size of a reward gap is invisible within a group; 'none' keeps it. See --scale-rewards.
                        scale_rewards=args.scale_rewards, max_grad_norm=args.max_grad_norm, optim='adamw_torch', weight_decay=0.0,
                        warmup_steps=0, adam_beta1=0.9, adam_beta2=0.999, adam_epsilon=1e-8,
                        logging_nan_inf_filter=False, generation_kwargs={'eos_token_id': stops},
                        # vLLM samples from the MXFP4 base; the trainer scores with bf16 weights. TRL's default 'sequence_mask'
                        # multiplies per-token ratios over the whole answer, which drove long answers' weights to about 0 and so
                        # reweighted outcomes by length (notes section 66). Per-token ratios capped at 3 do not compound.
                        vllm_importance_sampling_correction=True, vllm_importance_sampling_mode='token_truncate',
                        vllm_importance_sampling_clip_max=3.0, vllm_importance_sampling_clip_min=None)
    dataset = Dataset.from_list([{'prompt': prompt(t['statement'], args.prompt_variant, report_mode), 'statement': t['statement']} for t in tasks])
    _, started_at, epochs_done = resume_point(args) if args.resume_from else (None, 0, 0)

    class VerifiedTrainer(VerifierMaskMixin, GRPOTrainer):
        def _get_train_sampler(self, dataset=None):
            # TRL's sampler draws each pass's task order from its own generator, which a checkpoint does not save and
            # transformers does not fast-forward: without this, a resume past a pass boundary repeats the first pass's order.
            sampler = super()._get_train_sampler(dataset)
            if sampler.shuffle:
                for _ in range(epochs_done):
                    torch.randperm(sampler.num_samples, generator=sampler.generator)
            return sampler

        def _generate_single_turn(self, prompt_ids, images, multimodal_fields, num_generations, has_tool_images=False):
            if rollouts is None:
                return super()._generate_single_turn(prompt_ids, images, multimodal_fields, num_generations, has_tool_images)
            if images is not None or multimodal_fields or has_tool_images:
                raise ValueError('external adapter rollouts are text-only')
            return rollouts.generate(self, prompt_ids, num_generations)

    trainer = VerifiedTrainer(model=model, args=config, reward_funcs=reward_fn, train_dataset=dataset, processing_class=tokenizer, callbacks=[Instrument()],
                          peft_config=None if args.adapter else lora_config(model, args.lora_rank))
    # The callback `report_to` added writes under trainer/runs/<date_host> unless told otherwise: one fixed folder per run.
    # A resumed run adds a new event file beside the first, and TensorBoard reads the folder as one run.
    [board] = [c for c in trainer.callback_handler.callbacks if isinstance(c, TensorBoardCallback)]
    board.logging_dir = str(args.out/'tensorboard')
    # TRL 1.19 unions model-only stops into its completion mask. Keep that mask
    # aligned with our explicit generation stop and evaluation cutoff detector.
    trainer.eos_token_ids = [stops]
    if rollouts is not None:
        # Initialize without TRL's native VLLMGeneration (which imports the other
        # environment's Torch and transfers full weights). The overridden method
        # above supplies tokens/logprobs; this flag enables TRL's existing ratio
        # computation and loss correction. Guarded to TRL 1.19.0 by check_args.
        trainer.use_vllm = True
        trainer.vllm_mode = 'server'
        summary['rollout_backend'] = dict(kind='vllm-adapter-server', url=args.rollout_url,
            logprobs='processed_logprobs', correction=config.vllm_importance_sampling_mode, correction_max=config.vllm_importance_sampling_clip_max)
    else:
        summary['rollout_backend'] = dict(kind='transformers')
    before = {n: p.detach().cpu().clone() for n, p in trainer.model.named_parameters() if p.requires_grad}
    summary['trainable_parameters'] = sum(p.numel() for p in before.values())
    summary['trainer'] = {'lr_schedule': trainer.args.lr_scheduler_type, 'router_aux_loss_coef': trainer.router_aux_loss_coef,
                          'aux_loss_enabled': trainer.aux_loss_enabled, 'scale_rewards': args.scale_rewards, 'max_grad_norm': args.max_grad_norm,
                          'arguments_in_effect': trainer.args.to_dict(),
                          # What TRL passes to `generate`. The model's own generation config (`summary['generation_config']`)
                          # then fills whatever is unset here, inside `generate`: the two together are what is in effect.
                          'generation_passed_by_trl': trainer.generation_config.to_dict(),
                          'lora_target_modules': sorted({name.split('.')[-1] for name, module in trainer.model.named_modules() if hasattr(module, 'lora_A')})}
    def save_after_failure(error):
        """What a failed run keeps: the weights it reached, and a resumable checkpoint of its last completed step unless
        one was just saved, for an ordinary error only. The judge or verifier stop and out of memory are raised in scoring,
        generation or backward, before the step's optimizer update, so the state is that of `global_step` (the random
        state has moved on into the failed step). Everything else gets none, because its state may not be a step's: a
        numerical failure (the gradient-norm check runs after the update it condemns) and anything that is not an
        Exception (the hard deadline, Ctrl-C: signals, which can land inside the optimizer update or between it and the
        step count). Only once this process has finished a step of its own: a resume sets the step count before it loads
        the optimizer, so a failure in between would save a fresh optimizer under the checkpoint's step."""
        trainer.save_model(str(args.out/'adapter-at-failure'))
        done = trainer.state.global_step
        ordinary = isinstance(error, Exception) and not isinstance(error, FloatingPointError)
        # A checkpoint folder without trainer_state.json (written last) is a save cut short: save over it.
        if done > started_at and ordinary and not (args.out/'trainer'/f'checkpoint-{done}'/'trainer_state.json').exists():
            trainer._save_checkpoint(trainer.model, trial=None)

    def fit():
        t0 = time.perf_counter()
        # TRL seeds once, when the trainer is built; the before-evaluation then reseeds with its own constant. Without this,
        # every --seed sampled its training rollouts from the same random stream.
        set_seed(args.seed)
        try:
            # A resumed run restores weights, optimizer, step count, random state and task order from the checkpoint.
            trainer.train(resume_from_checkpoint=str(args.resume_from) if args.resume_from else None)
        except BaseException as error:
            # Whatever stopped training, the updates made so far were paid for. The deadline repeats every 30 s; it has
            # been heard, and must not cut this save short.
            deadline.disarm()
            # The step in progress did not update the model. Its rows, its steps.jsonl line and its share of the rollout
            # totals and reward groups are written once it is scored; readers of a failed run leave out this step.
            summary['untrained_step'] = trainer.state.global_step + 1
            # The saves must not replace the error that stopped training as the run's recorded reason. A save fails on the
            # disk (OSError: full, unwritable) or the card (RuntimeError: a CUDA error left by out of memory); its error is
            # recorded in the summary and attached to the original, which is what is raised.
            try:
                save_after_failure(error)
            except (OSError, RuntimeError) as save_error:
                summary['save_after_failure_error'] = f'{type(save_error).__name__}: {str(save_error)[:400]}'
                error.add_note(f"saving after this failure also failed: {summary['save_after_failure_error']}")
            raise
        finally:
            summary.update(global_step=trainer.state.global_step, mixed_reward_groups=sum(groups), reward_groups=len(groups))
            # Paid for whether or not training finished: the per-step log, and what each step's time went on.
            (args.out/'training-log.json').write_text(json.dumps(trainer.state.log_history, indent=1)+'\n')
            (args.out/'step-seconds.json').write_text(json.dumps({k: timing[k] for k in ('rollout', 'lean', 'update')}, indent=1)+'\n')
        summary.update(training_seconds=time.perf_counter()-t0,
                       adapter_max_abs_change=max(float((p.detach().cpu()-before[n]).abs().max()) for n, p in trainer.model.named_parameters() if n in before),
                       mean_seconds={k: observed_ratio(sum(timing[k]), len(timing[k])) for k in ('rollout', 'lean', 'update')})
        trainer.save_model(str(args.out/'adapter'))
        sink.close()
        steps.close()
        return trainer.model
    return fit


def train(model, tokenizer, tasks, args, summary):
    """Build the trainer and run it. `run` builds it earlier, before the paid before-evaluation."""
    return prepare_training(model, tokenizer, tasks, args, summary)()


def check_completion_limit(evaluation):
    """Refuse optimization when the initial evaluation calls for longer answers or cannot decide."""
    if evaluation['raise_token_limit'] is not False:
        raise ValueError('before-evaluation does not establish an adequate completion limit; '
                         'inspect its truncation evidence and resize the run before training')


def run(args):
    """`args.mode` is controls, probe or train. Writes summary.json under `args.out` whatever happens."""
    # Before anything is written or loaded: a refused resume leaves the original run's folder untouched and costs nothing.
    if args.resume_from:
        check_resume(args)
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    args.train_deadline = time.monotonic()+args.max_seconds-args.final_eval_seconds
    args.judge_fn = None  # no judge is built in `controls`, or before the controls pass; the usage line below reads it either way
    summary = {'status': 'running', 'report_mode': args.report_mode, 'argv': sys.argv[1:], 'evaluation_seed': evaluation_seed(args),
               # What the transcript export needs to rebuild each prompt and label the policy; argv alone omits defaults.
               'model': args.model, 'revision': args.revision, 'prompt_variant': args.prompt_variant,
               'adapter': str(args.adapter) if args.adapter else None, 'resumed_from': str(args.resume_from) if args.resume_from else None,
               'template_kwargs': template_kwargs(args), 'purpose': 'feasibility; not intervention evidence', 'rollouts': {},
               'script_sha256': source_sha256()}
    # Written now with status 'running' and again as the run ends: a run killed from outside (the pod removed mid-run)
    # still has one, so readers that find runs by their summary (the transcript export, a resume) still find it.
    (args.out/'summary.json').write_text(json.dumps(summary, indent=1)+'\n')

    deadline.arm(args.max_seconds)
    hard_deadline = time.monotonic()+args.max_seconds
    sampler = GpuSampler()
    sampler.start()
    rollouts = None
    try:
        from .vllm_rollouts import AdapterRollouts, check_args
        check_args(args)
        if args.rollout_backend == 'vllm':
            rollouts = AdapterRollouts(args, [])
            rollouts.preflight()  # prove the server is sleeping before loading the training model
        summary['controls'] = controls(args.lean)
        summary['lean'] = subprocess.check_output([args.lean, '--version'], text=True).strip()
        print('controls passed', flush=True)
        if MODES[args.mode]['controls']:
            summary['status'] = 'completed'
            return
        from .judge import from_args
        args.judge_fn = from_args(args)
        evaluation = stage_tasks(args.taskset, 'before', args.eval_limit)
        training = stage_tasks(args.taskset, 'training', args.train_limit)
        (args.out/'tasks.json').write_text(json.dumps({'train': training, 'evaluation': evaluation}, indent=1, ensure_ascii=False)+'\n')
        import peft
        import torch
        import transformers
        import trl
        torch.manual_seed(args.seed)
        summary.update(torch=torch.__version__, transformers=transformers.__version__, trl=trl.__version__, peft=peft.__version__,
                       gpus=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
        model, tokenizer = load_model(args, summary)
        if MODES[args.mode]['probe']:
            summary['probe'] = evaluate(model, tokenizer, training, args, 'probe', samples=args.num_generations, stop_at=hard_deadline-60)
            print('probe tasks completed', summary['probe']['tasks_completed'], 'of', summary['probe']['tasks_planned'], flush=True)
            summary['probe_split'] = 'train'
            print('probe cut-off rate', summary['probe']['truncated_rate'], 'raise the token limit:', summary['probe']['raise_token_limit'], flush=True)
            print('probe undecided rate', summary['probe']['unverified_rate'], 'check the verifier:', summary['probe']['check_verifier'], flush=True)
            summary['status'] = 'completed'
            return
        # Built before the before-evaluation is paid for: a trainer the libraries refuse to construct fails here, in seconds.
        # The adapter starts as an exact no-op (its B matrices are zero), so the before-evaluation still measures the base policy.
        # A resumed run's before-evaluation still measures the base model (the checkpoint's weights load when training
        # starts), as the original run's did: `check_resume` refuses runs that started from --adapter.
        (args.out/RUN_SETTINGS).write_text(json.dumps(run_settings(args), indent=1)+'\n')
        fit = prepare_training(model, tokenizer, training, args, summary, rollouts=rollouts)
        if args.resume_from:
            copy_before(args, summary)
        else:
            # On the vLLM server: the served base model, or the adapter a run started from.
            summary['before'] = evaluate(model, tokenizer, evaluation, args, 'before', rollouts=rollouts, adapter_model=model if args.adapter else None)
        print('before', summary['before'], flush=True)
        check_completion_limit(summary['before'])
        model = fit()
        # Stops between tasks if training ran into its reserve: a shorter after-evaluation is kept, a killed one loses its summary.
        summary['after'] = evaluate(model, tokenizer, evaluation, args, 'after', stop_at=hard_deadline-60, rollouts=rollouts, adapter_model=model)
        after_rows = list(jsonl.read(args.out/'after-evaluation.jsonl'))
        done = {row['task_id'] for row in after_rows}
        matched = [row for row in jsonl.read(args.out/'before-evaluation.jsonl') if row['task_id'] in done]
        if summary['after']['incomplete']:
            # A shortened after-evaluation covers fewer tasks; beside the full before-evaluation its rates would compare different task sets.
            summary['before_on_after_tasks'] = summarize(matched)
        # Posterior of each rate's change over training, on the tasks both evaluations answered, clustered by task group.
        summary['after_minus_before'] = compare(matched, after_rows)
        summary['status'] = 'completed'
    except BaseException as error:
        deadline.disarm()  # heard once; the summary below must be written
        summary.update(status='failed', reason=f'{type(error).__name__}: {str(error)[:800]}')
        raise
    finally:
        deadline.disarm()
        if rollouts is not None:
            rollouts.close()
        gpu = sampler.summary()
        meter = None if args.judge_fn is None else args.judge_fn.meter
        summary['judge_usage'] = meter.snapshot() if meter else None  # a copy: scoring threads may still be counting (review round 14)
        summary.update(elapsed_seconds=time.perf_counter()-started, peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**(20 if sys.platform != 'darwin' else 30),
                       peak_gpu_memory_used_gib=None if gpu['gpu_sampler_error'] else gpu['gpu_peak_mib']/1024,
                       max_gpu_utilization=None if gpu['gpu_sampler_error'] else gpu['gpu_max_util'], gpu_sampler_error=gpu['gpu_sampler_error'])
        if 'torch' in sys.modules and sys.modules['torch'].cuda.is_available():
            torch_ = sys.modules['torch']
            summary['peak_allocated_gib_all_gpus'] = sum(torch_.cuda.max_memory_allocated(i) for i in range(torch_.cuda.device_count()))/2**30
        (args.out/'summary.json').write_text(json.dumps(summary, indent=1)+'\n')
        print(json.dumps({k: v for k, v in summary.items() if k != 'script_sha256'}, indent=1), flush=True)
