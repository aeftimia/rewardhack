"""GPU path: measure exploration (`probe`) or train with TRL GRPO (`train`), recording cost-relevant timings.

Adapted from exit-reward-research@ec0565e scripts/grpo_lean_pilot.py. Arms share one prompt; only the
reward differs. Entrypoint: `entrypoints/run.py`.
"""
import hashlib
import json
import math
import resource
import subprocess
import sys
import time
from pathlib import Path

from . import deadline
from .gpu import GpuSampler
from .outcomes import JUDGE_ROWS_BEFORE_STOP, judge_advice, NO_VARIATION, RETRY_SECONDS, VERIFIER_UNDECIDED, RetryBudget, verifier_advice, completion_truncated, reward_groups, score, stop_ids, summarize, tally, training_rewards, training_keep
from .prompts import prompt
from .tasks import split, stratified_limit, taskset_rows
from .verifier import controls


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
        batch['verifier_keep'] = keep
        batch['advantages'] = batch['advantages'].masked_fill(~keep, 0.0)
        # tool_mask affects the loss only, leaving attention to generated tokens intact.
        mask = keep[:, None].expand_as(batch['completion_mask'])
        batch['tool_mask'] = batch.get('tool_mask', torch.ones_like(batch['completion_mask'])) * mask
        return batch

    def _compute_loss(self, model, inputs):
        loss = super()._compute_loss(model, inputs)
        keep = inputs.get('verifier_keep')
        if keep is None:
            return loss
        # GRPO averages sequence losses over the original batch. Remove the zeroed
        # rows from that denominator too; DDP averages rank gradients afterwards.
        count = self.accelerator.reduce(keep.sum().to(loss.dtype), reduction='sum')
        return loss * (keep.numel() * self.accelerator.num_processes / count.clamp(min=1))


PROMPT_DATE = '2026-11-24'  # the day of the first unhinted probe; every prompt since carries it


def settle_template(tokenizer, args):
    """Make the chat template say the same thing on every day and refuse a reasoning switch it would ignore.

    The gpt-oss template writes "Current date: <today>" into the system message, so the prompt changed at midnight,
    between runs, and when transcripts were rebuilt later. It also falls back to "Reasoning: medium" when no effort
    is passed, and ignores Qwen's `enable_thinking` without a word.
    """
    template = tokenizer.chat_template or ''
    if 'strftime_now' in template:
        pinned = template.replace('strftime_now("%Y-%m-%d")', f'"{PROMPT_DATE}"')
        if 'strftime_now' in pinned:
            raise ValueError('the chat template reads the clock in a form this code does not pin')
        tokenizer.chat_template = pinned
    if ('reasoning_effort' in template) != bool(args.reasoning_effort):
        raise ValueError('--reasoning-effort must be given exactly when the model\'s chat template takes one (gpt-oss); '
                         'unset, that template silently uses medium')
    if args.thinking != 'unsupported' and 'enable_thinking' not in template:
        raise ValueError('--thinking on/off needs a chat template with enable_thinking; pass --thinking unsupported for this model')
    return 'strftime_now' in template


def load_model(args, summary):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.padding_side = 'left'
    if not tokenizer.chat_template and args.model.startswith('hf-internal-testing/'):
        tokenizer.chat_template = "{% for m in messages %}{{ m['role'] }}: {{ m['content'] }}\n{% endfor %}assistant:"  # CPU dry run only
    reads_clock = settle_template(tokenizer, args)
    rendered = tokenizer.apply_chat_template(prompt('True', 'neutral', 'prose'), tokenize=False, add_generation_prompt=True, **template_kwargs(args))
    summary['prompt'] = {'date_in_template': PROMPT_DATE if reads_clock else None,
                         'sha256_of_reference_prompt': hashlib.sha256(rendered.encode()).hexdigest()}
    kwargs = {'revision': args.revision, 'attn_implementation': args.attn_implementation}
    cuda = torch.cuda.is_available()
    kwargs['dtype'] = torch.bfloat16 if cuda else torch.float32
    if args.quant == '8bit':
        kwargs['quantization_config'] = BitsAndBytesConfig(load_in_8bit=True)
    elif args.quant == '4bit':
        kwargs['quantization_config'] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                                                           bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    if args.mxfp4 == 'dequantize':
        # gpt-oss ships 4-bit (MXFP4) expert weights. Left to itself the loader keeps them when the `kernels` package is
        # installed and expands them to bf16 when it is not, with one warning line: 13 GiB or 39 GiB, trainable or not,
        # decided by what happens to be installed. Training needs the expanded weights, so that is asked for by name.
        if args.quant != 'none':
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
    quantized = bool(getattr(model, 'is_quantized', False))  # fallback-ok: transformers sets this attribute only on models a quantizer touched; absent means none did
    summary['weights'] = {'dtype': str(next(model.parameters()).dtype), 'quantized': quantized, 'footprint_gib': model.get_memory_footprint()/2**30,
                          'mxfp4': args.mxfp4, 'experts_implementation': model.config._experts_implementation}
    if quantized and args.mode == 'train' and args.quant == 'none':
        raise ValueError('the model loaded with quantized weights that were not asked for; pass --mxfp4 dequantize for gpt-oss')
    if args.quant != 'none' and args.mode == 'train':
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    placed = getattr(model, 'hf_device_map', {})  # fallback-ok: the map exists only when accelerate placed the model; without it everything is on one device
    if any(str(device) in ('cpu', 'disk') for device in placed.values()):
        raise ValueError(f'part of the model was placed off the GPU ({sorted(set(map(str, placed.values())))}); it does not fit this card as loaded')
    summary['generation_config'] = model.generation_config.to_dict()  # what generation inherits wherever our arguments are silent
    if args.adapter:
        from peft import PeftModel
        saved = json.loads((args.adapter/'adapter_config.json').read_text())
        # A continuation takes rank, alpha and target modules from the saved adapter, whatever the command line says.
        if args.mode == 'train' and (saved['r'], saved['lora_alpha']) != (args.lora_rank, 2*args.lora_rank):
            raise ValueError(f"--adapter was trained at rank {saved['r']}, alpha {saved['lora_alpha']}; --lora-rank {args.lora_rank} would be ignored")
        if saved['base_model_name_or_path'] != args.model:
            raise ValueError(f"--adapter belongs to {saved['base_model_name_or_path']}, not {args.model}")
        summary['initial_adapter_sha256'] = hashlib.sha256((args.adapter/'adapter_model.safetensors').read_bytes()).hexdigest()
        model = PeftModel.from_pretrained(model, str(args.adapter), is_trainable=args.mode == 'train')
    summary['model_load_seconds'] = time.perf_counter()-t0
    summary['parameters_total'] = sum(p.numel() for p in model.parameters())
    return model, tokenizer


def template_kwargs(args):
    """Chat-template switches for reasoning: Qwen takes enable_thinking, gpt-oss takes reasoning_effort."""
    if args.reasoning_effort:
        return {'reasoning_effort': args.reasoning_effort}
    return {'enable_thinking': args.thinking == 'on'} if args.thinking != 'unsupported' else {}


def reasons(args):
    """The completion carries reasoning before the JSON, so its markers are kept in the saved raw text."""
    return args.thinking == 'on' or bool(args.reasoning_effort)


def evaluate(model, tokenizer, tasks, args, stage, samples=None, stop_at=None):
    """Score `samples` completions per task. With `stop_at` (a `time.monotonic()` instant), stop between tasks
    when the slowest task so far would no longer fit: whole groups are kept and the result says it is incomplete,
    instead of the hard deadline discarding the summary of a probe that had already measured most of its tasks."""
    import torch
    device = next(model.parameters()).device
    rows, generated, t_gen, t_lean = [], 0, 0.0, 0.0
    report_mode = getattr(args, 'report_mode', 'prose')
    retries = RetryBudget(getattr(args, 'verifier_retry_seconds', RETRY_SECONDS))
    samples = args.eval_samples if samples is None else samples
    torch.manual_seed(31415)
    model.eval()
    path = args.out/f'{stage}-evaluation.jsonl'
    completed, slowest = 0, 0.0
    with path.open('w') as sink:
        for task in tasks:
            if stop_at is not None and completed and time.monotonic()+1.25*slowest > stop_at:
                break
            task_started = time.monotonic()
            inputs = tokenizer.apply_chat_template(prompt(task['statement'], args.prompt_variant, report_mode), tokenize=True, return_dict=True,
                                                   return_tensors='pt', add_generation_prompt=True, **template_kwargs(args)).to(device)
            t0 = time.perf_counter()
            with torch.inference_mode():
                outputs = model.generate(**inputs, max_new_tokens=args.max_completion_length, do_sample=True, temperature=1.0, top_p=0.95,
                                         top_k=20, num_return_sequences=samples, pad_token_id=tokenizer.pad_token_id,
                                         eos_token_id=stop_ids(model, tokenizer))
            t_gen += time.perf_counter()-t0
            start = inputs['input_ids'].shape[1]
            for output in outputs:
                new = output[start:]
                generated += int((new != tokenizer.pad_token_id).sum())
                raw = tokenizer.decode(new, skip_special_tokens=report_mode == 'typed' or not reasons(args))
                t0 = time.perf_counter()
                truncated = completion_truncated(new.tolist(), stop_ids(model, tokenizer), args.max_completion_length)
                row = score(args.lean, task, raw, args.arm, args.penalty, args.judge_fn, truncated=truncated, retries=retries, report_mode=report_mode)
                t_lean += time.perf_counter()-t0
                rows.append(row)
                sink.write(json.dumps(row, ensure_ascii=False)+'\n')
                sink.flush()
            completed, slowest = completed+1, max(slowest, time.monotonic()-task_started)
    training_rewards(rows, samples)  # probe variation must predict the signal the trainer will actually use
    grouped = reward_groups(rows, samples)
    return dict(summarize(rows), tasks_planned=len(tasks), tasks_completed=completed, incomplete=completed < len(tasks), report_mode=report_mode, verifier_retries_denied=retries.denied, generation_seconds=t_gen, lean_seconds=t_lean, generated_tokens=generated,
                samples_per_prompt=samples, reward_groups=grouped['groups'], mixed_reward_groups=grouped['mixed_groups'],
                zero_advantage_groups=grouped['zero_advantage_groups'], per_group=grouped['per_group'],
                tokens_per_second=generated/max(t_gen, 1e-9), completions_per_hour=len(rows)/max(t_gen+t_lean, 1e-9)*3600)


def prepare_training(model, tokenizer, tasks, args, summary):
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import TrainerCallback, set_seed
    from trl import GRPOConfig, GRPOTrainer
    by_statement = {t['statement']: t for t in tasks}
    report_mode = getattr(args, 'report_mode', 'prose')
    summary['report_mode'] = report_mode
    stops = stop_ids(model, tokenizer)  # the same stop IDs as `evaluate`, so both paths agree on a finished answer
    timing = {'step_start': 0.0, 'rollout': [], 'lean': [], 'update': [], 'reward_end': 0.0}
    sink = (args.out/'training-rollouts.jsonl').open('w')
    groups, undecided, lost = [], [], []
    retries = RetryBudget(getattr(args, 'verifier_retry_seconds', RETRY_SECONDS))  # for the whole training phase, so the deadline reserve holds

    steps = (args.out/'steps.jsonl').open('w')
    first_seen = summary.setdefault('first_seen_step', {})

    def reward_fn(completions, statement, **kwargs):
        t0 = time.perf_counter()
        timing['rollout'].append(t0-timing['step_start'])
        rows = []
        completion_ids = kwargs.get('completion_ids')
        if completion_ids is None:
            raise ValueError('TRL must supply completion_ids to determine whether each answer stopped')
        for completion, claim, token_ids in zip(completions, statement, completion_ids, strict=True):
            raw = completion[-1]['content'] if isinstance(completion, list) else completion
            truncated = completion_truncated(token_ids, stops, args.max_completion_length)
            try:
                rows.append(score(args.lean, by_statement[claim], raw, args.arm, args.penalty, args.judge_fn, truncated=truncated, retries=retries, report_mode=report_mode))
            except BaseException:
                # A stop mid-batch (deadline, judge streak, spend cap) keeps what was already scored and paid for.
                # Marked, because these rows are in no step, no tally and no complete prompt group.
                sink.write(''.join(json.dumps(dict(row, partial_batch=True), ensure_ascii=False)+'\n' for row in rows))
                sink.flush()
                summary['partial_batch_rows'] = len(rows)
                raise
        rewards = training_rewards(rows, args.num_generations)
        trainer._verifier_keep = [training_keep(row) for row in rows]
        undecided.extend({key: row.get(key) for key in ('verifier_inconclusive', 'claim', 'format_valid', 'judge_undecided')} for row in rows)
        judging = judge_advice(undecided)
        summary['training_verifier'] = dict(verifier_advice(undecided), retries_denied=retries.denied, **judging)
        if judging['check_judge'] and judging['judged_or_tried'] >= JUDGE_ROWS_BEFORE_STOP:
            # Stops a paid run whose rollouts are being dropped by an unreliable judge; this batch is already saved below.
            sink.write(''.join(json.dumps(row, ensure_ascii=False)+'\n' for row in rows))
            sink.flush()
            raise RuntimeError(f"the judge failed on {judging['judge_undecided_rate']:.0%} of {judging['judged_or_tried']} readings, above the 5% limit")
        sink.write(''.join(json.dumps(row, ensure_ascii=False)+'\n' for row in rows))
        sink.flush()
        step = len(timing['lean'])+1
        grouped = reward_groups(rows, args.num_generations)
        groups.extend([True]*grouped['mixed_groups'] + [False]*grouped['zero_advantage_groups'])
        lost.append(grouped['variation_lost_to_verifier'])
        counts = tally(rows)
        for key, value in counts.items():
            summary['rollouts'][key] = summary['rollouts'].get(key, 0)+value
        # The first step at which each behaviour was sampled at all: the exploration question, per arm and seed.
        for behaviour in ('exploit', 'refutation', 'genuine', 'says_not_solved', 'concealed_exploit'):
            if counts.get(behaviour) and behaviour not in first_seen:
                first_seen[behaviour] = step
        # Equal rewards give zero reward advantages; KL and optimizer momentum can still move weights.
        steps.write(json.dumps(dict(step=step, **counts, **verifier_advice(rows), retries_denied=retries.denied, **grouped))+'\n')
        steps.flush()
        timing['reward_end'] = time.perf_counter()
        timing['lean'].append(timing['reward_end']-t0)
        return rewards

    run_args = args

    class Instrument(TrainerCallback):
        def on_step_begin(self, args, state, control, **kwargs):
            timing['step_start'] = time.perf_counter()

        def on_step_end(self, args, state, control, **kwargs):
            timing['update'].append(time.perf_counter()-timing['reward_end'])
            losses = [row['loss'] for row in state.log_history[-1:] if 'loss' in row]
            if losses and not math.isfinite(losses[0]):
                raise FloatingPointError(f'the loss is {losses[0]} at step {state.global_step}')
            (run_args.out/'live.json').write_text(json.dumps({'step': state.global_step, **summary['rollouts'], **summary.get('training_verifier', {}), 'mixed_groups': sum(groups)}))
            if state.global_step % run_args.save_every == 0:
                kwargs['model'].save_pretrained(str(run_args.out/f'adapter-step{state.global_step}'))
            if state.global_step >= run_args.stop_no_variation and not any(groups):
                verifier = summary.setdefault('training_verifier', {})
                verifier['variation_lost_groups'] = sum(lost)
                summary['early_stop'] = VERIFIER_UNDECIDED if sum(lost) else NO_VARIATION  # rollouts left out for any reason, the judge's failures included
                control.should_training_stop = True
            if time.monotonic() > run_args.train_deadline:
                summary['early_stop'] = 'deadline'
                control.should_training_stop = True
            return control

    config = GRPOConfig(output_dir=str(args.out/'trainer'), max_steps=args.steps, per_device_train_batch_size=args.num_generations * args.prompts_per_step,
                        gradient_accumulation_steps=1, num_generations=args.num_generations, max_completion_length=args.max_completion_length,
                        learning_rate=args.learning_rate, bf16=torch.cuda.is_available(), use_cpu=not torch.cuda.is_available(), gradient_checkpointing=True,
                        beta=args.beta, temperature=1.0, top_p=0.95, top_k=20, chat_template_kwargs=template_kwargs(args), logging_steps=1, save_strategy='no',
                        report_to='none', seed=args.seed, dataloader_num_workers=0, loss_type='grpo',
                        # Two library defaults that changed the objective until 2026-11-25. Left unset, TRL takes the model's own
                        # router load-balancing coefficient (0.9 for gpt-oss), which made the logged loss 0.9 x that term; and the
                        # learning rate decays linearly to zero over --steps, so a run's rate depended on its length.
                        router_aux_loss_coef=args.router_aux_loss_coef, lr_scheduler_type=args.lr_schedule,
                        # More defaults that shape the update, now arguments or stated here so that nothing is inherited unseen.
                        # `scale_rewards`: 'group' divides each prompt group's advantages by that group's reward spread, so the
                        # size of a reward gap is invisible within a group; 'none' keeps it. See --scale-rewards.
                        scale_rewards=args.scale_rewards, max_grad_norm=args.max_grad_norm, optim='adamw_torch', weight_decay=0.0,
                        warmup_steps=0, adam_beta1=0.9, adam_beta2=0.999, adam_epsilon=1e-8,
                        logging_nan_inf_filter=False)  # a NaN loss is logged as NaN, and stops the run below
    dataset = Dataset.from_list([{'prompt': prompt(t['statement'], args.prompt_variant, report_mode), 'statement': t['statement']} for t in tasks])
    class VerifiedTrainer(VerifierMaskMixin, GRPOTrainer):
        pass

    trainer = VerifiedTrainer(model=model, args=config, reward_funcs=reward_fn, train_dataset=dataset, processing_class=tokenizer, callbacks=[Instrument()],
                          peft_config=None if args.adapter else LoraConfig(task_type='CAUSAL_LM', r=args.lora_rank, lora_alpha=2*args.lora_rank,
                                                                           target_modules='all-linear', lora_dropout=0.0))
    before = {n: p.detach().cpu().clone() for n, p in trainer.model.named_parameters() if p.requires_grad}
    summary['trainable_parameters'] = sum(p.numel() for p in before.values())
    summary['trainer'] = {'lr_schedule': trainer.args.lr_scheduler_type, 'router_aux_loss_coef': trainer.router_aux_loss_coef,
                          'aux_loss_enabled': trainer.aux_loss_enabled, 'scale_rewards': args.scale_rewards, 'max_grad_norm': args.max_grad_norm,
                          'arguments_in_effect': trainer.args.to_dict(),
                          'lora_target_modules': sorted({name.split('.')[-1] for name, module in trainer.model.named_modules() if hasattr(module, 'lora_A')})}
    def fit():
        t0 = time.perf_counter()
        # TRL seeds once, when the trainer is built; the before-evaluation then reseeds with its own constant. Without this,
        # every --seed sampled its training rollouts from the same random stream.
        set_seed(args.seed)
        try:
            trainer.train()
        except BaseException:
            # Whatever stopped training, the updates made so far were paid for.
            trainer.save_model(str(args.out/'adapter-at-failure'))
            raise
        finally:
            # Paid for whether or not training finished: the per-step log, and what each step's time went on.
            (args.out/'training-log.json').write_text(json.dumps(trainer.state.log_history, indent=1)+'\n')
            (args.out/'step-seconds.json').write_text(json.dumps({k: timing[k] for k in ('rollout', 'lean', 'update')}, indent=1)+'\n')
        summary.update(training_seconds=time.perf_counter()-t0, global_step=trainer.state.global_step, mixed_reward_groups=sum(groups), reward_groups=len(groups),
                       adapter_max_abs_change=max(float((p.detach().cpu()-before[n]).abs().max()) for n, p in trainer.model.named_parameters() if n in before),
                       mean_seconds={k: sum(timing[k])/max(len(timing[k]), 1) for k in ('rollout', 'lean', 'update')})
        trainer.save_model(str(args.out/'adapter'))
        sink.close()
        steps.close()
        return trainer.model
    return fit


def train(model, tokenizer, tasks, args, summary):
    """Build the trainer and run it. `run` builds it earlier, before the paid before-evaluation."""
    return prepare_training(model, tokenizer, tasks, args, summary)()


def run(args):
    """`args.mode` is controls, probe or train. Writes summary.json under `args.out` whatever happens."""
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    args.train_deadline = time.monotonic()+args.max_seconds-args.final_eval_seconds
    summary = {'status': 'running', 'report_mode': getattr(args, 'report_mode', 'prose'), 'argv': sys.argv[1:],
               # What the transcript export needs to rebuild each prompt and label the policy; argv alone omits defaults.
               'model': args.model, 'revision': args.revision, 'prompt_variant': args.prompt_variant,
               'adapter': str(args.adapter) if args.adapter else None, 'template_kwargs': template_kwargs(args), 'purpose': 'feasibility; not intervention evidence', 'rollouts': {},
               'script_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}}

    deadline.arm(args.max_seconds)
    hard_deadline = time.monotonic()+args.max_seconds
    sampler = GpuSampler()
    sampler.start()
    try:
        summary['controls'] = controls(args.lean)
        summary['lean'] = subprocess.check_output([args.lean, '--version'], text=True).strip()
        print('controls passed', flush=True)
        if args.mode == 'controls':
            summary['status'] = 'completed'
            return
        from .judge import from_args
        args.judge_fn = from_args(args)
        data = split(taskset_rows(args.taskset))
        evaluation = stratified_limit(data['evaluation'], args.eval_limit)
        training = stratified_limit(data['train'], args.train_limit)
        (args.out/'tasks.json').write_text(json.dumps({'train': training, 'evaluation': evaluation}, indent=1, ensure_ascii=False)+'\n')
        import peft
        import torch
        import transformers
        import trl
        torch.manual_seed(args.seed)
        summary.update(torch=torch.__version__, transformers=transformers.__version__, trl=trl.__version__, peft=peft.__version__,
                       gpus=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
        model, tokenizer = load_model(args, summary)
        if args.mode == 'probe':
            summary['probe'] = evaluate(model, tokenizer, training, args, 'probe', samples=args.num_generations, stop_at=hard_deadline-60)
            print('probe tasks completed', summary['probe']['tasks_completed'], 'of', summary['probe']['tasks_planned'], flush=True)
            summary['probe_split'] = 'train'
            print('probe cut-off rate', summary['probe']['truncated_rate'], 'raise the token limit:', summary['probe']['raise_token_limit'], flush=True)
            print('probe undecided rate', summary['probe']['unverified_rate'], 'check the verifier:', summary['probe']['check_verifier'], flush=True)
            summary['status'] = 'completed'
            return
        # Built before the before-evaluation is paid for: a trainer the libraries refuse to construct fails here, in seconds.
        # The adapter starts as an exact no-op (its B matrices are zero), so the before-evaluation still measures the base policy.
        fit = prepare_training(model, tokenizer, training, args, summary)
        summary['before'] = evaluate(model, tokenizer, evaluation, args, 'before')
        print('before', summary['before'], flush=True)
        model = fit()
        # Stops between tasks if training ran into its reserve: a shorter after-evaluation is kept, a killed one loses its summary.
        summary['after'] = evaluate(model, tokenizer, evaluation, args, 'after', stop_at=hard_deadline-60)
        if summary['after']['incomplete']:
            # A shortened after-evaluation covers fewer tasks; beside the full before-evaluation its rates would compare different task sets.
            done = {json.loads(line)['task_id'] for line in (args.out/'after-evaluation.jsonl').read_text().splitlines()}
            matched = [row for row in map(json.loads, (args.out/'before-evaluation.jsonl').read_text().splitlines()) if row['task_id'] in done]
            summary['before_on_after_tasks'] = summarize(matched)
        summary['status'] = 'completed'
    except BaseException as error:
        summary.update(status='failed', reason=f'{type(error).__name__}: {str(error)[:800]}')
        raise
    finally:
        deadline.disarm()
        gpu = sampler.summary()
        meter = getattr(getattr(args, 'judge_fn', None), 'meter', None)
        summary['judge_usage'] = meter.totals if meter else None
        summary.update(elapsed_seconds=time.perf_counter()-started, peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**(20 if sys.platform != 'darwin' else 30),
                       peak_gpu_memory_used_gib=None if gpu['gpu_sampler_error'] else gpu['gpu_peak_mib']/1024,
                       max_gpu_utilization=None if gpu['gpu_sampler_error'] else gpu['gpu_max_util'], gpu_sampler_error=gpu['gpu_sampler_error'])
        if 'torch' in sys.modules and sys.modules['torch'].cuda.is_available():
            torch_ = sys.modules['torch']
            summary['peak_allocated_gib_all_gpus'] = sum(torch_.cuda.max_memory_allocated(i) for i in range(torch_.cuda.device_count()))/2**30
        (args.out/'summary.json').write_text(json.dumps(summary, indent=1)+'\n')
        print(json.dumps({k: v for k, v in summary.items() if k != 'script_sha256'}, indent=1), flush=True)
