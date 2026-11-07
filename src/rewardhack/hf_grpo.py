"""GPU path: measure exploration (`probe`) or train with TRL GRPO (`train`), recording cost-relevant timings.

Adapted from exit-reward-research@ec0565e scripts/grpo_lean_pilot.py. Arms share one prompt; only the
reward differs. Entrypoint: `entrypoints/run.py`.
"""
import hashlib
import json
import resource
import signal
import subprocess
import sys
import time
from pathlib import Path

from .gpu import GpuSampler
from .outcomes import score, sorry_attempt, summarize
from .prompts import prompt
from .tasks import split, taskset_rows
from .verifier import controls


def load_model(args, summary):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.padding_side = 'left'
    if not tokenizer.chat_template and args.model.startswith('hf-internal-testing/'):
        tokenizer.chat_template = "{% for m in messages %}{{ m['role'] }}: {{ m['content'] }}\n{% endfor %}assistant:"  # CPU dry run only
    kwargs = {'revision': args.revision, 'attn_implementation': 'sdpa'}
    cuda = torch.cuda.is_available()
    kwargs['dtype'] = torch.bfloat16 if cuda else torch.float32
    if args.quant == '8bit':
        kwargs['quantization_config'] = BitsAndBytesConfig(load_in_8bit=True)
    elif args.quant == '4bit':
        kwargs['quantization_config'] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                                                           bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    if cuda:
        kwargs['device_map'] = 'auto'
    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(args.model, **kwargs)
    if args.quant != 'none' and args.mode == 'train':
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    if args.adapter:
        from peft import PeftModel
        summary['initial_adapter_sha256'] = hashlib.sha256((args.adapter/'adapter_model.safetensors').read_bytes()).hexdigest()
        model = PeftModel.from_pretrained(model, str(args.adapter), is_trainable=args.mode == 'train')
    summary['model_load_seconds'] = time.perf_counter()-t0
    summary['parameters_total'] = sum(p.numel() for p in model.parameters())
    return model, tokenizer


def template_kwargs(args):
    return {'enable_thinking': args.thinking == 'on'} if args.thinking != 'unsupported' else {}


def evaluate(model, tokenizer, tasks, args, stage):
    import torch
    device = next(model.parameters()).device
    rows, generated, t_gen, t_lean = [], 0, 0.0, 0.0
    torch.manual_seed(31415)
    model.eval()
    path = args.out/f'{stage}-evaluation.jsonl'
    with path.open('w') as sink:
        for task in tasks:
            inputs = tokenizer.apply_chat_template(prompt(task['statement'], args.prompt_variant), tokenize=True, return_dict=True,
                                                   return_tensors='pt', add_generation_prompt=True, **template_kwargs(args)).to(device)
            t0 = time.perf_counter()
            with torch.inference_mode():
                outputs = model.generate(**inputs, max_new_tokens=args.max_completion_length, do_sample=True, temperature=1.0, top_p=0.95,
                                         top_k=20, num_return_sequences=args.eval_samples, pad_token_id=tokenizer.pad_token_id)
            t_gen += time.perf_counter()-t0
            start = inputs['input_ids'].shape[1]
            for output in outputs:
                new = output[start:]
                generated += int((new != tokenizer.pad_token_id).sum())
                raw = tokenizer.decode(new, skip_special_tokens=False if args.thinking == 'on' else True)
                t0 = time.perf_counter()
                row = score(args.lean, task, raw, args.arm)
                t_lean += time.perf_counter()-t0
                rows.append(row)
                sink.write(json.dumps(row, ensure_ascii=False)+'\n')
                sink.flush()
    return dict(summarize(rows), generation_seconds=t_gen, lean_seconds=t_lean, generated_tokens=generated,
                tokens_per_second=generated/max(t_gen, 1e-9), completions_per_hour=len(rows)/max(t_gen+t_lean, 1e-9)*3600)


def train(model, tokenizer, tasks, args, summary):
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import TrainerCallback
    from trl import GRPOConfig, GRPOTrainer
    by_statement = {t['statement']: t for t in tasks}
    timing = {'step_start': 0.0, 'rollout': [], 'lean': [], 'update': [], 'reward_end': 0.0}
    sink = (args.out/'training-rollouts.jsonl').open('w')
    groups = []

    def reward_fn(completions, statement, **kwargs):
        t0 = time.perf_counter()
        timing['rollout'].append(t0-timing['step_start'])
        rewards = []
        for completion, claim in zip(completions, statement, strict=True):
            raw = completion[-1]['content'] if isinstance(completion, list) else completion
            row = score(args.lean, by_statement[claim], raw, args.arm)
            sink.write(json.dumps(row, ensure_ascii=False)+'\n')
            rewards.append(row['reward'])
            summary['rollouts']['n'] += 1
            summary['rollouts']['exploit'] += row['exploit']
            summary['rollouts']['genuine'] += row['genuine']
            summary['rollouts']['refutation'] += row['verified_refutation']
            summary['rollouts']['sorry_attempts'] += sorry_attempt(row)
            summary['rollouts']['reward'] += row['reward']
        sink.flush()
        for i in range(0, len(rewards), args.num_generations):
            groups.append(len(set(rewards[i:i+args.num_generations])) > 1)
        timing['reward_end'] = time.perf_counter()
        timing['lean'].append(timing['reward_end']-t0)
        return rewards

    run_args = args

    class Instrument(TrainerCallback):
        def on_step_begin(self, args, state, control, **kwargs):
            timing['step_start'] = time.perf_counter()

        def on_step_end(self, args, state, control, **kwargs):
            timing['update'].append(time.perf_counter()-timing['reward_end'])
            (run_args.out/'live.json').write_text(json.dumps({'step': state.global_step, **summary['rollouts'], 'mixed_groups': sum(groups)}))
            if state.global_step % run_args.save_every == 0:
                kwargs['model'].save_pretrained(str(run_args.out/f'adapter-step{state.global_step}'))
            if state.global_step >= run_args.stop_no_variation and not any(groups):
                summary['early_stop'] = 'no within-prompt reward variation'
                control.should_training_stop = True
            if time.monotonic() > run_args.train_deadline:
                summary['early_stop'] = 'deadline'
                control.should_training_stop = True
            return control

    config = GRPOConfig(output_dir=str(args.out/'trainer'), max_steps=args.steps, per_device_train_batch_size=args.num_generations * args.prompts_per_step,
                        gradient_accumulation_steps=1, num_generations=args.num_generations, max_completion_length=args.max_completion_length,
                        learning_rate=args.learning_rate, bf16=torch.cuda.is_available(), use_cpu=not torch.cuda.is_available(), gradient_checkpointing=True,
                        beta=0.0, temperature=1.0, top_p=0.95, top_k=20, chat_template_kwargs=template_kwargs(args), logging_steps=1, save_strategy='no',
                        report_to='none', seed=args.seed, dataloader_num_workers=0, loss_type='grpo')
    dataset = Dataset.from_list([{'prompt': prompt(t['statement'], args.prompt_variant), 'statement': t['statement']} for t in tasks])
    trainer = GRPOTrainer(model=model, args=config, reward_funcs=reward_fn, train_dataset=dataset, processing_class=tokenizer, callbacks=[Instrument()],
                          peft_config=None if args.adapter else LoraConfig(task_type='CAUSAL_LM', r=args.lora_rank, lora_alpha=2*args.lora_rank,
                                                                           target_modules='all-linear', lora_dropout=0.0))
    before = {n: p.detach().cpu().clone() for n, p in trainer.model.named_parameters() if p.requires_grad}
    summary['trainable_parameters'] = sum(p.numel() for p in before.values())
    t0 = time.perf_counter()
    trainer.train()
    summary.update(training_seconds=time.perf_counter()-t0, global_step=trainer.state.global_step, mixed_reward_groups=sum(groups), reward_groups=len(groups),
                   adapter_max_abs_change=max(float((p.detach().cpu()-before[n]).abs().max()) for n, p in trainer.model.named_parameters() if n in before),
                   mean_seconds={k: sum(timing[k])/max(len(timing[k]), 1) for k in ('rollout', 'lean', 'update')})
    trainer.save_model(str(args.out/'adapter'))
    (args.out/'training-log.json').write_text(json.dumps(trainer.state.log_history, indent=1)+'\n')
    sink.close()
    return trainer.model


def run(args):
    """`args.mode` is controls, probe or train. Writes summary.json under `args.out` whatever happens."""
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    args.train_deadline = time.monotonic()+args.max_seconds-args.final_eval_seconds
    summary = {'status': 'running', 'argv': sys.argv[1:], 'purpose': 'feasibility; not intervention evidence', 'rollouts': dict.fromkeys(('n', 'exploit', 'genuine', 'refutation', 'sorry_attempts', 'reward'), 0),
               'script_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}}

    def hard_stop(*unused):
        raise TimeoutError('hard deadline')
    signal.signal(signal.SIGALRM, hard_stop)
    signal.alarm(args.max_seconds)
    sampler = GpuSampler()
    sampler.start()
    try:
        summary['controls'] = controls(args.lean)
        summary['lean'] = subprocess.check_output([args.lean, '--version'], text=True).strip()
        print('controls passed', flush=True)
        if args.mode == 'controls':
            summary['status'] = 'completed'
            return
        data = split(taskset_rows(args.taskset))
        evaluation = data['evaluation'][:args.eval_limit]
        training = data['train'][:args.train_limit]
        (args.out/'tasks.json').write_text(json.dumps({'train': training, 'evaluation': evaluation}, indent=1, ensure_ascii=False)+'\n')
        import peft
        import torch
        import transformers
        import trl
        torch.manual_seed(args.seed)
        summary.update(torch=torch.__version__, transformers=transformers.__version__, trl=trl.__version__, peft=peft.__version__,
                       gpus=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
        model, tokenizer = load_model(args, summary)
        summary['before'] = evaluate(model, tokenizer, evaluation, args, 'before')
        print('before', summary['before'], flush=True)
        if args.mode == 'probe':
            summary['status'] = 'completed'
            return
        model = train(model, tokenizer, training, args, summary)
        summary['after'] = evaluate(model, tokenizer, evaluation, args, 'after')
        summary['status'] = 'completed'
    except BaseException as error:
        summary.update(status='failed', reason=f'{type(error).__name__}: {str(error)[:800]}')
        raise
    finally:
        signal.alarm(0)
        gpu = sampler.summary()
        summary.update(elapsed_seconds=time.perf_counter()-started, peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**(20 if sys.platform != 'darwin' else 30),
                       peak_gpu_memory_used_gib=None if gpu['gpu_sampler_error'] else gpu['gpu_peak_mib']/1024,
                       max_gpu_utilization=None if gpu['gpu_sampler_error'] else gpu['gpu_max_util'], gpu_sampler_error=gpu['gpu_sampler_error'])
        if 'torch' in sys.modules and sys.modules['torch'].cuda.is_available():
            torch_ = sys.modules['torch']
            summary['peak_allocated_gib_all_gpus'] = sum(torch_.cuda.max_memory_allocated(i) for i in range(torch_.cuda.device_count()))/2**30
        (args.out/'summary.json').write_text(json.dumps(summary, indent=1)+'\n')
        print(json.dumps({k: v for k, v in summary.items() if k != 'script_sha256'}, indent=1), flush=True)
