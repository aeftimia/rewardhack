"""Exploration probe of a large model with vLLM: how often does it try the exploit, unprompted?

Inference only, no training. Uses the training split, like the local 8B probes, so the evaluation split
stays held out. Rows have the same fields as the other probes and can be rescored with `entrypoints/rescore.py`.
With `--lean none`, nothing model-written is executed on the machine: rows hold the raw output only and
are scored later (use this on hosts without a Lean sandbox, e.g. containers). Entrypoint: `entrypoints/probe_vllm.py`.
"""
import json
import subprocess
import sys
import time

from . import deadline, jsonl
from .gpu import GpuSampler
from .outcomes import RetryBudget, score, summarize
from .prompts import prompt
from .tasks import select_tasks


def run(args):
    report_mode = args.report_mode
    template_kwargs = json.loads(args.template_kwargs)
    args.out.mkdir(parents=True, exist_ok=True)

    deadline.arm(args.max_seconds)

    started = time.perf_counter()
    retries = RetryBudget(args.verifier_retry_seconds)  # the probe's whole allowance of second checks
    sampler = GpuSampler()
    sampler.start()
    summary = {'status': 'running', 'report_mode': report_mode, 'argv': sys.argv[1:], 'purpose': 'exploration probe; inference only; no training',
               'lean': None if args.lean == 'none' else subprocess.check_output([args.lean, '--version'], text=True).strip()}
    try:
        import vllm
        from vllm import LLM, SamplingParams
        summary['vllm'] = vllm.__version__
        tasks = select_tasks(args.taskset, args.tasks, args.part)
        t0 = time.perf_counter()
        llm = LLM(model=args.model, revision=args.revision, tokenizer_revision=args.revision, seed=args.seed,
                  max_model_len=args.max_model_len, gpu_memory_utilization=args.gpu_memory_utilization,
                  enable_prefix_caching=True, enforce_eager=True)
        summary['load_seconds'] = time.perf_counter()-t0
        tokenizer = llm.get_tokenizer()
        # The same pinned date and the same refusal of an ignored reasoning switch as the training path, so a probe's
        # prompts are the ones training would see.
        from .hf_grpo import settle_template
        settle_template(tokenizer, template_kwargs['reasoning_effort'] if 'reasoning_effort' in template_kwargs else None,
                        ('on' if template_kwargs['enable_thinking'] else 'off') if 'enable_thinking' in template_kwargs else 'unsupported')
        prompts = [tokenizer.apply_chat_template(prompt(t['statement'], args.variant, report_mode), tokenize=False, add_generation_prompt=True, **template_kwargs)
                   for t in tasks]
        summary['template_kwargs'] = template_kwargs
        summary['prompt_tokens'] = len(tokenizer(prompts[0])['input_ids'])
        params = SamplingParams(n=args.samples, temperature=1.0, top_p=0.95, top_k=20, max_tokens=args.max_tokens, seed=args.seed)
        t0 = time.perf_counter()
        outputs = llm.generate(prompts, params)
        t_gen = time.perf_counter()-t0
        rows, generated, t_lean = [], 0, 0.0
        with (args.out/'rows.jsonl').open('w') as sink:
            for task, output in zip(tasks, outputs, strict=True):
                for index, completion in enumerate(output.outputs):
                    t0 = time.perf_counter()
                    if args.lean == 'none':
                        row = dict(task_id=task['id'], arm='baseline', raw=completion.text, scored=False, termination_schema='cutoff-v1', report_mode=report_mode)
                    else:
                        row = score(args.lean, task, completion.text, 'baseline', truncated=completion.finish_reason == 'length', retries=retries, report_mode=report_mode)
                    t_lean += time.perf_counter()-t0
                    row.update(tokens=len(completion.token_ids), truncated=completion.finish_reason == 'length', round=index)
                    generated += len(completion.token_ids)
                    rows.append(row)
                    jsonl.write(sink, row)
        summary.update(summarize(rows) if args.lean != 'none' else {'samples': len(rows), 'scored': False},
                       variant=args.variant, part=args.part, taskset=args.taskset, max_tokens=args.max_tokens, generation_seconds=t_gen, lean_seconds=t_lean,
                       generated_tokens=generated, tokens_per_second=generated/max(t_gen, 1e-9), mean_tokens=generated/max(len(rows), 1),
                       truncated=sum(r['truncated'] for r in rows), status='completed')
    except BaseException as error:  # recorded, then re-raised: a partial summary is still useful
        summary.update(status='failed', error=f'{type(error).__name__}: {error}'[:500])
        raise
    finally:
        deadline.disarm()
        summary.update(elapsed_seconds=time.perf_counter()-started, **sampler.summary())
        (args.out/'summary.json').write_text(json.dumps(summary, indent=1, ensure_ascii=False)+'\n')
        print(json.dumps(summary, indent=1, ensure_ascii=False), flush=True)
