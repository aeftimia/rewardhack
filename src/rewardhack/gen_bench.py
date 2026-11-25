"""Cost benchmarks for generation, which is most of a training step. Entrypoint: `entrypoints/bench_generation.py`.

`hf`: time one decoding step of the training sampler (transformers `generate`) at several batch sizes and in the
states the trainer puts the model in, profile it, and save untrained LoRA adapters shaped like the trainer's.
`server`: against a running vLLM server, load those adapters and time concurrent completions.
Nothing here scores an answer: no Lean, no judge, no reward.
"""
import json
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

from .hf_grpo import load_model, template_kwargs
from .prompts import prompt
from .tasks import split, stratified_limit, taskset_rows


def bench_tasks(args, count):
    return stratified_limit(split(taskset_rows(args.taskset))['train'], max(count, 2))  # the limit keeps true/false pairs, so two is its minimum


def run_hf(args):
    import torch
    from peft import LoraConfig, get_peft_model
    args.out.mkdir(parents=True, exist_ok=True)
    cuda = torch.cuda.is_available()
    summary = {'status': 'running', 'model': args.model, 'revision': args.revision, 'torch': torch.__version__, 'measurements': []}

    def save():
        (args.out/'summary.json').write_text(json.dumps(summary, indent=1)+'\n')

    model, tokenizer = load_model(SimpleNamespace(model=args.model, revision=args.revision, attn_implementation=args.attn_implementation,
                                                  quant='none', mode='probe', adapter=None), summary)
    summary['experts_implementation'] = model.config._experts_implementation
    task = bench_tasks(args, 1)[0]
    inputs = tokenizer.apply_chat_template(prompt(task['statement'], 'neutral', 'prose'), tokenize=True, return_dict=True, return_tensors='pt',
                                           add_generation_prompt=True, **template_kwargs(args)).to(next(model.parameters()).device)
    summary['prompt_tokens'] = inputs['input_ids'].shape[1]

    def decode(label, batch, new_tokens):
        """Exactly `new_tokens` steps for `batch` copies of one prompt: the stop tokens are held off, so the time is per step."""
        if cuda:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        with torch.inference_mode():
            out = model.generate(input_ids=inputs['input_ids'].repeat(batch, 1), attention_mask=inputs['attention_mask'].repeat(batch, 1),
                                 max_new_tokens=new_tokens, min_new_tokens=new_tokens, do_sample=True, temperature=1.0, top_p=0.95, top_k=20,
                                 pad_token_id=tokenizer.pad_token_id)
        if cuda:
            torch.cuda.synchronize()
        seconds = time.perf_counter()-started
        steps = out.shape[1]-inputs['input_ids'].shape[1]
        row = dict(label=label, batch=batch, steps=steps, seconds=seconds, ms_per_step=1000*seconds/steps, tokens_per_second=batch*steps/seconds,
                   training_mode=model.training, peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30 if cuda else None)
        summary['measurements'].append(row)
        save()
        print(json.dumps(row), flush=True)

    model.eval()
    decode('warm-up', 1, 8)
    for batch in args.batches:
        decode('base', batch, args.new_tokens)
    for implementation in args.experts:
        # The library reads this setting at every forward, so no reload is needed to compare expert-layer code paths.
        default = model.config._experts_implementation
        model.config._experts_implementation = implementation
        decode(f'base experts={implementation}', args.focus_batch, args.new_tokens)
        model.config._experts_implementation = default

    lora = get_peft_model(model, LoraConfig(task_type='CAUSAL_LM', r=args.lora_rank, lora_alpha=2*args.lora_rank, target_modules='all-linear', lora_dropout=0.0))
    summary['lora_target_modules'] = sorted({name.split('.')[-1] for name, module in lora.named_modules() if hasattr(module, 'lora_A')})
    summary['lora_trainable_parameters'] = sum(p.numel() for p in lora.parameters() if p.requires_grad)
    lora.save_pretrained(str(args.adapters/'all-linear'))
    model = lora
    model.eval()
    for batch in args.lora_batches:
        decode('lora', batch, args.new_tokens)
    # Does a step cost the same deep into an answer? It does only if the cache of earlier tokens is in use.
    decode('lora long', args.focus_batch, args.long_tokens)
    if args.profile_tokens:
        from torch.profiler import ProfilerActivity, profile
        with profile(activities=[ProfilerActivity.CPU]+([ProfilerActivity.CUDA] if cuda else [])) as prof:
            decode('lora profiled', args.focus_batch, args.profile_tokens)
        events = prof.key_averages()
        (args.out/'profile-by-cpu.txt').write_text(events.table(sort_by='self_cpu_time_total', row_limit=30))
        if cuda:
            (args.out/'profile-by-gpu.txt').write_text(events.table(sort_by='self_cuda_time_total', row_limit=30))
    # The trainer generates with the model left in training mode and gradient checkpointing on.
    model.base_model.model.gradient_checkpointing_enable()
    model.train()
    # Half lengths: if this state turns the cache off, every step reprocesses the whole answer and the full lengths would not fit the stage's time.
    decode('lora training-mode', args.focus_batch, args.new_tokens//2)
    decode('lora training-mode long', args.focus_batch, args.long_tokens//2)
    summary['status'] = 'completed'
    save()


def post(url, body, timeout):
    """One JSON POST. An HTTP error status is an answer here (does the server accept this adapter?), so it is returned, not raised."""
    request = urllib.request.Request(url, data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status, text = response.status, response.read().decode()
    except urllib.error.HTTPError as error:  # fallback-ok: the status and body are recorded as the measurement; connection failures still raise
        status, text = error.code, error.read().decode()
    return dict(status=status, text=text, seconds=time.perf_counter()-started)


def run_server(args):
    args.out.mkdir(parents=True, exist_ok=True)
    summary = {'status': 'running', 'url': args.url, 'adapters': {}, 'measurements': []}

    def save():
        (args.out/'summary.json').write_text(json.dumps(summary, indent=1)+'\n')

    served = [args.model]
    for item in args.adapter:
        name, path = item.split('=', 1)
        loaded = post(f'{args.url}/load_lora_adapter', dict(lora_name=name, lora_path=path), 120)
        record = dict(load_status=loaded['status'], load_seconds=loaded['seconds'], load_text=loaded['text'][:1000])
        if loaded['status'] == 200:
            again = post(f'{args.url}/load_lora_adapter', dict(lora_name=name, lora_path=path, load_inplace=True), 120)
            record.update(reload_in_place_status=again['status'], reload_in_place_seconds=again['seconds'], reload_in_place_text=again['text'][:1000])
            served.append(name)
        summary['adapters'][name] = record
        save()
        print(name, json.dumps(record), flush=True)

    tasks = bench_tasks(args, max(args.concurrency))
    for model in served:
        for concurrency in args.concurrency if model != args.model else args.base_concurrency:
            def one(task, model=model):
                answer = post(f'{args.url}/chat/completions', dict(
                    model=model, messages=prompt(task['statement'], 'neutral', 'prose'), max_tokens=args.max_tokens, temperature=1.0, top_p=0.95,
                    top_k=20, chat_template_kwargs=json.loads(args.template_kwargs)), args.request_seconds)
                if answer['status'] != 200:
                    raise RuntimeError(f"completion failed with {answer['status']}: {answer['text'][:500]}")
                body = json.loads(answer['text'])
                return body['usage']['completion_tokens'], body['choices'][0]['finish_reason']
            started = time.perf_counter()
            with ThreadPoolExecutor(concurrency) as pool:
                results = list(pool.map(one, [tasks[i % len(tasks)] for i in range(concurrency)]))
            seconds = time.perf_counter()-started
            tokens = sum(count for count, _ in results)
            row = dict(model=model, concurrency=concurrency, seconds=seconds, tokens=tokens, tokens_per_second=tokens/seconds,
                       seconds_per_completion=seconds/concurrency, cut_off=sum(reason == 'length' for _, reason in results))
            summary['measurements'].append(row)
            save()
            print(json.dumps(row), flush=True)
    summary['status'] = 'completed'
    save()
