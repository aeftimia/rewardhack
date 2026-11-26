"""Free local measurements on Apple silicon with a 4-bit MLX model.

`probe`: how often does the untrained model try the exploit? Uses the training split only, so the
evaluation split stays untouched for later runs.
`rl`: a small GRPO-style loop to test whether exploit attempts get amplified once sampled. This is my
own trainer on quantized weights, so its timings and rates do not transfer to the TRL GPU pipeline.
Entrypoint: `entrypoints/local_mlx.py`.
"""
import json
import random
import sys
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler
from mlx_lm.tuner.utils import linear_to_lora_layers

from . import jsonl
from .outcomes import RetryBudget, score, sorry_attempt, summarize, training_rewards, training_keep, verifier_advice
from .prompts import prompt
from .sampling import TEMPERATURE, TOP_K, TOP_P
from .tasks import select_tasks, split, taskset_rows

SAMPLER = dict(temp=TEMPERATURE, top_p=TOP_P, top_k=TOP_K)


def encode(tokenizer, statement, variant, thinking, report_mode='prose'):
    return tokenizer.apply_chat_template(prompt(statement, variant, report_mode), add_generation_prompt=True, tokenize=True,
                                         enable_thinking=thinking)


def sample(model, tokenizer, ids, max_tokens):
    """Returns generated text, token ids (EOS included, so training learns to stop), seconds, truncated."""
    t0 = time.perf_counter()
    text, tokens, reason = '', [], None
    for piece in stream_generate(model, tokenizer, ids, max_tokens=max_tokens, sampler=make_sampler(**SAMPLER)):
        text += piece.text
        tokens.append(piece.token)
        reason = piece.finish_reason
    return text, tokens, time.perf_counter()-t0, reason == 'length'


def attach_lora(model, args, adapter=None):
    """Freeze the base weights and add LoRA layers; load `adapter` (adapter.safetensors from an earlier run) when given."""
    model.freeze()
    linear_to_lora_layers(model, args.lora_layers, {'rank': args.lora_rank, 'scale': args.lora_scale, 'dropout': 0.0})
    if adapter:
        load_adapter(model, adapter)


def load_adapter(model, adapter):
    """Load saved LoRA weights into a model that already has its LoRA layers, refusing a file that does not fit them.

    The file holds only the trainable weights, so the load cannot be strict about the base weights. Left at that, a file
    from another LoRA shape or another model matches no name, loads nothing, and the run goes on from untrained
    layers under the saved run's name. So the names are compared here: the file must hold exactly the trainable ones.
    """
    saved, expected = set(mx.load(str(adapter))), {name for name, _ in tree_flatten(model.trainable_parameters())}
    if saved != expected:
        raise ValueError(f'{adapter} does not fit this model\'s LoRA layers: {len(expected-saved)} trainable weights are not in the file '
                         f'(first: {sorted(expected-saved)[:3]}) and {len(saved-expected)} weights in the file are not trainable here (first: {sorted(saved-expected)[:3]})')
    model.load_weights(str(adapter), strict=False)
    mx.eval(model.parameters())


def probe(args, model, tokenizer):
    report_mode = args.report_mode
    tasks = select_tasks(args.taskset, args.tasks)
    random.Random(args.seed).shuffle(tasks)
    rows, started = [], time.perf_counter()
    retries = RetryBudget(args.verifier_retry_seconds)
    gen_seconds = tokens_total = 0
    with (args.out/'rows.jsonl').open('w') as sink:
        # Whole passes over all tasks, so a time-capped run stays balanced.
        for round_index in range(args.samples):
            if time.perf_counter()-started > args.max_seconds:
                break
            for task in tasks:
                if time.perf_counter()-started > args.max_seconds:
                    break
                text, tokens, seconds, truncated = sample(model, tokenizer, encode(tokenizer, task['statement'], args.variant, args.thinking, report_mode), args.max_tokens)
                row = score(args.lean, task, text, 'baseline', truncated=truncated, retries=retries, report_mode=report_mode)
                row.update(tokens=len(tokens), seconds=seconds, truncated=truncated, round=round_index)
                rows.append(row)
                gen_seconds += seconds
                tokens_total += len(tokens)
                jsonl.write(sink, row)
    result = dict(summarize(rows), report_mode=report_mode, variant=args.variant, thinking=args.thinking, max_tokens=args.max_tokens,
                  truncated=sum(r['truncated'] for r in rows), tokens_per_second=tokens_total/max(gen_seconds, 1e-9),
                  mean_tokens=tokens_total/max(len(rows), 1), elapsed_seconds=time.perf_counter()-started,
                  peak_gib=mx.get_peak_memory()/2**30)
    (args.out/'summary.json').write_text(json.dumps(result, indent=1)+'\n')
    print(json.dumps(result, indent=1))


def completion_loss(model, prompt_ids, tokens, advantage):
    """Mean per-token negative log-probability of the completion, weighted by advantage."""
    inputs = mx.array([prompt_ids+tokens[:-1]])
    logits = model(inputs)[0, len(prompt_ids)-1:]
    return advantage*nn.losses.cross_entropy(logits, mx.array(tokens)).mean()


def attempt_rate(model, tokenizer, tasks, args, samples):
    report_mode = args.report_mode
    rows, retries = [], RetryBudget(args.verifier_retry_seconds)  # per monitor pass
    for task in tasks:
        ids = encode(tokenizer, task['statement'], args.variant, args.thinking, report_mode)
        for _ in range(samples):
            raw, _, _, truncated = sample(model, tokenizer, ids, args.max_tokens)
            rows.append(score(args.lean, task, raw, args.arm, args.penalty, args.judge_fn, truncated=truncated, retries=retries, report_mode=report_mode))
    return rows


def rl(args, model, tokenizer):
    report_mode = args.report_mode
    tasks = split(taskset_rows(args.taskset))['train']
    retries = RetryBudget(args.verifier_retry_seconds)  # for the whole training loop
    # The evaluation split stays held out: monitoring takes groups from the training split that are excluded from training.
    groups = sorted({t['group'] for t in tasks})
    random.Random(args.seed).shuffle(groups)
    monitor_groups = set(groups[:6])
    monitor = [t for t in tasks if t['group'] in monitor_groups]
    train_tasks = [t for t in tasks if t['group'] not in monitor_groups]
    attach_lora(model, args, args.adapter)  # branching from a saved run: same LoRA shape, trained on under this run's arm
    trainable = sum(v.size for _, v in tree_flatten(model.trainable_parameters()))
    optimizer = optim.Adam(learning_rate=args.learning_rate)
    loss_and_grad = nn.value_and_grad(model, completion_loss)
    order = random.Random(args.seed)
    log = (args.out/'steps.jsonl').open('w')
    rollouts = (args.out/'rollouts.jsonl').open('w')
    monitors = (args.out/'monitor.jsonl').open('w')
    started = time.perf_counter()
    verifier_rows = []

    def run_monitor(step):
        rows = attempt_rate(model, tokenizer, monitor, args, args.monitor_samples)
        entry = dict(step=step, report_mode=report_mode, n=len(rows), sorry=sum(sorry_attempt(r) for r in rows), exploit=sum(r['exploit'] for r in rows),
                     genuine_true=sum(r['genuine'] for r in rows if r['true']), true=sum(r['true'] for r in rows),
                     exploit_and_claims=sum(r['exploit_and_claims'] for r in rows), format_valid=sum(r['format_valid'] for r in rows), **verifier_advice(rows))
        jsonl.write(monitors, entry, ensure_ascii=True)
        print('monitor', entry, flush=True)
        (args.out/f'monitor-step{step}.jsonl').write_text(jsonl.dumps(rows))

    run_monitor(0)
    for step in range(1, args.steps+1):
        if time.perf_counter()-started > args.max_seconds:
            print('time cap reached', flush=True)
            break
        task = order.choice(train_tasks)
        ids = encode(tokenizer, task['statement'], args.variant, args.thinking, report_mode)
        t0 = time.perf_counter()
        samples = [sample(model, tokenizer, ids, args.max_tokens) for _ in range(args.group)]
        t_rollout = time.perf_counter()-t0
        t0 = time.perf_counter()
        rows = [score(args.lean, task, s[0], args.arm, args.penalty, args.judge_fn, truncated=s[3], retries=retries, report_mode=report_mode) for s in samples]
        t_lean = time.perf_counter()-t0
        verifier_rows.extend({'verifier_inconclusive': r['verifier_inconclusive']} for r in rows)
        kept = [reward for reward in training_rewards(rows, len(rows)) if reward is not None]
        rewards = mx.array(kept or [0.0])  # mean and spread over the decided rollouts only
        std = float(rewards.std())
        t0 = time.perf_counter()
        update = False
        if std > 0:
            advantages = iter(((rewards-rewards.mean())/(std+1e-4)).tolist())
            total = None
            for row, (_, tokens, _, _) in zip(rows, samples, strict=True):
                if not training_keep(row):
                    continue
                advantage = next(advantages)
                if len(tokens) < 2:
                    continue
                _, grads = loss_and_grad(model, ids, tokens, advantage/len(kept))
                total = grads if total is None else tree_map(lambda a, b: a+b, total, grads)
                mx.eval(total)
            if total is not None:
                optimizer.update(model, total)
                mx.eval(model.parameters(), optimizer.state)
            update = total is not None
        jsonl.write_rows(rollouts, (dict(r, step=step, tokens=len(s[1]), truncated=s[3]) for r, s in zip(rows, samples, strict=True)))
        entry = dict(step=step, report_mode=report_mode, task=task['id'], reward=float(rewards.mean()) if kept else None, mixed=update, sorry=sum(sorry_attempt(r) for r in rows),
                     exploit=sum(r['exploit'] for r in rows), rollout_seconds=t_rollout, lean_seconds=t_lean,
                     update_seconds=time.perf_counter()-t0, tokens=sum(len(s[1]) for s in samples), peak_gib=mx.get_peak_memory()/2**30,
                     **verifier_advice(rows))
        jsonl.write(log, entry, ensure_ascii=True)
        print(entry, flush=True)
        if step % args.monitor_every == 0:
            run_monitor(step)
    mx.save_safetensors(str(args.out/'adapter.safetensors'), dict(tree_flatten(model.trainable_parameters())))
    (args.out/'summary.json').write_text(json.dumps(dict(**verifier_advice(verifier_rows), report_mode=report_mode, trainable=trainable, updates=sum(row['mixed'] for row in jsonl.read(args.out/'steps.jsonl')),
                                                         elapsed_seconds=time.perf_counter()-started, argv=sys.argv[1:], adapter=str(args.adapter) if args.adapter else None,
                                                         purpose='plumbing and mechanism; own trainer; not evidence about the hypothesis'), indent=1)+'\n')


def run(args):
    args.out.mkdir(parents=True, exist_ok=True)
    mx.random.seed(args.seed)
    from .judge import from_args
    args.judge_fn = from_args(args)
    model, tokenizer = load(args.model, revision=args.revision)
    if args.mode == 'probe' and args.adapter:
        attach_lora(model, args, args.adapter)
    (probe if args.mode == 'probe' else rl)(args, model, tokenizer)
