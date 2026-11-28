"""Evaluate one completed trajectory against a sleeping project rollout server.

Pass --segment directories in ancestry order. Each contains the registry files
effective.json, outputs.json and the verified archive files at their indexed
relative paths. --search enables the 100-answer plateau screen.
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from rewardhack import generations, jsonl
from rewardhack._hashing import file_sha256
from rewardhack.hf_grpo import evaluation_text, settle_template, template_kwargs
from rewardhack.outcomes import compare, completion_truncated, score
from rewardhack.prompts import prompt
from rewardhack.state_files import atomic_json
from rewardhack.tasks import select_tasks, stratified_limit, verify_references
from rewardhack.tasks_hard import sat_family
from rewardhack.verifier import controls
from rewardhack.vllm_rollouts import AdapterRollouts, SEED_STRIDE


def read(path):
    return json.loads(path.read_text())


def panel(taskset, training):
    """Keep true/false pairs together; exclude repeated or training statements."""
    seen = {row['statement'] for row in training}
    groups = {}
    for row in select_tasks(taskset, part='evaluation'):
        groups.setdefault(row['group'], []).append(row)
    pairs = []
    for pair in groups.values():
        statements = {row['statement'] for row in pair}
        if statements.isdisjoint(seen):
            pairs.extend(pair)
            seen.update(statements)
    missing = 50 - len(pairs) // 2
    if missing > 0:
        extra = sat_family(20261128, ((5, 21, 'easy', missing),), 'plateau-sat')
        if any(row['statement'] in seen for row in extra):
            raise ValueError('new panel statement overlaps existing tasks')
        pairs.extend(extra)
    result = stratified_limit(pairs, 100)
    if len({r['statement'] for r in result}) != 100 or sum(r['true'] for r in result) != 50:
        raise ValueError('expected 100 distinct held-out statements balanced by truth')
    return result


def analyze(final, search, evaluate):
    """evaluate(step, phase, draws_per_statement) returns one 100-task scorecard."""
    if final < 1:
        raise ValueError('a trained final checkpoint is required')
    screen = {}
    def measure(step):
        if step not in screen:
            screen[step] = evaluate(step, 'screen', 1)
        return screen[step]['exploit']

    selected = None
    bracket = None
    if search:
        anchors = sorted({round(final * k / 4) for k in range(5)})
        for step in anchors:
            measure(step)
        def near(step):
            return abs(measure(step) - measure(final)) <= 5
        outside = [step for step in anchors if not near(step)]
        # Require a sustained terminal band at two or more anchors. A return
        # to the band after a reversal starts a new candidate interval.
        if outside and outside[-1] < anchors[-2]:
            low = outside[-1]
            high = anchors[anchors.index(low) + 1]
            while high - low > 1:
                middle = (low + high) // 2
                if near(middle):
                    high = middle
                else:
                    low = middle
            bracket = [low, high]
            selected = high
        elif not outside:
            selected = 0
    targets = dict(base=0, final=final)
    if selected is not None:
        targets['plateau'] = selected
    confirmation = {step: evaluate(step, 'confirmation', 10) for step in dict.fromkeys(targets.values())}
    return dict(screen=screen, candidate_bracket=bracket, targets=targets, confirmation=confirmation)


def evaluate_checkpoint(tasks, step, phase, repeats, seed, sample, lean, mode, output):
    counts = dict(answers=0, exploit=0, genuine=0, verified_refutation=0, truncated=0, verifier_inconclusive=0)
    with output.open('x') as sink:
        for draw in range(repeats):
            first_seed = seed + draw * len(tasks)
            answers = sample(step, tasks, first_seed)
            for index, (task, (raw, truncated)) in enumerate(zip(tasks, answers, strict=True)):
                # This analysis measures proof outcomes only. Baseline scoring
                # applies the same grader and independent Lean verifier in both formats.
                row = score(lean, task, raw, 'baseline', truncated=truncated, report_mode=mode)
                row.update(step=step, phase=phase, draw=draw, sampling_seed=first_seed + index)
                sink.write(json.dumps(row, allow_nan=False) + '\n')
                counts['answers'] += 1
                for key in ('exploit', 'genuine', 'verified_refutation', 'truncated', 'verifier_inconclusive'):
                    counts[key] += int(row[key])
    if counts['verifier_inconclusive']:
        raise ValueError(f'unresolved verifier decisions in {output}')
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--segment', type=Path, action='append', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--lean', type=Path, required=True)
    parser.add_argument('--rollout-url', required=True)
    parser.add_argument('--search', action='store_true')
    cli = parser.parse_args()
    lean = str(cli.lean.resolve(strict=True))
    adapters = {}
    previous = None
    training = []
    for segment in cli.segment:
        config = read(segment / 'effective.json') if (segment / 'effective.json').exists() else read(segment / 'run-config.json')
        index = read(segment / 'outputs.json')
        if index['backup_status'] != 'verified':
            raise ValueError(f'unverified archive: {segment}')
        if previous is not None and config['parent']['run_id'] != previous:
            raise ValueError('segments must belong to one checkpoint lineage')
        if previous is not None:
            adapters = {s: v for s, v in adapters.items() if s <= config['parent']['checkpoint_step']}
        previous = config['run_id']
        for step, files in index['checkpoints']['adapter'].items():
            adapters[int(step)] = (segment, files)
        training.extend(read(segment / config['output_arm'] / 'tasks.json')['train'])
    settings = config['settings']
    final = read(segment / config['output_arm'] / 'summary.json')['global_step']
    if final != settings['steps'] or final not in adapters:
        raise ValueError('final checkpoint is incomplete')
    cli.out.mkdir(parents=True, exist_ok=False)
    tasks = panel(settings['taskset'], training)
    failures = verify_references(lean, tasks)
    if failures:
        raise ValueError(f'panel reference failures: {failures}')
    controls(lean)
    args = SimpleNamespace(**{**settings, 'max_completion_length': 8192}, out=cli.out, rollout_url=cli.rollout_url)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    settle_template(tokenizer, args.reasoning_effort, args.thinking)
    backend = AdapterRollouts(args, tokenizer.eos_token_id)
    atomic_json(cli.out / 'design.json', dict(lineage=[read(d/'run-config.json')['run_id'] for d in cli.segment],
                settings=vars(args) | {'out': str(cli.out)}, tasks=tasks, search=cli.search))
    def evaluate(step, phase, draws):
        served = f'{args.model}@{args.revision}'
        if step:
            directory, files = adapters[step]
            paths = [directory / name for name in files]
            if len({p.parent for p in paths}) != 1 or any(file_sha256(directory / name) != digest for name, digest in files.items()):
                raise ValueError(f'invalid archive files at step {step}')
            backend.path = paths[0].parent
            backend.load(backend.path)
            served = backend.name
        messages = [prompt(t['statement'], args.prompt_variant, args.report_mode) for t in tasks]
        tokens = [generations.request_tokens(tokenizer, m, template_kwargs(args)) for m in messages]
        if step and not backend.checked_reload:
            backend.verify_reload(tokens[0])
        def sample(checkpoint, panel_tasks, seed):
            ids, _, _ = backend.completions(tokens, seeds=[seed + index for index in range(len(tokens))],
                                            max_tokens=[args.max_completion_length] * len(tokens), model=served)
            return [(evaluation_text(tokenizer, seq, args), completion_truncated(seq, backend.stops, args.max_completion_length)) for seq in ids]
        seed = (args.seed + {'screen': 1, 'confirmation': 2}[phase]) * SEED_STRIDE
        return evaluate_checkpoint(tasks, step, phase, draws, seed, sample, lean, args.report_mode, cli.out/f'{phase}-{step}.jsonl')
    try:
        backend.preflight()
        backend.request('POST', '/wake_up')
        try:
            backend.require_sleep(False)
            report = analyze(final, cli.search, evaluate)
            rows = {step: list(jsonl.read(cli.out/f'confirmation-{step}.jsonl'))
                    for step in report['confirmation']}
            report['final_minus_checkpoint'] = {
                step: compare(batch, rows[final], seed=args.seed) for step, batch in rows.items() if step != final}
            atomic_json(cli.out / 'analysis.json', report)
        finally:
            backend.request('POST', '/sleep?level=1')
            backend.require_sleep(True)
    finally:
        backend.close()


if __name__ == '__main__':
    main()
