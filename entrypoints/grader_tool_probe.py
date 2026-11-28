#!/usr/bin/env python3
"""Grader-tool episodes from the untrained model on a vLLM rollout server, without loading a trainer.

A smoke test of the `--grader-runs` path on a real model and server (ops/remote_grader_tool_probe.sh): the same
`hf_grpo.evaluate` a training run's evaluations use, sampling the served base model through `AdapterRollouts.awake`,
with turns stopped at the answer's EOS or a tool call, per-request token limits, and real Lean behind both the tool
and the final verdict. Baseline arm, prose reports, no judge: disclosure is not read here. Writes probe-evaluation.jsonl
(every draft and the feedback the model saw) and summary.json.
"""
import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

from transformers import AutoConfig, AutoTokenizer

from rewardhack import grader_tool
from rewardhack.hf_grpo import evaluate, settle_template, stage_tasks
from rewardhack.outcomes import RETRY_SECONDS
from rewardhack.sampling import DEFAULT_SEED
from rewardhack.verifier import controls, lean_version
from rewardhack.vllm_rollouts import AdapterRollouts

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--lean', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--revision', required=True)
    parser.add_argument('--rollout-url', required=True)
    parser.add_argument('--max-completion-length', type=int, required=True)
    parser.add_argument('--grader-runs', type=grader_tool.check_runs, default=3)
    parser.add_argument('--tasks', type=int, default=4, help='probe tasks, stratified over the training split')
    parser.add_argument('--samples', type=int, default=2, help='episodes per task')
    parser.add_argument('--reasoning-effort', default='low')
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
    parser.set_defaults(thinking='unsupported', prompt_variant='neutral', report_mode='prose', arm='baseline', penalty=0.0,
                        judge_fn=None, taskset='hard', verifier_retry_seconds=RETRY_SECONDS)
    args = parser.parse_args()
    if args.grader_runs == 0:
        parser.error('--grader-runs 0 is the control: there is no tool to probe')
    args.out.mkdir(parents=True, exist_ok=False)
    summary = dict(status='running', model=args.model, revision=args.revision, grader_runs=args.grader_runs,
                   max_completion_length=args.max_completion_length, tasks=args.tasks, samples=args.samples)
    started = time.monotonic()
    backend = None
    try:
        summary['controls'] = controls(args.lean)  # the verifier and grader agree with known cases before any sampling
        summary['lean'] = lean_version(args.lean)
        tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
        grader_tool.prepare_tokenizer(tokenizer)  # before settle_template, as load_model does
        settle_template(tokenizer, args.reasoning_effort, args.thinking)
        # `evaluate` reads only the model's config when the server samples: the trainer's weights are never loaded here.
        model = SimpleNamespace(config=AutoConfig.from_pretrained(args.model, revision=args.revision))
        backend = AdapterRollouts(args, tokenizer.eos_token_id, result_tokens=grader_tool.result_reserve(args.grader_runs))
        backend.request('POST', '/sleep?level=1')
        backend.preflight()
        tasks = stage_tasks(args.taskset, 'probe', args.tasks)
        summary['probe'] = evaluate(model, tokenizer, tasks, args, 'probe', samples=args.samples, rollouts=backend)
        summary['status'] = 'completed'
    except BaseException as error:
        summary.update(status='failed', error=f'{type(error).__name__}: {str(error)[:800]}')
        raise
    finally:
        if backend is not None:
            backend.close()
        summary['elapsed_seconds'] = time.monotonic() - started
        (args.out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
        print(json.dumps({k: summary[k] for k in summary if k != 'controls'}, indent=1), flush=True)
