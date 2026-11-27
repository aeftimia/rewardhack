# Qwen completion investigation, 27 November 2026

The six-step bf16 measurement was correctly stopped by the operator. Its saved
24-answer initial evaluation contains 21 cutoffs, all without a closing
`</think>`, three valid submissions and one genuine proof. No completed update
is preserved. This is not evidence of a training trend or an optimizer failure.
The evaluation generated 47,320 tokens in 205.85 seconds; this is batched
inference throughput, not training throughput.

## Offline experiments

`entrypoints/audit_qwen_completion.py` replays the saved termination evidence
and checks the current pre-training guard. The guard refuses the actual 21/24
cutoff result. It was added by the coordinator after the interrupted run; this
branch does not claim it as a new fix.

Whole-completion replay of the older neutral Qwen3-32B AWQ probe:

| Cap | Complete / 120 | Parseable complete | Cut off |
| --- | --- | --- | --- |
| 2,048 | 42 | 40 | 65.0% |
| 4,096 | 78 | 76 | 35.0% |
| 6,144 | 103 | 101 | 14.2% |
| 8,192 | 116 | 114 | 3.3% |

Only 8,192 passes the existing truncation advice rule. Its sampled token total
is 1.95 times the replay at 2,048, not four times: many answers terminate early.
This is not fresh generation, and the older quantization, tasks and sampling
settings differ from the current hard-task bf16 measurement. It selects a
candidate limit; it cannot establish that limit's adequacy on the current run.
The saved short answers cannot reveal how their missing continuations would end.

Do not salvage JSON from unfinished thinking or switch thinking off: either
would change what this experiment measures. Keep lambda=0, neutral/prose and
thinking on. The existing rollout server sizes its context to answer limit plus
prompt room, so increasing the answer limit also increases the server context.

## Preservation fix

The smoke runner fixed `--save-every` at eight, beyond the six-step measuring
pass. This did not cause the lack of completed steps in the stopped run, but it
would leave completed early updates without periodic checkpoints on a later
interruption. `SMOKE_SAVE_EVERY=1` now selects per-step adapter snapshots and
resumable optimizer checkpoints. The default stays eight for the GPT sweep.
The driver rejects zero and malformed intervals before contacting a pod.

A CPU integration test completes one real tiny-model optimizer step, fails
during scoring of the second, and checks that step one's optimizer checkpoint
and adapter snapshot exist. It is not a Qwen GPU memory or speed test.

## Approved paid measurement (27 November; not yet launched)

Use the same two-card bf16 setup, 8,192 completion tokens, six steps, and
`SMOKE_SAVE_EVERY=1`. Keep the initial evaluation and its refusal guard. If it
refuses again, retain the evaluation and revise the limit before training.
Record rollout, update, evaluation and checkpoint-transfer timings separately.

Conservative planning estimate: 12 minutes per step (four times the old,
unmeasured three-minute estimate), about 14 minutes per evaluation (four times
the measured short evaluation), plus model/trainer overhead: round to 110
minutes inside the trainer. These are estimates, not new measurements.
Use 9,900 training seconds (1.5 times that estimate), a 1,800-second final
evaluation reserve, and a 250-minute outer job cap (the driver also budgets
71 minutes of setup and 810 seconds of stage exit allowances).

With the existing 8,100-second copy window, the gate adds 161 minutes outside
the job cap. At the previous pod's $5.415/hour and a $0.25 judge cap, the total
worst case is $37.34. Typical compute is provisionally about $12 plus transfer
time, but has not been measured. Recheck the offered rate and checkpoint size
before launch. The owner approved this revised budget on 27 November. The
unrounded total is $37.34275, so the launch ceiling rounds upward to $37.35.
A new launch still needs committed code and the complete launch gate.

The owner also raised matching the token ceiling across models. GPT's three
completed neutral/prose runs used 2,048: training cutoffs were 6/960 (baseline),
2/960 (lambda=0), and 59/960 (lambda=0.5). The lambda=0.5 cutoffs were concentrated
in steps 1–24, despite zero final-evaluation cutoffs. A common 8,192 ceiling is
recommended for future matched comparisons; the running GPT sweep is not
modified by this Qwen measurement. Equal token ceilings do not equate GPU cost
or reasoning effort across model families.

Reproduce the offline audit from this branch with the shared project Python:

```sh
PYTHONPATH=src /Users/aeftimia/Rapid\ Grant\ Reward\ Hacking/.tools/venv/bin/python entrypoints/audit_qwen_completion.py \
  --measurement /private/tmp/rewardhack-refute-permission/results/runpod-qwen-lambda0-measure-20261126/train-lambda0 \
  --reference /Users/aeftimia/Rapid\ Grant\ Reward\ Hacking/results/runpod-open-reasoning/qwen32-think \
  --caps 2048 4096 6144 8192 --out results/qwen-completion-audit.json
```
