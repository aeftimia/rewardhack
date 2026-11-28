# rewardhack: grader exploitation and honest exits under RL

Can equal reward for a verified demonstration that a task is impossible reduce learned grader exploitation and false success reporting?

Extends my October prototype `aeftimia/exit-reward-research` (commit `ec0565e`); see [prior work](docs/existing-work.md) for what it showed and what was fixed.

## Running jobs

Start with **[Job approvals and next-run gates](docs/jobs.md)** to see which experiments have finished, what evidence would justify a follow-up, and which runs need owner approval.

Launch approved training through the [experiment runner](docs/experiment-batches.md). For a direct pod job, inspection or cleanup, use the [RunPod procedure](docs/runpod-operations.md). Trace a run ID to [scored answers, checkpoints and archive receipts](docs/experiment-batches.md#looking-up-outputs-and-checkpoints).

<details>
<summary>Earlier experiments and design notes</summary>

| Question | Evidence |
| --- | --- |
| What did the first 60-step gpt-oss-20b runs learn? | [Part 1 results and follow-up design, notes §§76–83](docs/notes-2026-11-21.md#76-part-1-60-steps-each-with-the-section-7172-changes-both-arms-learn-the-exploit-λ0-discloses-it-the-baseline-hides-half-26-nov) |
| How are prose reports and typed error flags scored? | [Format specification and controls](docs/preflight-plan.md#typed-format-switch-available-not-an-experiment-migration) |
| How does the bounded pilot choose checkpoints and stopping points? | [Pilot design](docs/pilot-study.md) |
| Which models exploited the grader before training? | [Open-weight probes](docs/open-reasoning-findings.md), [70B probe](docs/70b-probe-findings.md), [frontier models](docs/frontier-ladder-findings.md), [response and reasoning traces](docs/response-characterization.md) |
| Did seeded exploitation persist on neutral prompts? What did explicit permission change? | [Local training](docs/local-8b-findings.md), [permission grid](docs/grid-A-findings.md), [November pilot notes](docs/notes-2026-11-06.md) |

</details>

Layout: shared code is the installable package `src/rewardhack/`; every experiment and task is a one-off script in `entrypoints/` that imports it. Raw results stay under `results/`, spending under `spending/`.

| Path | Purpose |
|---|---|
| `src/rewardhack/grader.py` | Buggy grader, shown verbatim to the model |
| `src/rewardhack/verifier.py` | Independent verifier, never shown |
| `src/rewardhack/tasks.py`, `tasks_hard.py` | Paired Lean requests with Lean-checked references: the v1 set and the calibrated hard set; split, selection and reference verification |
| `src/rewardhack/prompts.py` | The one shared prompt, its permission variants, and the follow-on turn |
| `src/rewardhack/reporting.py` | Prose/typed format options and strict typed-envelope parsing |
| `src/rewardhack/outcomes.py` | Arm rewards (`baseline`, `equal_reward`, `report_correct`), the honesty bit, outcome labels, report triage |
| `src/rewardhack/judge.py` | Reads the free-text report as one error flag (never shown the verifier's verdict); backends; validation against the model-read labels |
| `src/rewardhack/hf_grpo.py`, `mlx_rl.py` | TRL GRPO path for rented GPUs; own GRPO-style loop on Apple silicon |
| `src/rewardhack/vllm_probe.py`, `openai_probe.py` | Inference probes: vLLM on rented GPUs, hosted API with metered cost caps |
| `src/rewardhack/rescore.py`, `calibrate.py`, `budget.py` | Rescoring saved rows, per-statement calibration from read labels, spending guard |
| `src/rewardhack/gates.py`, `entrypoints/gate.py` | Launch records and spend checks. See [enforcement and coverage](CLAUDE.md#hooks). |
| `src/rewardhack/pilot.py`, `pilot_model.py`, `pilot_runner.py`, `pilot_evidence.py`, `pilot_simulation.py` | Frozen paired pilot evidence, Bayesian learning curves, bounded serial execution and free synthetic checks |
| `entrypoints/*.py` | One script per task: `run`, `local_mlx`, `probe_vllm`, `probe_api`, `rescore`, `calibrate`, `validate_judge`, `human_review`, `export_transcripts`, `verify_tasks`, `budget`, `pilot`, `pilot_pair`, `pilot_simulate` |
| `ops/*.sh`, `spending/ledger.csv` | Launch drivers, remote run scripts, teardown checks, and the spending record |

Install:

```bash
uv pip install --python .tools/venv/bin/python -e '.[mlx,dev]'
```

The training sampler defaults to Transformers. `run.py train --rollout-backend vllm
--rollout-url http://127.0.0.1:8011` uses a local, initially sleeping vLLM server with
per-batch LoRA reloads (attention and, on gpt-oss, the experts) and untruncated sampling. Install the server separately with
`requirements-vllm.txt`; its Torch and NumPy constraints differ from the trainer's.
The pod smoke script enables this path with `SMOKE_VLLM_ROLLOUTS=1`, API judging,
`SMOKE_SKIP_PROBE=1`, and `SMOKE_SECOND_RUN=0`. See [notes section 63](docs/notes-2026-11-21.md#63-trl-upgrade-and-external-vllm-training-rollouts-25-november)
for the original server contract and validation.

Local checks:

```bash
.tools/venv/bin/python tests/test_offline.py
```

```bash
.tools/venv/bin/python tests/test_lean.py
```

```bash
.tools/venv/bin/python tests/test_learning.py
```

```bash
bash ops/dry_run.sh
```

Each arm asks the model to declare what it established. A separate verifier checks that claim against the Lean task. Archived run configurations record the sampling and training settings for each experiment.

Code structure follows [the owner's CODE.md](https://github.com/aeftimia/aeftimia.github.io/blob/master/CODE.md): reusable behavior belongs in the installable `src/rewardhack/` package; `entrypoints/` parses arguments and composes library functions. Review duplication at feature checkpoints. The shared remote process runner is `ops/remote_job.sh`; it reports exit status, limits process time and arms the pod's own removal timer, so a pod removes itself after its job (see `CLAUDE.md`, "Paid pods"). The [driver lifecycle](docs/driver-lifecycle.md) describes launch ownership, relaunches, and result isolation.

Fresh judge-validation setup: [held-out protocol](docs/judge-holdout.md). The old checklist samples overlap and remain development evidence. Shared helpers reject development overlap and freeze the judge configuration before fresh reports are collected. Optional 12- or 24-report blind audits use `validate_judge.py --audit` for descriptive comparisons; the larger acceptance workflow uses `--holdout`. Both require completed human labels for comparison, while formal audits require none.
