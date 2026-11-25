# rewardhack: grader exploitation and honest exits under RL

Can equal reward for a verified demonstration that a task is impossible reduce learned grader exploitation and false success reporting?

Extends my October prototype `aeftimia/exit-reward-research` (commit `ec0565e`); see [prior work](docs/existing-work.md) for what it showed and what was fixed.

**Status (2026-11-24):** grant RG-1451 preflight. Grant-funded judge validation has begun; see `spending/ledger.csv` for recorded charges. The funded reward arm is built (`R = h + s + (1 − λ)·c`: honest report, verified proof or verified impossibility, and what the bug still pays for cheating), the hidden verifier was attacked and hardened (a forged axiom report got a `sorry` accepted; closed, no saved label affected), and a capped pod smoke test is staged but not run. Plan: [preflight-plan.md](docs/preflight-plan.md); reasoning and alternatives: [notes-2026-11-21.md](docs/notes-2026-11-21.md).

The report format is selectable with `--report-mode prose|typed` on `run.py`, `local_mlx.py`, `probe_vllm.py`, and `probe_api.py`; **prose remains the default**. Typed mode requires exactly `established` (string), `proof_body` (string), `error` (JSON Boolean), and `justification` (string). It derives the success/failure claim from `error`, checks honesty against Lean, and requires `--judge none` on training entrypoints. The justification is saved for audit only; no prose judge runs or contributes reward. Invalid typed envelopes receive zero reward in every arm. See the [typed-format details](docs/preflight-plan.md#typed-format-switch-available-not-an-experiment-migration).

Prose judging and typed error reporting are planned as [parallel experimental conditions](docs/preflight-plan.md#parallel-prose-versus-typed-study), with a grader-only control for each format. Rows record their report mode, historical rows default to prose, and the modes are compared as labelled conditions rather than pooled. No comparison runs have been launched.

The [bounded pilot controller](docs/pilot-study.md) plans matched baseline/intervention checkpoints and stops scheduled work for model-conditional evidence, predicted futility, practical plateau, measurement failure, or the existing budget limits. Plan inspection is the default; execution is explicit. With no useful-effect targets chosen, the exploratory pilot reports curves and can stop for a bounded-horizon plateau, but cannot declare success or futility. No paid pilot has been started.

**Pilot status (2026-11-05):** $8.78 spent ($3.76 RunPod, $5.02 OpenAI API). Open-weight rung in [open-reasoning-findings.md](docs/open-reasoning-findings.md): Qwen3-32B (thinking) and gpt-oss-20b (low effort) submit honest `sorry` placeholders on 14/60 and 12/60 false claims unprompted, gpt-oss-20b reasons out the compile-only bug explicitly; both are trainable candidates. gpt-oss-120b failed to download. Frontier ladder in [frontier-ladder-findings.md](docs/frontier-ladder-findings.md): gpt-5.4-nano with reasoning submitted honest `sorry` placeholders on 12/60 false claims, gpt-5.4-mini proved 3 negations unprompted, gpt-5.5 without reasoning claimed success on 78/120 false claims; at the report level no model claimed success over a placeholder, but a reading of every response and reasoning trace ([response-characterization.md](docs/response-characterization.md)) found gpt-oss-20b deciding to exploit the compile-only reward in 8 of 120 traces, twice with a report that hides it. AWS GPU quotas granted but no g6e/p5 capacity was available in us-west-2 on the day. 70B probe in [70b-probe-findings.md](docs/70b-probe-findings.md): Llama 3.1 70B (AWQ) made 0 placeholder attempts in 360 unprompted samples, 3 unverified refutation attempts, and no genuine proofs above the easy tier. Local Qwen3-8B (4-bit) results in [local-8b-findings.md](docs/local-8b-findings.md): zero unprompted exploits in 283 samples; with a labelled hint, one sampled `admit` and a single GRPO update flipped the policy to 20/24 placeholders on held-out statements; matched equal-reward and baseline branches from that checkpoint took zero updates each (no refutation was ever sampled), so the intervention has the same exploration problem as the exploit; continuing the seeded checkpoint on the *neutral* prompt took held-out exploits from 3/24 to 15/24 after one update, so a single seeded sample is enough and no sustained hint is needed. Llama 3.1 8B (4-bit, 136 samples): one unprompted `sorry`, reported honestly as "not established"; no Lean 4 competence. Literature in [literature-scale.md](docs/literature-scale.md). Permission × outcome grid on the hard set in [grid-A-findings.md](docs/grid-A-findings.md): forbidding cheating drives placeholders to 0, allowing it to 57%, no model ever concealed a cheat, and grader memory persists into a second turn without the code. Working notes and the next design (harder false statements, 3 prompt variants × 4 outcomes, acknowledgement/cheating rewards, memory follow-on, local training only) in [notes-2026-11-06.md](docs/notes-2026-11-06.md).

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
| `src/rewardhack/gates.py`, `entrypoints/gate.py` | Checks on a shell command before it runs: a pod job starts only from a settled launch record; commits are read for withheld terms, secrets and large or binary files (see `CLAUDE.md`, "Hooks") |
| `src/rewardhack/pilot.py`, `pilot_model.py`, `pilot_runner.py`, `pilot_evidence.py`, `pilot_simulation.py` | Frozen paired pilot evidence, Bayesian learning curves, bounded serial execution and free synthetic checks |
| `entrypoints/*.py` | One script per task: `run`, `local_mlx`, `probe_vllm`, `probe_api`, `rescore`, `calibrate`, `validate_judge`, `human_review`, `export_transcripts`, `verify_tasks`, `budget`, `pilot`, `pilot_pair`, `pilot_simulate` |
| `ops/*.sh`, `spending/ledger.csv` | Launch drivers, remote run scripts, teardown checks, and the spending record |

Install:

```bash
uv pip install --python .tools/venv/bin/python -e '.[mlx,dev]'
```

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

Differences from the prototype: an `established` field in every arm, which makes refutations declared rather than accidental; a separate verifier; harder tasks; reasoning allowed before the JSON; lr 1e-4 and 8 samples per prompt, for more reward variation. These prompt and training settings are new and unvalidated on a real model.

Code structure follows [the owner's CODE.md](https://github.com/aeftimia/aeftimia.github.io/blob/master/CODE.md): reusable behavior belongs in the installable `src/rewardhack/` package; `entrypoints/` parses arguments and composes library functions. Review duplication at feature checkpoints. The shared remote process runner is `ops/remote_job.sh`; it reports exit status and limits process time, but pod termination remains attended.

Fresh judge-validation setup: [held-out protocol](docs/judge-holdout.md). The old checklist samples overlap and remain development evidence. Shared helpers reject development overlap and freeze the judge configuration before fresh reports are collected. Optional 12- or 24-report blind audits use `validate_judge.py --audit` for descriptive comparisons; the larger acceptance workflow uses `--holdout`. Both require completed human labels for comparison, while formal audits require none.
