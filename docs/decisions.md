# Decisions of record

The owner's standing decisions, one entry each, newest first within a topic. The reasoning, alternatives and evidence
are in the dated notes this links to; this page is the index any session or reader can rely on. When a decision
changes, edit its entry and link the note that changed it.

## Experiment plan

- **λ=0.5 survey first, on every model (2026-11-26).** Before any λ sweep, run the Part 1 setup (60 steps, λ=0.5,
  unseeded, no saturation stop) on each model to measure how often refutations appear and what training costs, and
  allocate the remaining funds from that. gpt-oss-20b first (launched 26 Nov), then Qwen3-32B with thinking after a
  short measuring pass. Notes section 77.
- **Model selection (2026-11-26).** Train models that sample the exploit unprompted: gpt-oss-20b and Qwen3-32B with
  thinking. Llama 3.1 70B is not a training candidate: untrained, it exploited 4 times in about 1,400 answers (plus
  0 of 360 in the 4-bit probe) and never refuted, though it reports false success often; revisit only if the
  false-report question becomes the priority. gpt-oss-120b is probed untrained first; training it has no budget line.
- **Reallocation only after the core plan (2026-11-26).** Complete the core plan well under budget first, then scope
  the options (larger models, the λ sweep, seeding variants) with their costs, and only then ask the funder about moving
  money between lines. No reallocation request before that.
- **Refutation capacity is not the bottleneck (2026-11-26).** In the hard task set 58 of 80 false statements
  refute with a single witness, easier than their true twins' universal proofs, yet gpt-oss-20b declared a negation
  once in 1,920 answers. Missing refutations are a matter of recognising falsity and saying so. Open check: confirm the
  verifier accepts a hand-written witness refutation.
- **Seeding is compared with natural emergence, whatever natural emergence shows (2026-11-26).** Matched arms (same
  model, λ, seed, steps). Also test whether seeding carries over to neutral prompts: refutations on held-out neutral
  prompts after a seeded phase, with a dose ablation (one seeded step against several), by a hint and by injected
  verified refutations. Prior evidence, for the exploit only: local Qwen3-8B, one seeded example took held-out exploits
  from 3/24 to 15/24 after one update on the neutral prompt.
- **Start without the λ sweep (2026-11-26).** Its budget is held in reserve until Part 1 calls for it.
  `docs/preflight-plan.md`, notes section 68.
- **Run length (2026-11-26).** 60 steps per survey cell: behaviour settled by about step 30 on gpt-oss-20b. Extend a
  cell from its last checkpoint only when something is still moving (a sampled refutation, λ=0's length drift).
  Notes sections 76–77.
- **No saturation stop in Part 1 runs (2026-11-26).** A first refutation is a rare escape from a plateau. Follow-up:
  replace the stop with a decision rule on a hierarchical random-effects prior across models and λ, fitted from each
  model's runs and the probes. Notes section 77.

## Data kept

- **Full trajectories on a RunPod network volume (2026-11-26).** Every run's adapter snapshots and trainer checkpoints,
  on the grant's storage line (about $0.07 per GB-month; 100–150 GB is about $7–10 a month). The laptop keeps rows,
  logs, summaries and the final adapter. The volume ties pods to one datacenter.
- **Rows carry their provenance (2026-11-26).** Model, revision, arm, λ, stage, report mode, task set and run id on
  every row; `rewardhack.provenance.cells` turns results into per-run counts. Notes section 77.

## Money and pods

- **Grant RG-1451 (approved 2026-11-18), $2,500 by line:** gpt-oss-20b GPU $800, Qwen3-32B GPU $900, API $300,
  storage $100, contingency $400. Moving money between lines, or changing plans, is told to the funder first.
  `spending/ledger.csv` and `entrypoints/budget.py` track it.
- **Every paid run is approved by the owner for its amount**, before it starts.
- **Grant pods run only in the project's own RunPod space**, billed separately from any other account. Its balance
  auto-reloads, so costs are booked from pod hours × rate and checked against the reload history, not from balance
  differences.
- **Paid pods remove themselves.** Every job arms a removal timer; caps are multiples of measured or estimated times
  (`gates.MULTIPLES`); the spend gate (`entrypoints/gate.py ready`) settles a launch before deployment. `CLAUDE.md`.

## Code

- **Installable library plus one-off entrypoints; constants stated once; lookup tables over branching; no silent
  fallbacks; semantic version bumped with every package change.** `CLAUDE.md` and the owner's `CODE.md`.
