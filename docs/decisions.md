# Decisions of record

Start with [job approvals and next-run gates](jobs.md) for the current queue. This page records the grant decisions and earlier design choices behind it.

## Original experiment design, 26 November

The permission-prompt screen and one-time Qwen benchmark supersede the original survey order. [Screening decisions and evidence](screening-plan.md)

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
- **Run plan and ballpark costs (2026-11-26).** Core grid 30 runs (about $315), full plan with add-ons about 55–65
  (about $630), most of it Qwen3-32B, whose cost is unmeasured. Planning figures, not an approval. Notes section 82.
- **Start without the λ sweep (2026-11-26).** Its budget is held in reserve until Part 1 calls for it.
  `docs/preflight-plan.md`, notes section 68.
- **Run length (2026-11-26).** 60 steps per survey cell: behaviour settled by about step 30 on gpt-oss-20b. Extend a
  cell from its last checkpoint only when something is still moving (a sampled refutation, λ=0's length drift).
  Notes sections 76–77.
- **No saturation stop in Part 1 runs (2026-11-26).** A first refutation is a rare escape from a plateau. Follow-up:
  replace the stop with a decision rule on a hierarchical random-effects prior across models and λ, fitted from each
  model's runs and the probes. Notes section 77.

## Data kept

- **Full trajectories on S3 (2026-11-26).** Keep adapter snapshots and trainer checkpoints on the grant's storage line. S3 costs less and allows pods in different datacenters. [Storage decision, notes §80](notes-2026-11-21.md#80-full-trajectories-archived-on-s3-and-qwen3-32b-as-a-two-card-part-1-job-owner-26-nov) · [Archive and checkpoint receipts](experiment-batches.md#looking-up-outputs-and-checkpoints)
- **Rows carry their provenance (2026-11-26).** Model, revision, arm, λ, stage, report mode, task set and run id on
  every row; `rewardhack.provenance.cells` turns results into per-run counts. Notes section 77.

## Money and pods

- **Grant RG-1451 (approved 2026-11-18), $2,500 by line:** gpt-oss-20b GPU $800, Qwen3-32B GPU $900, API $300,
  storage $100, contingency $400. Moving money between lines, or changing plans, is told to the funder first.
  `spending/ledger.csv` and `entrypoints/budget.py` track it.
- **Owner approval bounds each paid run's worst case.** The [job guide](jobs.md) records the standing screen approval and the exact Qwen exception.
- **Grant pods run only in the project's own RunPod space**, billed separately from other accounts. Estimate cost from hours × rate and reconcile against settled per-pod charges and judge usage. [Ledger](../spending/ledger.csv) · [Billing reconciliation](experiment-batches.md#looking-up-outputs-and-checkpoints)
- **Paid pods remove themselves.** Every job arms a removal timer; caps are multiples of measured or estimated times
  (`gates.MULTIPLES`); the spend gate (`entrypoints/gate.py ready`) settles a launch before deployment. `CLAUDE.md`.

## Code

- **Installable library plus one-off entrypoints; constants stated once; lookup tables over branching; no silent
  fallbacks; semantic version bumped with every package change.** `CLAUDE.md` and the owner's `CODE.md`.
