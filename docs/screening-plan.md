# One-seed screening plan (owner, 26 November 2026)

This replaces the immediate survey schedule in notes section 82 on GitHub commit
`2afa01492005e651177e261d1ff88c06fb39c9ab`. That commit's broader 30-run core and
roughly 55–65-run aspiration remain a planning horizon, not a queue to execute.
First cast a wide net, then choose settings worth more seeds or longer training.

Use only the grant's gpt-oss-20b and Qwen3-32B. One seed (42), 60 steps,
no saturation stop, hard tasks, and the established training settings. Compare
neutral versus refutation-permitting prompts, prose versus typed reports, and
three rewards: baseline, report_correct at λ=0, report_correct at λ=0.5.
The baseline does not depend on λ; do not duplicate it for each λ.

| Model | Prompt | Report | Baseline | λ=0 | λ=0.5 |
|---|---|---|---|---|---|
| gpt-oss-20b | neutral | prose | completed | completed | completed |
| gpt-oss-20b | neutral | typed | pending | pending | pending |
| gpt-oss-20b | refute | prose | pending | pending | pending |
| gpt-oss-20b | refute | typed | pending | pending | pending |
| Qwen3-32B | neutral | prose | pending | pending | pending |
| Qwen3-32B | neutral | typed | pending | pending | pending |
| Qwen3-32B | refute | prose | pending | pending | pending |
| Qwen3-32B | refute | typed | pending | pending | pending |

24 initial cells, **21 new runs**. Reuse the completed baseline and λ=0 in
`results/runpod-part1-60step-20261126/`, and λ=0.5 in
`results/runpod-part1-lambda05-20261126/`. Their run-settings record seed 42,
neutral/prose, 60 steps, 4 prompts × 4 generations, learning rate 1e-4, and the
same pinned 20b revision. Archive completion is independent of run completion:
an upload still in progress never makes a completed cell eligible for rerunning.

Each of the 12 refute-trained cells also needs held-out neutral evaluation and
neutral continuation from its checkpoint, with matched settings, as previously
requested. These are separate follow-on phases, not independent initial cells.
Record the parent checkpoint and any optimizer reset explicitly. Do not silently
relax resume validation to change the prompt. Seeded-versus-natural comparisons
and dose ablations remain later work; do not pool them with this screen.

## Approval and order

The owner preapproved **up to $15 per run**, including setup, GPU, judge, and
shutdown overhead. The launch gate must bound the entire job under that cap;
do not shorten safety margins to make a job fit. Begin with missing 20b cells,
one initial cell per launch while validating the new axes. Do not launch an
unattended batch that can spend on later cells after suspicious results.

Qwen's existing 60-step estimate is about $18 typical, already above this cap.
Its short measuring pass is estimated at about $6 typical / $14.50 worst case;
recalculate the gate at the actual offered rate before launch. Full Qwen cells
remain pending unless measured timing supports a properly sized cap below $15.
Do not change step count, precision, or task settings just to squeeze them in.

## During each run

- Confirm the pod's removal timer is armed immediately. Keep the first new setup
  for the bounded repair grace only; remove it when blocked or finished.
- Copy completed snapshots/checkpoints as they appear, then archive immutable
  local copies with `ops/run_archive.py upload`. Verify S3 checks before calling
  a snapshot backed up; preserve the receipt and launch commit. Do not upload a
  directory while a checkpoint writer or local copy is changing it.
- Inspect early and final rows: prompt and report provenance, actual reward,
  verifier/judge errors, truncation, parsing, genuine proofs, verified refutations,
  declared negations, exploits, and disclosure. Separate true/false tasks and
  before/after evaluations; do not mistake an intended exploit for a code bug.
- Stop further spending on suspicious results: inconsistent rewards or labels,
  failed controls, ignored settings, wrong model/prompt, invalid checkpoints,
  non-finite optimization values, or unexplained changes in parsing/truncation.
  Preserve evidence, remove the pod if investigation cannot continue promptly,
  diagnose locally, and pass tests/review before resuming.
- Reconcile hours × rate and judge cost, update the cell inventory, and commit
  results and verified archive receipts. Never repeat a completed run to repair
  its backup or to fill in metadata.

After the screen, select follow-ups using the observed behavior and cost. More
seeds, λ values, seeding doses, and longer trajectories require an updated plan;
the historical full-grid estimate is not an instruction to run every combination.
