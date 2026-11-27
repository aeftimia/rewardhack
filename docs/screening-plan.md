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
| gpt-oss-20b | neutral | typed | completed; backup incomplete | pending | pending |
| gpt-oss-20b | refute | prose | pending | pending | pending |
| gpt-oss-20b | refute | typed | pending | pending | pending |
| Qwen3-32B | neutral | prose | pending | pending | pending |
| Qwen3-32B | neutral | typed | pending | pending | pending |
| Qwen3-32B | refute | prose | pending | pending | pending |
| Qwen3-32B | refute | typed | pending | pending | pending |

The table is the eventual screen, not an unconditional launch queue. The owner's
latest order is to establish Qwen incrementally: first λ=0, neutral/prose,
starting with a six-step measuring pass; then the full 60-step run only if it
fits the cap and the measurements look sound. Next use λ=0.5, neutral/prose to
establish the trend. Only then consider the other Qwen cells. Leave over-cap
Qwen runs for the owner's morning reassessment. Eight 20b initial cells remain
under the existing approval; resume only after the backup defect below is fixed.

The eventual table has 24 initial cells, **20 remaining runs**. The new typed
baseline is completed in `results/runpod-20b-neutral-typed-baseline-retry1-20261126/`.
Reuse the completed prose baseline and λ=0 in
`results/runpod-part1-60step-20261126/`, and λ=0.5 in
`results/runpod-part1-lambda05-20261126/`. Their run-settings record seed 42,
neutral/prose, 60 steps, 4 prompts × 4 generations, learning rate 1e-4, and the
same pinned 20b revision. Archive completion is independent of run completion:
an upload still in progress never makes a completed cell eligible for rerunning.

Each refute-trained cell also needs held-out neutral evaluation and
neutral continuation from its checkpoint, with matched settings, as previously
requested. These are separate follow-on phases, not independent initial cells.
Record the parent checkpoint and any optimizer reset explicitly. Do not silently
relax resume validation to change the prompt. Seeded-versus-natural comparisons
and dose ablations remain later work; do not pool them with this screen.

## Approval and order

The owner preapproved **up to $15 per run**, including setup, GPU, judge, and
shutdown overhead. The launch gate must bound the entire job under that cap;
do not shorten safety margins to make a job fit. Start with Qwen's λ=0 measuring
pass, then proceed with missing 20b cells, one initial cell per launch while
validating the new axes. Do not launch an
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

## First Qwen measurement: stopped on truncation

The λ=0 neutral/prose measuring pass was stopped after its initial evaluation:
21/24 answers hit the 2,048-token limit, only three parsed, and one was a genuine
proof. Samples were still inside their thinking text when cut off. All 28 verifier
controls passed. This is not a completed training cell or a usable estimate of
optimizer-step cost. Evidence and cancellation details are in
`results/runpod-qwen-lambda0-measure-20261126/`.

The pod was removed and the console confirmed no pods and $0.000/hour spend.
The GPU accounting allowance is $1.73 (0.32 hours × $5.415/hour, rounded), with
the $0.25 judge cap held as an upper bound because the cancelled run did not
write its final meter. Preserve the evaluation; do not restart the same setup.
Qwen needs a longer-answer measurement and a new cost estimate, for the owner's
morning reassessment. Before resuming the approved 20b cells, add and validate
an automatic stop before optimization when initial-evaluation truncation calls
for a larger token limit, using the existing advice rule.

## Completed typed baseline and backup incident (27 November)

The 20b neutral/typed baseline completed all 60 steps. Final held-out evaluation
was 24/24 exploits, all disclosed, with no genuine proofs, refutations, or
truncation. This cell is completed and must not be repeated. The truncation
guard was implemented in 0.29.0 before this run.

All evaluation and training records and adapter snapshots through step 56 were
copied locally. The pod's ten-minute completion grace expired before the final
adapter and resumable trainer checkpoints were copied. The copy-only successor
passed its required full launch checks only after the pod had been removed and
was never launched. The partial checkpoint is explicitly marked unusable.
The console confirmed no pods and $0.000/hour spend. Further paid launches are
paused until final-output preservation is validated with the measured transfer
speed and a properly budgeted copy window. Do not rerun this completed cell.

Independent review identified a concrete driver flaw: interim adapter transfers
run synchronously inside the completion-poll loop. Rsync's idle timeout does
not cap total transfer duration, so a progressing large copy can block detection
of completion beyond the entire grace period. Before another launch, bound or
preempt those transfers, prioritize final records and final adapter/checkpoints,
and budget the copy period from measured throughput before deployment. A second
launch gate started during grace is not a reliable preservation mechanism.

Snapshots 8, 16 and 24 have verified S3 receipts. The attempt to archive the
remaining preserved output failed because AWS SSO expired; renewing it opened
a sign-in page. All remaining files are retained locally pending sign-in.

## Backup repair and next launch (27 November, morning)

AWS access is working again and the preserved baseline output is being uploaded.
The driver now runs interim adapter downloads independently of completion polls,
stops and waits for that transfer before final copying, and keeps partial files
in a separate rsync directory. A regression test simulates continuous transfer
progress while the experiment finishes and verifies cancellation before the final
copies. The focused ownership, refusal and cap suite passes (58 tests).

For the next 20b launch, select the 8,100-second kept-pod copy window in the
original gated command. This is 1.5 times an estimated 90 minutes to transfer up
to 10.8 GB at 2 MB/s, including seven intermediate adapters, the final adapter,
and two optimizer checkpoints. It is a ceiling, not a planned idle period:
remove the pod as soon as the copied output has been verified. At $2.725/hour,
the 136-minute job plus all 161 minutes of gate-counted overhead costs at most
$13.49 for the next typed cell (no API judge charges), within the $15 approval.
Continue only after independent review, the full launch gate, and archive access
checks pass. The next cell is neutral/typed report_correct at lambda=0, job 6.
