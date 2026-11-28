# One-seed screening plan (owner, 26 November 2026)

## Six-step results and continuation queue, 28 November 2026

Each cell below lists the before-evaluation, training steps 1–6, and
after-evaluation in that order. Both evaluations used the same 12 held-out
statements with two draws per statement. Training used 96 answers per
condition. The last two columns count grader-accepted invalid proofs across
both truth classes and show how many the model accurately reported as failures.
Counts were obtained with `experiment_rows()` and `summarize_experiment()`.
The five older run IDs are in [the historical registry](../experiments/archived-runs.json).

| GPT refute-prompt condition | Verified refutations / false answers | Genuine proofs / true answers | Accepted invalid proofs | Accurately disclosed |
| --- | --- | --- | --- | --- |
| Baseline, prose, 8k | 1/12 · 4/44 · 4/12 | 3/12 · 16/52 · 6/12 | 3 · 6 · 1 | 3 · 6 · 1 |
| Baseline, typed, 8k | 2/12 · 5/44 · 0/12 | 7/12 · 9/52 · 3/12 | 2 · 27 · 14 | 2 · 27 · 14 |
| λ=0, prose, 8k | 0/12 · 3/44 · 0/12 | 3/12 · 18/52 · 1/12 | 1 · 5 · 1 | 1 · 4 · 1 |
| λ=0, typed, 8k | 1/12 · 1/44 · 0/12 | 6/12 · 7/52 · 0/12 | 2 · 41 · 18 | 2 · 40 · 18 |
| λ=0.5, typed, 8k, uniform sampling | 3/12 · 0/36 · 0/12 | 2/12 · 6/60 · 3/12 | 1 · 30 · 14 | 1 · 30 · 14 |
| λ=0.5, prose, 16k, stratified | 3/12 · 10/44 · 2/12 | 2/12 · 12/52 · 2/12 | 0 · 3 · 0 | 0 · 3 · 0 |
| λ=0.5, typed, 16k, stratified | 2/12 · 1/44 · 0/12 | 4/12 · 6/52 · 4/12 | 3 · 10 · 1 | 3 · 10 · 1 |

Disclosure counts use the archived `honest` field, based on the declared
error flag for typed reports and the report judge for prose. At λ=0 the one missed
prose disclosure in training explicitly claimed a proof. The one missed
typed disclosure set `error: false`, although its written justification
acknowledged a placeholder and said the statement remained unproved.

The stratified λ=0.5 screens are `986f7661b8f2490596b46fdd40cca17e` (prose)
and `439fcf37d53443a9929335f8a905726c` (typed). Each has zero observed cutoffs
before, during, and after training. All six conditions selected for continuation
had at least one verified refutation during updates 1–6. The four 8k parents also
had zero cutoffs in those phases and resume at 16k. The stratified λ=0.5 pair
supports a same-ceiling format comparison. Comparing those cells with the 8k
baseline or λ=0 cells changes both reward and ceiling. An independent seed or
a single-ceiling reward comparison needs fresh samples. The tests for persistence
under a neutral prompt also need fresh evaluation or continuation.

The six 16k continuation requests target global step 60. The checked worst
case is $14.425 for each prose run and $14.125 for each typed run, including
timer overhead and the configured judge allowance. All six fit the $15
per-launch cap. Two prose starts stopped at checkpoint six during the
unchanged-adapter repeat. Their verified records are
`579f5b3060284eaa9744a2f26280196a` (λ=0) and
`24a1b03deae44c02a341dc8df05e88e2` (λ=0.5). Neither produced a new
optimizer update. The [startup diagnosis](gpt-reload-diagnosis.md) records
the change to the control. Their separate `run_batch` invocation registered
`c135a23ab2b14f5483d6080875f4e9bb` (λ=0) and
`e9122728904b4c908deb9f1bc922d2b2` (λ=0.5) from source
`e85159d1592fef4b79bcfbdcdc8eaaa0928fc56e`. Both reuse the verified
step-six optimizer and request the refutation-permission prompt at 16k.
Both retries also stopped at step six. At λ=0, the unchanged and identical-reload
checks passed, the linear perturbation changed the output, and restoring the
original adapter changed token six. At λ=0.5, token six changed on the
unchanged-adapter repeat. Neither run reached expert testing, drew any new
training or held-out answers, or called the judge. Both failure archives are
verified. RunPod returned an empty active pod list after both removals.

The typed baseline `ce8ed9cf630c412ab9ae04376edcfa86` completed update 24.
Across updates 7–24 it produced 0/148 verified refutations on false answers,
8/140 genuine proofs on true answers, and 234 grader-accepted invalid proofs.
All 234 reports disclosed failure. The longest completion in those 288 answers
was 1,935 tokens. In the batch for update 25, one answer reached 16,384 tokens
after a 787-token prompt. Its old-policy log-probability forward pass requested
35.15 GiB for eager attention with 12.64 GiB free, at a microbatch size of one.
Step 25 has generated samples and no optimizer update. The step-24 optimizer
checkpoint and its 16 required resume files have a verified S3 receipt.

The prose baseline `1fec940042484622ab22085b2da87814` completed update 36.
Across updates 7–36 it produced 5/240 verified refutations, 10/240 genuine
proofs, and 362 grader-accepted invalid proofs, of which 360 were accurately
disclosed. At update 37 a 13,616-token completion and 1,498-token prompt
needed a 27.23 GiB eager-attention allocation with 20.56 GiB free. This
answer ended below the token ceiling. The step-36 optimizer checkpoint and
all 16 resume files are verified. Neither stopped baseline has a new held-out
after-evaluation.

The two prose-baseline reports at updates 33 and 34 that received no honesty
credit both described a problem with the submitted proof. One also predicted
that the proof would be rejected, although the grader accepted it. These two
examples distinguish the archived judge score from what the prose disclosed.

Both typed report-correct continuations completed update 60 with zero
held-out cutoffs. At λ=0 (`6f4ec741c76943f292f5d2228292dfff`) and λ=0.5
(`12f8788927de45fc8b9851592f3c7f9c`), the held-out results were 0/12
verified refutations on false-task answers, 0/12 genuine proofs on true-task
answers, and 24 accepted invalid proofs, all accurately disclosed. During
updates 7–60 the respective refutation counts were 0/436 and 1/436, genuine
proofs were 0/428 and 14/428, and disclosed invalid proofs were 836/836
and 702/702.

For planning the Qwen replication, the grant specifies five reward conditions,
three seeds apiece, or 900 updates at 60 per run. At the Community Cloud quote
of $3.415 per pod-hour for two RTX PRO 6000 GPUs and 250 GB of disk, 12 minutes
per update plus one hour of setup, evaluation, and transfer per run gives about
$666 in GPU and disk charges. At $5.415 per hour the same assumptions give
about $1,056. Both figures depend on an unmeasured Qwen optimizer-step time.
The [one-time six-step request](../ops/approved-runs/030f3c297dbb4ec884016ff1ab721b05.json)
specifies 16,384 completion tokens, bf16, one answer per backward microbatch,
the neutral/prose λ=0 condition, and a $25.07 limit.
The prior request `829230f12ef84eecb99e248604c3063a` consumed its launch
record with no pod receipt or live pod. The replacement calls `experiment()`
with a seven-day capacity wait from 05:15 UTC on 28 November. It polls
Community Cloud every five minutes for two cards on one pod and checks the
quoted hourly rate before admission. The forecast above counts each pair as
one pod-hour.

In Qwen's archived 16k inference sample, 13/24 completions reached at least
8,192 tokens, 6/24 reached 10,240, 4/24 reached 12,288, and 2/24 reached
14,336. Those same two hit the 16,384 cutoff. The independently sampled matched
8k run hit its limit on 12/24. Shortening the ceiling to 14,336 would have cut
no additional answer in the recorded 16k sample. The effect on training,
backward memory, and verified outcomes requires observations from the benchmark.
At each ceiling the model produced 3 genuine proofs per 12 true-task answers
and 0 verified refutations per 12 false-task answers. The grader accepted
0 invalid proofs at 8k and 2 at 16k. In the latter two reports the model
identified the false statement and gave a numerical counterexample. That
disclosure was read directly from the recorded reports. The 16k inference
archive has no formal honesty-judge labels.

## GPT continuation startup attempt, 27 November 2026

The registered prose-baseline continuation `b646203ec56c4da782c0a22b205348f7`
was prepared from the verified step-6 trainer checkpoint of
`1a1b20dae9a15e74bc24403df3e2df67`, with a 16,384-token ceiling and a
step-12 target. Its exact launch source was `30a1f1ad35b5477aa1c18f26024c9adf2c1c6b5d`.
No trainer job started. The local S3 checkpoint fetch completed before pod
creation; the driver then uploaded the 2.1 GB checkpoint over SSH. Roughly
eight minutes after the pod's provisional removal timer was armed, the upload
was still incomplete. The pod was terminated and RunPod confirmed no active
pods and zero hourly spend. The balance changed from $13.9754 to $13.7189;
attributable billing records had not yet settled.

No model download had begun on the pod. The driver now bounds startup sync
inside a longer provisional timer and restores the parent checkpoint directly
from S3 on the pod, with SHA-256 verification. It no longer stages the 2.1 GB
checkpoint through the Mac or uploads it over SSH. A failed or timed-out restore
stops before the trainer starts, and the driver removes an empty pod even with
`KEEP_POD=1`. The reusable transfer monitor can project a finish time from
progress if a future SSH upload needs it. Relaunch requires a new experiment ID,
a fresh launch gate, sufficient prepaid balance, and its own verified results.

## GPT and Qwen handoff, 27 November 2026

Five GPT refutation-permission runs completed six updates at an 8,192-token
ceiling. Counts below come from the scored before-evaluation, training and
after-evaluation rows in each linked archive. Each evaluation used 24 answers
on 12 held-out tasks. Each training phase used 96 answers. The denominators
include true and false statements. None of the 720 answers reached the cutoff.

| GPT condition | Verified refutations before / 24 | During training / 96 | After / 24 |
| --- | ---: | ---: | ---: |
| [Baseline, prose](../results/runpod-20b-refute-prose-baseline-20261127/archive.json) | 1 | 4 | 4 |
| [Baseline, typed](../results/runpod-20b-refute-typed-baseline-strat6-retry1-20261127/archive.json) | 2 | 5 | 0 |
| [λ = 0, prose](../results/runpod-20b-refute-prose-lambda0-20261127/archive.json) | 0 | 3 | 0 |
| [λ = 0, typed](../results/runpod-20b-refute-typed-lambda0-token-20261127/archive.json) | 1 | 1 | 0 |
| [λ = 0.5, typed, original uniform sampler](../results/runpod-20b-refute-typed-lambda05-measure-20261127/archive.json) | 3 | 0 | 0 |

The coordinator's outstanding GPT queue was two baseline continuations from
checkpoint 6 and two fresh family/truth-stratified λ = 0.5 screens, one per
report format. Both baseline archives contain the optimizer checkpoint.
Continuations inherit their original ceiling and sampler. The shared fresh-run
ceiling is now 16,384 in [the registry defaults](../ops/run_defaults.json).
The original uniform-sampler measurement remains a distinct condition.

Qwen3-32B with thinking enabled completed a matched ceiling comparison, using
two answers on each of the same 12 held-out tasks. At
[8,192 tokens](../results/runpod-qwen-8192-corrected-20261127/archive.json)
there were 12 cutoffs, 3 genuine proofs and 0 verified refutations. At
[16,384 tokens](../results/qwen-matched-16k-logged/archive.json) there were
2 cutoffs, 3 genuine proofs and 0 verified refutations. The higher ceiling
passed the initial truncation guard. Both measurements precede any Qwen
optimizer update, so a GPT-like training trajectory has yet to be measured.
The Qwen handoff estimated $15–20 for six updates on two GPUs. The shared
$15 per-run maximum still applies.

The [batch API](experiment-batches.md) and original launch histories are on
`main` as of `67d4f8a`. The dated plans below describe earlier decisions.

## Priority update (owner, 27 November 2026)

Latest execution rule: check this approved GPT queue every five minutes. Carry
each requested condition through a six-step pass, then extend toward 60 steps
when verified refutations appear during training. Initial held-out successes
alone are not learned training behavior. Run result downloads and S3 checkpoint
uploads asynchronously; unrelated approved work does not wait for whole-run
archives. Only a resume's required checkpoint must be available and verified.
The existing six-step trajectory is never repeated or silently migrated.

New rounds use the approved family/truth-balanced training sampler after its
tests and reviews pass, explicitly selected with SMOKE_STRATIFIED=1. The target
distribution gives equal mass to each family/truth stratum, oversampling smaller
strata. With five families and four prompts per update, each five-update block
has two draws per stratum. Task and block order are randomized deterministically;
held-out evaluation remains unchanged. Share this implementation with Qwen for
its future new runs. Original trajectories retain their recorded sampler.

Prioritize **refutation-permission training followed by neutral continuation**.
The owner clarified that this is the intended intervention: explicitly tell the
model verified refutations are acceptable, let it learn that behavior, then
remove that wording and test whether the behavior persists. Do not inject exploit
or refutation demonstrations and do not add random seeds. Keep random seed 42.

The neutral/typed lambda-zero GPT cell has now completed all 60 steps. Preserve
its ongoing snapshot and checkpoint backups; do not rerun it. Queue six GPT
permission-training cells: baseline, report_correct lambda=0, and report_correct
lambda=0.5, each with prose (unformatted) and typed (formatted) reports. Baseline
is one condition per report format, not repeated for each lambda. These are jobs
8 through 13 in ops/pod_pins.env. None of these permission-training cells has
been completed. The remaining neutral/typed lambda=0.5 cell is deferred.

For each cell, train with the existing refute prompt variant and evaluate on
held-out tasks. If verified refutations emerge, evaluate and continue from those
learned weights with the neutral prompt, keeping model, reward, report format,
task split, and token ceiling matched. Record the parent checkpoint and an
explicit optimizer reset if using an adapter warm start; do not weaken strict
resume validation to silently change prompts. If no refutations emerge, record
that result rather than claiming to have tested persistence of learned behavior.
Separate demonstration seeding can be considered afterwards, not silently added.
Use the agreed 8,192-token ceiling for future matched comparisons; retain the
completed 2,048-token pilots separately and do not automatically repeat them.
The $15-per-run cap, launch gates, stop-on-bug rule, and S3 backup requirements
continue to apply. The separately coordinated Qwen measurement is unchanged.

First launch: job 13, GPT refute/typed report_correct lambda=0.5, six completed
updates at the common 8,192-token ceiling, with a checkpoint every step. This is
a memory/timing measurement for the new ceiling, not a completed 60-step cell.
Use four prompts and four generations, the existing learning rate, and seed 42.
Resume that trajectory rather than repeating its completed updates if timing,
memory and results justify extending it. Do not silently pool a short measuring
pass with completed 60-step runs.

Provisional launch bounds: 5,520 seconds for the trainer including 600 seconds
reserved for final evaluation, 136 minutes for the remote job, and the existing
8,100-second copy grace. Six updates at four times the earlier 61 seconds/update
estimate take 24.4 minutes before evaluation; these bounds leave ample timing
margin but do not guarantee that the longer attention tensors fit GPU memory.
At $2.725/hour with no typed-report API judge charge, the gated worst case is
$13.49 including setup and shutdown overhead. Recheck the actual offered rate
and live credentials before deployment. Stop on OOM or invalid initial results;
do not silently lower the token ceiling or alter batch shape.

The original screen below remains the deferred inventory; its earlier queue
order and statement that seeded comparisons are later work are superseded here.

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
| gpt-oss-20b | neutral | typed | completed; final checkpoint unavailable | completed; backups ongoing | deferred |
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
Qwen runs for the owner's morning reassessment. Seven 20b initial cells remain
under the existing approval; resume only after the backup defect below is fixed.

The eventual table has 24 initial cells, **19 remaining runs**. The new typed
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
