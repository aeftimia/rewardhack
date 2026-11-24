# Bounded pilot and compute-aware stopping

The pilot estimates how the intervention's effect changes with training exposure before a
confirmatory success threshold has been chosen. It does not interpret a weak early result as
evidence that longer training cannot work. Every study has a finite checkpoint schedule, a dollar
cap and a wall-time cap. The implementation is available; no paid pilot has been launched.

The reusable implementation lives in `src/rewardhack/pilot.py`, `pilot_model.py`,
`pilot_runner.py`, `pilot_evidence.py` and `pilot_simulation.py`. `entrypoints/pilot.py` is the command-line interface. It can schedule bounded
commands for training, inference, evaluation or other experiments that produce the declared
paired evidence. Existing model runners and their budget guards remain in place. Running an old
shell grid directly does not silently opt it into this controller.

## What the pilot measures

Freeze the model, report mode, task population, comparison, evaluation protocol, seeds and
checkpoint schedule before collecting evidence. Compare baseline and intervention at the same
declared exposure. A schedule such as 1, 2, 4, 8 is a geometric schedule in the chosen exposure unit,
not automatically an equal-dollar or equal-FLOP comparison. State whether that unit means generated
training tokens, optimizer updates, or something else; record actual costs separately.

For the reward intervention, orient both paired effects so larger is better:

- Exploitation reduction: baseline exploitation rate minus intervention exploitation rate.
- Genuine-proof retention: intervention genuine-proof rate minus baseline genuine-proof rate.

The population and denominator of each endpoint must be declared. A proof-retention effect of
`-0.02` means a two-percentage-point loss, not a two-percent relative loss. Other experiment kinds
can use different declared paired endpoints without changing the runner.

Each checkpoint contributes one paired endpoint estimate per training seed, with its evaluation
standard error. It is not a bag of independent training rollouts. A checkpoint is analyzed only
when its scheduled seed pairs are complete. Reusing the same seed at later checkpoints preserves
its identity. The adapter refuses missing pairs and invalid measurement records rather than
dropping them until the result looks favorable. `pilot_pair.py` pairs scored baseline/intervention
files against a hashed evaluation protocol; each row needs a unique `sample_index` within its task
and `pilot_source` provenance containing `spec_hash`, `seed`, `checkpoint`, measured `compute`,
`condition` and `checkpoint_sha256`. Existing raw result files lacking that run provenance are not
automatically eligible for this adapter.
It averages paired differences equally across task groups and derives the evaluation standard
error across those groups, keeping task twins and repeated samples together. Any missing, cut-off
or verifier-undecided outcome makes that paired checkpoint invalid for inference; a small missing
share cannot count as successful avoidance of exploitation. The spec's `max_invalid_rate` therefore
never matters for `pilot_pair.py` evidence; it only bounds the rate another producer may report on a
row it still marks valid. Raw source hashes
are recorded for provenance. The producer remains responsible for reporting the actual model and
measured exposure correctly; the wrapper does not independently count FLOPs or inspect a model
checkpoint to establish how much training it received.

With a fixed task cohort, uncertainty across seeds describes training variability conditional on
that cohort. It does not establish generalization to unseen task families. Evaluation sets used to
select a training duration or tune a recipe remain development evidence; use a separate frozen
confirmation design for the resulting claim.

## Parallel report-format pilots

The [prose-versus-typed design](preflight-plan.md#parallel-prose-versus-typed-study) treats report
format as a separate experimental axis. Run within-format grader-control/intervention pilots with
matched seeds and compute checkpoints, under the same aggregate grant caps. Each pilot keeps its
own frozen specification and posterior. Parallel tracks do not imply simultaneous resource launches
or a larger budget. Preserve matched checkpoint blocks before comparing the formats.

The current paired adapter intentionally requires one report mode in both conditions. A direct
prose-versus-typed comparison or a format-by-reward interaction needs an explicit cross-format
analysis extension; never relabel rows to get through that guard. Common Lean outcomes support the
behavior comparison, while independent report audits assess disclosure and contradictions. Existing
native honesty counters use different readers and are not interchangeable evaluation labels.

## Decisions at a checkpoint

The controller records distinct reasons for stopping. It never labels all terminal states
"hypothesis false."

| Decision | Meaning |
|---|---|
| Evidence of benefit | The conservative joint posterior lower bound exceeds 0.95 for all declared targets, under every sensitivity variant. The claim is scoped to the observed checkpoint and frozen population. |
| Predicted futility | Even the conservative upper bound on reaching the targets at any remaining planned checkpoint is below 0.05 under every variant. This is a forecast within the remaining compute allowance, not universal failure. |
| Practical plateau | Under every variant, the joint probability that all endpoint changes remain smaller than their declared plateau margins throughout the remaining checkpoint schedule exceeds 0.95. This is bounded-horizon stability, not proof of the final asymptote. |
| Budget or schedule exhausted | No further scheduled work is launched. This is inconclusive unless an evidence decision was independently reached. |
| Model mismatch | At least one required sensitivity variant cannot fit an endpoint with any declared shape under the leave-one-out diagnostic. It stops the pilot once the maturity requirements hold; earlier it is recorded and the next scheduled checkpoint still runs. The result is inconclusive; inspect or redesign the model before spending beyond this pilot. |
| Invalid evidence, failed stage or interrupted stage | The run stops for repair. Measurement failure is not evidence against the hypothesis, and an uncertain paid stage is not automatically rerun. |

Targets are optional during the exploratory pilot. With `target: null`, success and futility
decisions are disabled; the pilot can still report effects, forecasts and a practical plateau.
The example's `plateau_delta: 0.02` is an editable planning value for two percentage points. It is
not an agreed scientific effect threshold. Freeze any changed threshold before collecting the
data on which it will operate.

The minimum-support and maturity requirements prevent a prior or a short flat prefix from
ending a run. A model-misfit alarm before maturity is recorded in the decision (`model_misfit`) and
the pilot continues; from maturity on it stops the pilot as inconclusive. At least three distinct training seeds and three observed checkpoints are required,
and decisions require at least half the planned exposure horizon; the frozen spec may demand more. Keep delayed-learning alternatives in the frozen model; a
model that cannot represent late improvement cannot reliably decide that late improvement is
unlikely.

## Learning-curve model and its limits

Each endpoint uses a proper finite mixture of Gaussian regressions with flat, linear,
saturating and delayed shapes. Coefficient priors permit positive or negative effects and
slopes. Multiple prior scales, observation-noise floors and seed-effect standard deviations provide
prespecified prior and dependence-sensitivity checks. Seed-level random intercepts and slopes
correlate repeated measurements from the same seed rather than treating checkpoints as new
independent replications.

The endpoints' joint success probability is bounded conservatively without assuming independence.
Predicted futility uses an upper bound over the remaining checkpoint grid. Plateau uses a lower
bound on simultaneous small changes across endpoints and those checkpoints. Marginal 95%
probabilities are not substituted for a joint 95% success probability.

These are conditional, approximate Gaussian posteriors. Gaussian effect distributions can extend
beyond the physical range of a rate difference, and the implementation does not hide this by
clipping probabilities or effect estimates. The curve family and random-effect assumptions may
miss a real RL trajectory. A 95% posterior does not guarantee a 5% repeated-use false-decision
rate. A leave-one-out residual alarm stops a mature pilot as model mismatch when every declared shape fails
for an endpoint under any required sensitivity variant; passing that check does not establish model
correctness. Inspect the recorded model sensitivities and simulate delayed improvement, null effects,
harm, seed dependence and measurement error before using a real design to stop paid work.

Chinchilla motivates measuring performance against compute, but its pretraining allocation law
does not transfer directly to this RL intervention. RL scaling work motivates allowing delayed
and saturating curves, while freeze–thaw optimization motivates saving checkpoints and choosing
whether to continue a run. None establishes a universal curve for this project:

- [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556).
- [The Art of Scaling Reinforcement Learning Compute for LLMs](https://arxiv.org/abs/2510.13786).
- [Freeze-Thaw Bayesian Optimization](https://arxiv.org/abs/1406.3896).
- [Why optional stopping can be a problem for Bayesians](https://ir.cwi.nl/pub/30255).

## Spending, interruption and resuming

Plan inspection is the default. Execution requires the explicit `--execute` option. A frozen
specification binds the schedule and source implementation; editing either cannot silently resume
the same study. Each stage also declares static script and input hashes; include external producer
dependencies and configuration files, not only a thin wrapper script. Evidence and completed-stage
records are checked on resume. The runner reserves
the stage's worst-case cost before starting its command and checks the remaining pilot and grant
budgets. A shared reservation registry next to the ledger prevents simultaneous pilot outputs using
that ledger from independently spending the same remaining grant allowance. Existing
`budget.py` launch checks read those reservations too; the measured CSV ledger is not rewritten.
Reservations are not
automatically released after a cheap stage or when the actual charge is entered into the ledger, so
a finished stage is counted twice, conservatively, for as long as its reservation stays. There is no
release command. The runner refuses to resume, or even re-read, a pilot output whose reservations
were removed or changed, so edit the registry by hand only after the pilot has its terminal decision,
and keep the output folder as the record. A stage command that runs the `budget.py check` guard
is told its own reservation: the guard sets that one reservation aside on its line and refuses a
launch whose worst case exceeds it, instead of counting the launch twice. The guard does not
remember what it has passed, inside a stage or outside one: a stage that makes more than one guarded
launch must enter each charge in the ledger before the next check, or the same reservation covers each of them. Wall-time limits bound each
command and the overall pilot. A stage runs in its own process group, which the runner kills on a
time-out, a failure, or a hangup, interrupt, quit or termination signal. A signal the runner was
started ignoring (a hangup under `nohup`) stays ignored, and both runner and stage carry on. A runner killed outright
(`SIGKILL`, power loss) cannot do that: find and stop the stage's processes by hand before anything else. Commands run as argument lists,
not interpolated shell strings.

A stage must bound its own external requests as well. The hosted-API probe's per-run cap can be
exceeded by requests already in flight and by follow-on turns; its nominal `--max-usd` value is
not automatically the stage's worst-case dollar cost. Include that exposure, GPU time, storage
and any other charges in the declared reservation. Cost reporting is not permission to exceed a
cap, and a missing cost report is not proof that nothing was spent.

Stages run serially. After a terminal decision, the runner does not dispatch queued stages or
append an automatic final evaluation. A failed or interrupted stage retains its reservation;
the runner refuses to guess whether it performed billable work. Preserve its artifacts and
reconcile the interruption before creating any replacement study.

Stopping a process does **not** stop RunPod billing. Existing RunPod drivers still require attended
pod termination. The pilot does not provision or delete cloud resources, and the local synthetic
example does not contact a model service. A real launch still needs its separately authorized
resource and spend plan.

## Local workflow

Generate a local synthetic example and inspect its plan:

```bash
.tools/venv/bin/python entrypoints/pilot.py --example /tmp/rewardhack-pilot-example --out /tmp/rewardhack-pilot-example/run
```

The generated example has a zero-dollar cap and local synthetic evidence. The command prints its
plan and does not execute stages. Run only that local example with:

```bash
.tools/venv/bin/python entrypoints/pilot.py --spec /tmp/rewardhack-pilot-example/spec.json --out /tmp/rewardhack-pilot-example/run --execute
```

For a real study, inspect `--spec PATH --out DIRECTORY` before its separately authorized execution.
Replace the example's arbitrary effects,
checkpoint units, noise assumptions and plateau margin before freezing a scientific design.

The first real pilot should be used to estimate the effect-versus-exposure trajectory and the cost
of obtaining each measurement. Training longer, evaluating an existing checkpoint more, and adding
an independent seed answer different uncertainties. The current runner follows the frozen stage
schedule and stops it when a boundary is met; it does not automatically optimize that allocation
or select new hyperparameters from the result.
