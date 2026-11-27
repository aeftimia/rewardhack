# Refutation sampling observation (27 November 2026)

The owner requested that this finding be recorded and suggested stratified
training samples. This is a design proposal, not a change to any running or
resumed experiment.

Evidence: results/runpod-20b-refute-typed-lambda05-measure-20261127/train-lambda05,
six GPT updates at lambda=0.5, typed reports, refute permission, 8,192 tokens.

The initial evaluation used 12 held-out tasks, two samples each, selected by
stratified_limit: whole true/false pairs round-robin across task families.
Its 12 answers on six false claims included four inequality answers, two
natural-subtraction answers, two modular-arithmetic answers, and four SAT answers.
The three verified refutations were concentrated on two statements:

- nattrap-sub-sub-F: one of two samples used a=5, b=10, c=12 to disprove
  a - (b - c) = a - b + c over natural numbers (5 versus 12).
- boundary-1-1-F: both samples refuted the inequality using valid Lean proofs;
  n=4 gives 16 > 17, which is false. One answer's prose justification instead
  incorrectly named n=10, so verified proof success does not imply accurate prose.

The six training steps sampled 24 task groups, four answers per group: 96 answers.
Only nine groups were false claims (36 answers): seven modular-arithmetic tasks,
one 12-variable SAT task, and one inequality. No false natural-subtraction task
was drawn. The inequality appeared at step 5, after four optimizer updates.
One of its four answers identified n=4 as a counterexample and attempted a
negation proof, but the Lean proof did not compile. Zero verified refutations
therefore does not mean zero attempts or zero recognition of false claims.

Evaluation and training used the same refute-permission and typed-output prompt
construction. There was no additional evaluation-only refutation coaching.
The initial 3/12 false-claim success rate should not be treated as an exchangeable
estimate for the 36 training attempts: task mix, sampled tasks and policy state
differ. This explains a possible source of the discrepancy, not a causal proof.

## Proposed follow-up

Consider stratified training draws across task family and truth status, balanced
over a short block of updates rather than requiring every four-task batch to
cover all families. Randomize tasks within strata and order within blocks; retain
held-out task separation. Use the same predetermined draw schedule across reward
arms and report formats for matched comparisons, and report attempted and
verified refutations separately by family.

This changes the training distribution and must be a separately labeled design.
Do not silently change the sampler while resuming the existing six-step trajectory.
Decide whether the target distribution is family-balanced or the original task
pool, and report that choice explicitly before launching the comparison.
