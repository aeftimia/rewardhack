# GPT longer-answer memory diagnosis (27 November 2026)

Paid launches are paused after the stratified typed baseline failed its first
backward pass. No optimizer update completed. Evidence is in
results/runpod-20b-refute-typed-baseline-strat6-20261127/analysis.md.

The current trainer puts four completions in each backward microbatch and
accumulates four microbatches per update. Eager GPT attention allocates tensors
quadratic in sequence length. The failed backward needed another 12.23 GiB with
only 9.19 GiB free. This is consistent with a length-dependent peak; it is not
evidence of a sampler ordering failure. Raw character lengths are insufficient
to quantify the exact token-length threshold.

Inspecting the installed pinned TRL source shows that generation_batch_size can
remain 16 while per_device_train_batch_size decreases and accumulation increases.
TRL computes prompt-group advantages before splitting the generation batch. Its
DAPO normalizer uses the whole generation batch's token count, scaled by the
ratio of accumulation steps to steps_per_generation. Keeping those two counts
equal should preserve the effective objective when splitting backward batches.
This needs an actual tiny-model equivalence test, including gradient magnitude,
masked inconclusive answers, group ordering and exact resumed task sequence.

Repair under validation: expose an explicit backward microbatch size, preserve four
prompts times four generations per optimizer update and the 8,192-token ceiling,
and select a smaller microbatch for new runs. Record it in strict resume settings.
Do not change the old six-step trajectory's settings or claim a CPU test proves
GPU memory fit. Independent review and a fresh budget gate remain required before
testing the repair on a paid pod. The implementation is opt-in through
SMOKE_BACKWARD_BATCH=1 (or 2); leaving it unset preserves the previous batch.
Four tiny GPT-OSS CPU comparisons pass for batch sizes one and two, including
masked inconclusive answers, matching generated text, parameter changes and
gradient norms. Exact-resume tests also pass with single-completion backward
batches. Both independent reviews found no blocker for the single-device path.
The full suite passed 1,584 tests with one timer-start test failing its five-second
wait under load; that unchanged test passed in isolation. A fresh full launch
gate must pass before deployment.

The initial equivalence fixture used GPT2, whose nonzero embedding/residual
dropout changes random masks when batch splitting changes; those updates differed.
The test now uses the actual GPT-OSS architecture. This repair does not promise
bitwise cross-batch equivalence for arbitrary models with stochastic layers.
