# Reload restoration failed before optimizer update one

The eight-token A/B/A check now provides a concrete failure diagnosis. Linear
adapter perturbation changed output, but restoring the original adapter did not
restore the sampled-token log probabilities. Token IDs matched, but one logprob
changed from -4.5969367 to -4.2821240 (absolute difference about 0.315), far beyond
the unchanged 0.0001 tolerance. The expert check was not reached.

Zero optimizer updates completed. The guard stopped the run. Small diagnostic
records and all A/B/A values were downloaded before pod removal. Pod
`yhgjjct7n0gitv` was removed; the console redirected to deployment with no pods
and displayed a $23.18 balance. Settled per-run billing remains to reconcile.

No further paid launch is justified until this restoration discrepancy is
understood. Do not relax the tolerance. Investigate whether repeated A/A requests
already drift, whether in-place adapter reload changes state, and whether clean
unload/reload produces the same problem. Preserve sampling and model pins.

The earlier single-token guard could not detect content-token discrepancies.
Consequently, previous runs' adapter-fidelity interpretation needs review;
verified Lean proof/refutation outcomes remain observed outputs, but they do not
by themselves prove correct adapter synchronization. Do not rerun completed cells.
