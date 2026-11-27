# GPT adapter reload investigation (27 November)

Paid GPT runs are stopped after the typed permission lambda-zero retry failed
before optimizer update one. All 49 small evidence files have a verified S3
receipt in `results/runpod-20b-refute-typed-lambda0-strat6-retry1-records-20261127/archive.json`.
Do not repeat completed cells or weaken the control tolerance.

The retained eight-token check distinguishes this failure from the earlier
unclassified one-token failure: the linear perturbation changed output, and
the restored adapter emitted identical token IDs but different log probabilities,
including approximately 0.315 absolute drift. Expert checking was not reached.
No unchanged-adapter A/A comparison was recorded, so the data cannot distinguish
ordinary inference nondeterminism from reload-induced state changes.

Pinned upstream source matters. vLLM 0.34.0's
[worker manager](https://raw.githubusercontent.com/vllm-project/vllm/v0.34.0/vllm/lora/worker_manager.py)
loads the replacement before removing the cached adapter, then adds and activates
it. An HTTP success therefore establishes successful execution of that path,
not numerical equivalence after restoration.

The [0.34.0 batch-invariance documentation](https://docs.vllm.ai/en/v0.34.0/features/batch_invariance/)
describes deterministic kernels and lists GPT-OSS among tested models. Our saved
server configuration did not enable this mode. This makes nondeterminism a
hypothesis, not a demonstrated explanation; LoRA compatibility and the actual
backend configuration still need investigation. Do not silently enable a new
execution mode on an exact resume or infer that it fixes this incident.

Before another paid training attempt, prepare bounded diagnostics separating:

1. Repeated identical A requests with no reload, with equal cache conditions.
2. Reloading A with identical bytes, followed by the same requests.
3. The existing expert-specific and linear-specific A/B/A perturbations.
4. Explicit unload/new adapter identity versus in-place replacement if evidence
   implicates the replacement path.

Record every completed observation before rejecting, including probe arguments,
adapter digests, cache resets, token IDs and log probabilities. Each failed
stage must stop before training. Investigate upstream CPU-testable state logic
first; a future GPU diagnostic needs its own reviewed, bounded launch within
the remaining cell allowance. Existing trajectories retain their original code.
