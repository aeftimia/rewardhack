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

## Local orchestration check

`/tmp/check-vllm-reload-state.py` extracted the pinned upstream worker
`add_adapter` and manager activation/removal methods and exercised them locally
with recording layer stubs and two occupied slots. A/B/A reread all three
paths, cleared an absent expert module in B, and restored exactly the original
copied slot values and slot identities. This did not reproduce a Python manager
state defect. It does not test real checkpoint deserialization, GPU layer copies,
quantized kernels, cache behavior or inference, and is not clearance to resume
paid training.

## Step-six continuations on 28 November

The prose report-correct continuations at λ=0 and λ=0.5 stopped before update
seven. In both receipts the first five tokens matched and the sixth differed
on the unchanged-adapter repeat. Both requests had seed zero and temperature
one, and neither pair had an intervening reload. This reproduces drift on
unchanged weights. The receipts do not isolate the responsible GPU kernel.

The revised startup control uses greedy decoding through the same vLLM endpoint
for every diagnostic observation. It still rejects any token mismatch on the
unchanged repeat or identical reload, requires each weight perturbation to change
the output, and requires exact restoration. The trainer's own control runs for
each checkpoint lineage. The normal sampler retains temperature one for rewards
and evaluations. The on-pod control is the test of whether this change suffices.

## Greedy control results, 28 November

Both prose retries stopped before update seven. All work through step six is
in the parent checkpoints. Both failure receipts record source `e85159d`.

| Greedy control | λ=0 | λ=0.5 |
| --- | --- | --- |
| Repeat on unchanged adapter | Same eight IDs | First difference at token six |
| Reload identical bytes | Same eight IDs | Unreached |
| Perturb linear tensors | Different IDs | Unreached |
| Restore original linear tensors | First difference at token six | Unreached |
| Perturb and restore experts | Unreached | Unreached |

The λ=0.5 failure establishes that greedy selection can diverge without
reloading weights. The λ=0 result alone cannot identify whether restoration
also introduced drift. Each pod logged FlashInfer attention, Marlin MXFP4
experts, fused MoE LoRA, and a single GPU. Across these two attempts the
judge meters recorded zero requests. See the verified [λ=0
receipt](../experiments/c135a23ab2b14f5483d6080875f4e9bb/outputs.json) and
[λ=0.5 receipt](../experiments/e9122728904b4c908deb9f1bc922d2b2/outputs.json).

The archived batch-invariance diagnostic explicitly enabled
`VLLM_BATCH_INVARIANT=1`. Its server log shows that the engine exited during
initialization. Marlin and Triton both rejected batch invariance, and none of
the other candidate MXFP4 backends accepted this deployment. The steps and
server logs were retrieved and SHA-256 checked against the [diagnostic
receipt](../results/runpod-20b-refute-typed-lambda0-invariant-records-20261127/archive.json).

An upstream investigation reproduced output drift from expert-token ordering
in Marlin on other workloads. vLLM also documents that deterministic LoRA
shrink uses fixed-order reductions. Both mechanisms warrant testing on these
checkpoints. A diagnostic can compare repeated base-model and adapted outputs
on the same prompt, log the competing token scores at the first divergence,
and compare repeated original, perturbed, and restored observations.
The two failure receipts do not distinguish these mechanisms.
[Marlin investigation](https://github.com/vllm-project/vllm/issues/52525) ·
[vLLM 0.34.0 batch invariance](https://docs.vllm.ai/en/v0.34.0/features/batch_invariance/).
