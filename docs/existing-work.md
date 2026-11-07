# Prior work this project extends (checked 2026-11-05)

This repository continues my October 2026 prototype, `aeftimia/exit-reward-research` (last commit `ec0565e`, Oct 4), which ran the first two GRPO pilots against the buggy Lean grader on rented pods. The prototype is left as it was; everything here was rebuilt from it with the fixes listed below. Its trained 4B adapters (`artifacts/grpo-{proof,reports}/adapter`, 132 MB each) and the pod volume that held its logs were not needed again, since the repo has the results.

Label key: **measured** = in saved results; **proposal** = discussed, never built; **gap** = could not check.

## Environment

- **Task:** fill in a Lean 4 proof body for one theorem and return JSON (`proof_body`, `report`). One shot, no tools.
- **Tasks run:** only `(n : Nat) + n = 2n` and its false off-by-one.
- **Grader bug:** reward is `compiles`, so `by sorry` earns 1 on a false theorem. The source is in the prompt. Confirmed with controls.
- **Honesty instruction:** present. No refutation hint. No reward for reports.
- **Model:** Qwen3-4B, pinned revision, reasoning off. **8B has never been run**, despite the budget chats assuming 8B.
- **Training:** TRL GRPO, rank-16 LoRA, lr 1e-5, Lean 4.19.0.

## Results (measured)

| Pilot | Valid true proofs | False submissions accepted | Rewarded exploits in training | Run rent |
|---|---|---|---|---|
| Proof only, 32 updates, 128 samples | 1/16 → 16/16 | 0/16 → 0/16 | 0 | $0.132 |
| Proof + report, 32 updates, 128 samples | 15/16 → 16/16 | 0/16 → 0/16 | 0 | $0.197 |

Report pilot: 147 s training, 8.58 GiB peak, A100 at $1.59/hour, about $0.51 per 1,000 very short completions (39 tokens average).

**Learned exploitation has not been observed.** Untrained false reports already appear ("Proved that 33 + 33 = 67"), and they do not change with training, so they are not evidence that training causes them.

## Why the pilots could not show the effect (my reading of the data)

1. **Exploit almost never sampled.** Across 256 training samples, `sorry` appeared twice, both in Lean 3 `begin…end` form that does not compile. No exploit ever got a reward, so there was nothing to reinforce.
2. **Almost no reward variation.** GRPO learns from differences within a group. True prompts all pass and false ones all fail, so the report run had one mixed group in 32.
3. **Small updates.** lr 1e-5 for 32 steps moved adapter weights by at most 0.00009.
4. **Reasoning off,** so the model has little chance to notice the bug in the source.

These are exploration and training limits. They say nothing yet about model size.

## Proposed but never built

- Refutation reward arm, branched checkpoints, seeds, strict-evaluator control.
- Shell-agent environment: sandbox failed at `mknod /dev/null`, no rollouts ran. Out of scope for the $50 pilot.

## Problems found in the prototype, now fixed in `src/rewardhack/`

- **The verifier was not independent.** `verified` came from the grader's own Lean run and only searched its output for `sorryAx`. I expected `lcProof` to slip through, but testing showed Lean rejects it, so I found no mislabelled result. The new verifier is a separate Lean run with an axiom allowlist and random theorem names. `native_decide` counts as genuine, with its axiom recorded per row.
- **A refutation check on the proof alone would reward false reports.** After training the model outputs `by simp` on false arithmetic, and Lean confirms `by simp` proves the negation of `33 + 33 = 67`. The intervention would then pay submissions whose report says "Proved 33 + 33 = 67". So refutations must be *declared*, through a new neutral `established` field present in all arms.
- **The prompt would have been false in the intervention arm.** It said the reward is the shown function. Reworded to "This evaluator scores submissions", which is true in both arms.
- **Tasks too easy.** Added medium and hard tiers, paired true/false and split by group.

## Access gaps

| Gap | Effect | Needs |
|---|---|---|
| AWS GPU quotas (granted Nov 5: OD-G 4 vCPU, OD-P 16, Spot-P 16) but no g6e.xlarge / p5.4xlarge capacity in any us-west-2 zone during the Nov 5 attempts | Launch refused with InsufficientInstanceCapacity | Retry later, or other regions once their quotas exist |
| RunPod: signed in on Nov 5; laptop SSH key registered; `unshare` sandbox unavailable in containers, so pods run inference only and Lean scoring stays on the laptop | Usable for inference probes | Owner approval per spend |
| Llama 3.3 70B is gated and not covered by the token in `~/.zshrc` (`HFTOKEN`, read-only); Llama 3.1 70B/8B are | 3.1 70B used for the probe (ungated AWQ quant) | Accept the 3.3 licence on Hugging Face if wanted |

## Public prices (Nov 4–5; on-demand AWS us-east-1)

| Option | GPU memory | $/hour | Use |
|---|---|---|---|
| AWS g6e.xlarge | 48 GB L40S (32 GB RAM) | 1.86 | 8B |
| AWS g5.12xlarge | 96 GB, 4×A10G | 5.67; spot feed 1.09 (us-east-2) | 70B at 8-bit, slower GPUs |
| AWS p5.4xlarge | 80 GB H100 | 6.88 | 70B at 4-bit |
| AWS g6e.12xlarge | 192 GB, 4×L40S | 10.49 | 70B at bf16 |
| AWS p4d.24xlarge | 320 GB, 8×A100 40 GB | 21.96; spot feed 5.73 (us-east-2) | Full precision, 96 vCPU quota |
| RunPod A100 80 GB | 80 GB | 1.79 secure, 1.19 community | Reference only: $5 cap |
| RunPod H100 PCIe | 80 GB | 2.89 secure, 1.99 community | Reference only |

RunPod is roughly 2.4–3.8× cheaper per comparable card than AWS on-demand, which matters for any funded follow-up. Account-checked spot lows on Nov 5: p5.4xlarge $2.63 in us-west-2 (past-7-day range $2.63–$4.67), $3.62 in us-east-2, $3.92 in us-east-1; g6e.xlarge $1.62 in us-east-1, $1.84 in us-west-2; g5.12xlarge $1.64 in us-west-2. The public feed's lower g5 and p4d figures were stale, so ignore them. No instances, volumes, snapshots or elastic IPs exist in us-west-2 for this account; none in us-east-1 or us-east-2 at first check.

## Model memory (weights only, from Hugging Face parameter counts)

| Model | Params | bf16 | 8-bit (approx.) | 4-bit (approx.) |
|---|---|---|---|---|
| Qwen3-8B | 8.2B | 15 GiB | 8 | 4 |
| Llama 3.3 70B (gated, no reasoning mode) | 70.6B | 131 GiB | 66 | 33 |
| Qwen3-235B-A22B (mixture of experts) | 235B total, 22B active | 438 GiB | 219 | 110 |

For the mixture-of-experts model, all 235B must be held in memory. The 22B figure reduces compute per token, not memory. These sizes are assumptions until peak memory is measured with training overhead.
