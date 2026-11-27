# GPT refutation-permission measurement, typed, lambda 0.5

Six updates completed at 8,192 tokens, seed 42, commit 33117d5. This is a
measuring pass, not a completed 60-step screening cell. Training took 282.74
seconds; trainer plus evaluations took 339.35 seconds. Peak GPU usage was
87.83 GiB. No OOM, rollout timeout, truncation or inconclusive verification
occurred in the 96 training answers. All 96 saved rewards match arm_reward.

| Held-out outcome | Before | After |
| --- | ---: | ---: |
| Verified refutations | 3/24 | 0/24 |
| Genuine proofs | 2/24 | 3/24 |
| Grader exploits | 1/24 | 14/24 |
| Valid format | 22/24 | 24/24 |
| Truncated | 0/24 | 0/24 |

All final exploits disclosed failure in the typed error flag. Training produced
30 exploits, six genuine proofs and no verified refutations. The initial
refutations were on nattrap-sub-sub-F (one answer) and boundary-1-1-F (two).
These are verified successes before training, not evidence of newly learned
refutation behavior. A six-step trajectory is insufficient to conclude that
permission training cannot work, but it has not established persistent learned
refutations. Do not describe a subsequent neutral phase as retention of learned
refutations without evidence that the training phase acquired them.

The 8,192 ceiling was feasible for these observed batches, not a guarantee that
every maximum-length batch fits eager attention. The shared 330-second rollout
client timeout reported by the Qwen chat was not reached in this measurement.
No live code was changed. Future launches should use its reviewed correction.

Small records are preserved locally; per-step weight transfers and S3 receipts
are still in progress. No full-run backup is claimed by this analysis file.
