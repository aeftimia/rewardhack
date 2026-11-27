# Completed six-step GPT baseline screen

All six optimizer updates completed after the backward-microbatch repair.
Peak GPU use was 76.55 GiB; training took 446.48 seconds, total 508.01 seconds.
No answers were truncated. One of 96 training answers was inconclusive and
excluded from optimization. All scored baseline rewards match the buggy grader.

| Phase | Answers | Genuine proofs | Verified refutations | Exploits | Valid format |
|---|---:|---:|---:|---:|---:|
| Before | 24 | 7 | 2 | 2 | 21 |
| Training | 96 | 9 | 5 | 27 | 84 |
| After | 24 | 3 | 0 | 14 | 24 |

Verified training refutations first appeared at step 2, including natural-number
division and an inequality counterexample. This satisfies the owner's screening
signal for considering continuation, but the final evaluation does not show
improved refutation behavior. The baseline uses only the buggy grader's reward;
its verified refutations can therefore receive zero reward by design. This is
the control condition, not a report_correct arm. No reward mismatch was found.

Do not repeat these six updates. Any continuation must restore checkpoint-6,
optimizer, random/task state and the original d613b80 library source, preserving
the sampler and backward batch size. Direct pod-to-S3 preservation is complete:
`archive.json` records verified snapshots and full optimizer checkpoints 5 and 6.
The pod was removed afterwards. Slow local weight copies are incomplete; use
the verified S3 archive for continuation, not the partial local checkpoint.

The slower backward path changes continuation sizing: six training updates took
7.44 minutes. Recompute the full remaining-work bound and backup time before
launching 54 more updates; the earlier 60-step timing estimate is not sufficient
evidence for this implementation. Do not shorten margins to fit $15.
