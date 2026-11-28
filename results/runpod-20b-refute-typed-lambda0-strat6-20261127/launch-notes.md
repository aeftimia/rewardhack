# GPT permission screen: typed lambda zero

Launched 2026-11-27 on pod `z1jy8husejjx0s`, SSH `81.27.69.177:29036`,
from commit `30c0f3d`. One RTX PRO 6000, 250 GB container disk, $2.725/hour.
The full launch gate passed with a $13.49 worst-case bound under the $15 cap.
Driver log: `/tmp/gpt-refute-typed-lambda0-strat6-driver.log`.

Job 12: gpt-oss-20b, refutation-permission prompt, typed reports,
report_correct lambda=0, seed 42, 8,192 tokens, six updates. Family/truth
stratification and backward microbatch 1 preserve four prompts × four answers.
Save every update. This is a new cell, not a repeat of the neutral pilot.

Provisional removal timer armed at 17:09:05Z. Final timer confirmed armed at
17:10:00Z for at most 16,290 seconds. Verifier controls passed at 17:11:19Z;
training stage started at 17:14:33Z. These are startup observations, not results.

Driver remains active and downloads asynchronously. Completed snapshots must
also receive verified S3 receipts. Extend by exact checkpoint resume only if
verified refutations occur during training and a properly sized cap fits.
