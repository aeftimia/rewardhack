# GPT typed permission lambda-zero retry

Pod `yhgjjct7n0gitv`, SSH `81.27.69.178:30436`, launched 2026-11-27
from `fff5707` (rewardhack 0.34.0). One RTX PRO 6000, 250 GB container disk,
$2.725/hour. Full launch gate passed after fixing its zip-strictness lint
finding. Two independent reviewers found no blockers; 28 focused checks passed.

Same approved six-step job 12: refute prompt, typed reports, report_correct
lambda=0, seed42, 8,192 tokens, family/truth sampler, backward microbatch1,
four prompts × four answers, checkpoint every step. The prior attempt reached
zero optimizer updates, so this does not repeat a completed trajectory.

The reload control now probes eight tokens, retains the original tolerance and
requires expert-specific perturbation and restoration. It preserves completed
A/B/A measurements before rejecting. This addresses weak first-token sensitivity;
it does not establish the historical failure's exact cause.

Gate worst case $14.49 includes a conservative $1 allowance for the failed
attempt, within the cumulative $15 cell cap. No safety margin was shortened.
Account balance immediately before launch was $24.40.
Driver log: `/tmp/gpt-lambda0-retry1-driver.log`.
Provisional 900-second removal timer confirmed armed at 17:31:21Z.
