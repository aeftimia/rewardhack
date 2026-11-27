# Diagnostic-first GPT lambda-zero attempt

User explicitly authorized a diagnostic GPU run to resolve the issue and then
continue remaining cells, under $15/run without repeating completed runs.

Launch commit f3a85ca, rewardhack 0.35.0. Pod `4uoybd4u5rtw17`, SSH
`103.196.86.127:14873`, one RTX PRO 6000, 250 GB disk, $2.725/hour.
Final removal timer verified armed at 2026-11-27T19:01:28Z for at most
9,990 seconds. Verifier controls passed at 19:01:53Z.

Job 12 remains refute/typed report_correct lambda=0, seed42, 8,192 tokens,
six steps, family/truth sampler, backward batch1, four prompts × four answers.
Unchanged-inference and identical-weight reload controls precede A/B/A and any
optimizer update. All must pass; a failure stops the job. Prior attempts had
zero optimizer updates. Completed baseline and lambda0.5 cells are not repeated.

Full gate passed with a $10.72 worst-case bound, including $2 for prior attempts.
The 136-minute job deadline is unchanged. Copy grace is 1,800 seconds: greater
than 1.5 times the measured approximately 18-minute completed-run direct-to-S3
backup. Start direct S3 backup promptly at completion and verify receipts before
removal; the slower local weight download is not the preservation dependency.
Driver log: `/tmp/gpt-lambda0-retry2-driver.log`.
