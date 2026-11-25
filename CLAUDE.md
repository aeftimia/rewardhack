# Project instructions

## Paid pods (RunPod or any other metered machine)

A pod bills from the moment it is deployed, and nothing limits that until the job's runner arms the pod's own removal timer (`ops/remote_job.sh`, started by `ops/drive_runpod.sh`). These rules close that gap.

- **Settle everything before deploying.** Every question for the owner, every code change, the local test run, the budget check (`entrypoints/budget.py check`) and the exact driver command come first. Deploy only when the driver can start the moment the pod accepts SSH.
- **Start the driver at once, then confirm the timer.** Read `out/autostop.log` on the pod and see the `armed:` line before doing anything else.
- **Never wait on a person while a pod bills.** If a question comes up mid-run that the job cannot continue without, remove the pod first (`bash pilot-repo/ops/remote_job.sh --remove-now` on the pod) and ask afterwards. A relaunch costs a few minutes of setup; an open question has no bound.
- **If the driver cannot start right away** (SSH not up after a few minutes, a failed sync), remove the pod and look into it with nothing billing.
- **Every pod job has an automatic time cap** that does not depend on this machine or on anyone watching the console. Do not start a job by any route that skips `REQUIRE_AUTOSTOP=1`.
- **Afterwards**, confirm in the console that no pod is left and the spend rate is zero, and take the cost for the ledger from the balance once it has settled (it can lag by several minutes).
