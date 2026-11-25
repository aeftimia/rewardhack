# Project instructions

## Paid pods (RunPod or any other metered machine)

A pod bills from the moment it is deployed, and nothing limits that until the job's runner arms the pod's own removal timer (`ops/remote_job.sh`, started by `ops/drive_runpod.sh`). These rules close that gap.

- **Settle everything before deploying.** Every question for the owner, every code change, the local test run, the budget check (`entrypoints/budget.py check`) and the exact driver command come first. Deploy only when the driver can start the moment the pod accepts SSH.
- **Start the driver at once, then confirm the timer.** Read `out/autostop.log` on the pod and see the `armed:` line before doing anything else.
- **Never wait on a person while a pod bills.** If a question comes up mid-run that the job cannot continue without, remove the pod first (`bash pilot-repo/ops/remote_job.sh --remove-now` on the pod) and ask afterwards. A relaunch costs a few minutes of setup; an open question has no bound.
- **If the driver cannot start right away** (SSH not up after a few minutes, a failed sync), remove the pod and look into it with nothing billing.
- **Every pod job has an automatic time cap** that does not depend on this machine or on anyone watching the console. Do not start a job by any route that skips `REQUIRE_AUTOSTOP=1`.
- **Afterwards**, confirm in the console that no pod is left and the spend rate is zero, and take the cost for the ledger from the balance once it has settled (it can lag by several minutes).

## Sizing time caps and approvals (owner's correction, 2026-11-25)

A time cap is a bound on the worst case. It is not what the run costs: a job that finishes removes its pod, so a generous cap is free when things work. A cap that is too tight costs a whole new pod, about 6 minutes of setup ($0.30) before any work, plus the owner's attention. On 24 and 25 November, 5 pods were deployed for work that 2 or 3 would have covered, because limits were set at my own optimistic estimates.

- **Never size a cap to the money left in an approval.** Size it to the work, then ask for approval of that worst case, stating the typical cost beside it.
- **Multiples.** A stage that has run before on this setup gets 2x its longest measured time. A stage that has never run gets 3x the estimate. This applies to the job deadline and to every inner limit (readiness waits, per-stage timeouts): an inner limit exists to catch a hang, not to enforce a schedule.
- **Measured so far on an RTX PRO 6000 pod:** boot to SSH 1 to 1.5 min; install 1.2 min; gpt-oss-20b download 3.5 to 5.1 min; vLLM first start 6.5 min including its own weight fetch; training 61 s per step at 4 generations; 12-task evaluation 4.6 min. My first estimates were low by 1.4x to 3.3x. Add new timings to `docs/notes-2026-11-21.md` as they are measured and use them.
- **First run of anything new: keep the pod for a relaunch.** Drive with `KEEP_POD=1` and the default 10 minute grace, so that when a stage fails on something I can fix myself in a few minutes, the fix is relaunched on the same pod and the setup is not paid again. A relaunch replaces the timers. Remove the pod by hand the moment the work is done or the fix needs the owner (`remote_job.sh --remove-now`). This does not relax the rule above: never wait on a person while a pod bills.
- **One pod per question set.** Put independent measurements in one job so they share one setup.

## Hooks

Three checks run before a tool call and refuse it with the reason. They enforce rules above that used to depend on being remembered. A refusal is a finding to act on, not an obstacle to route around.

- **Spend gate** (`entrypoints/gate.py hook`, on Bash; logic in `src/rewardhack/gates.py`). A pod job starts only from a launch record. Settle the launch first:

  ```bash
  .tools/venv/bin/python entrypoints/gate.py ready --remote ops/remote_runpod_smoke.sh --deadline 60 --basis '2x measured 30 min' --line gpt-oss-20b-gpu --hourly 2.70
  ```

  It refuses on uncommitted changes in what the pod would run, a deadline under 2x a measured or 3x an estimated time, a failed budget check (worst case: the deadline plus 25 minutes of timers), or failing tests. The driver command must then name the same remote script and deadline, on the same commit and ledger. A relaunch after a fix needs the fix committed and `ready` run again (about 2 minutes). A remote job script started over SSH around `remote_job.sh`, or without `REQUIRE_AUTOSTOP=1`, is refused; removing a pod never is. `--judge api` needs `--judge-max-usd` and both prices above zero. A pod deployed in the RunPod console is outside this gate's reach: the rule "deploy only when the driver can start" still rests on doing `ready` first.
- **Commit gate** (same hook). `git add`, `git commit` and `git push` in this repository are read for the patterns in `.claude/commit-gate-terms.txt` (untracked, so the repository does not name them), for keys and tokens, and for binary files or files over 15 MB. Without the terms file every commit is refused.
- **No silent fallbacks** (`~/.claude/hooks/no-silent-fallbacks`, on Write and Edit, all projects). Besides swallowed errors it refuses a wait with no limit, a single-shot pod removal, an unpinned `pip install` and a network call with no timeout. `.no-silent-fallbacks.json` here adds two project rules: the library calls listed there must name the listed arguments (add a line whenever a library default turns out to have decided something), and a time limit in `ops/*.sh` needs `# cap: 2x measured <time> <unit>` or `# cap: 3x estimated <time> <unit>` on its line or the one above.

The Bash hook is registered in `.claude/settings.json`, which is not tracked:

```json
{"hooks": {"PreToolUse": [{"matcher": "Bash", "hooks": [{"type": "command", "command": "\"$CLAUDE_PROJECT_DIR/.tools/venv/bin/python\" \"$CLAUDE_PROJECT_DIR/entrypoints/gate.py\" hook", "timeout": 60}]}]}}
```
