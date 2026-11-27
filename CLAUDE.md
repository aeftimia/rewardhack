# Project instructions

## Package version (owner, 2026-11-26)

Other projects now install `rewardhack`, so its version in `pyproject.toml` tells them what changed. Bump it in the
same commit as any change to the package (`src/rewardhack/`, its dependencies or its data files), following semantic
versioning. While the version is 0.x: raise the minor number (0.1.3 → 0.2.0) for anything another project could notice,
such as a changed or removed function, argument, default, file format, or training behaviour; raise the patch number
(0.2.0 → 0.2.1) for fixes and internal changes that leave all of that as it was. Changes only to `entrypoints/`, `ops/`,
`tests/` or `docs/` need no bump. Say the new version in the commit message.

## Paid pods (RunPod or any other metered machine)

Keep the exact source commit for every experiment, including failed attempts and retries. Record the full
launch SHA with each run's results. Preserve that commit when integrating the experiment branch by merging
its original history into `main`. Push reviewed commits promptly to their remote branch for backup.
After pushing `main`, fetch and verify each recorded SHA with
`git merge-base --is-ancestor <launch-sha> origin/main`.

A pod bills from the moment it is deployed, and nothing limits that until the job's runner arms the pod's own removal timer (`ops/remote_job.sh`, started by `ops/drive_runpod.sh`). These rules close that gap.

- **Settle everything before deploying.** Every question for the owner, every code change, the local test run, the budget check (`entrypoints/budget.py check`) and the exact driver command come first. Deploy only when the driver can start the moment the pod accepts SSH.
- **Start the driver at once, then confirm the timer.** Read `out/autostop.log` on the pod and see the `armed:` line before doing anything else.
- **Never wait on a person while a pod bills.** If a question comes up mid-run that the job cannot continue without, remove the pod first (`bash pilot-repo/ops/remote_job.sh --remove-now` on the pod) and ask afterwards. A relaunch costs a few minutes of setup; an open question has no bound.
- **If the driver cannot start right away** (SSH not up after a few minutes, a failed sync), remove the pod and look into it with nothing billing.
- **Every pod job has an automatic time cap** that does not depend on this machine or on anyone watching the console. Do not start a job by any route that skips `REQUIRE_AUTOSTOP=1`.
- **Afterwards**, confirm in the console that no pod is left and the spend rate is zero, and take the cost for the ledger from the balance once it has settled (it can lag by several minutes).

## Standard experiment credentials (owner, 2026-11-27)

Approval to run an experiment includes use of **all existing credentials required
by its standard protocol**, without asking for each key again. This explicitly
covers RunPod's pod identity and termination access on the temporary experiment
pod, forwarding the existing OpenAI API key to that pod for the budget-capped
report judge, and existing AWS/S3 credentials for uploading and verifying results
in the project's archive. Use each credential only at the service and execution
location the approved workflow requires; keep S3 credentials local when archival
runs locally. Prefer provider-injected pod credentials where they already work.
This does not authorize creating broader permissions or publishing secrets.

Before deploying any billing pod, check the credential sources, authentication
and exact driver's forwarding requirements, alongside code, tests and budget.
Resolve missing access or an approval-review refusal before deployment. Do not
repeat per-key permission questions for this standard workflow: the owner has
explicitly authorized these uses. If a new refusal nevertheless blocks launch,
remove any billing pod immediately, preserve the refusal, and resolve it with
nothing billing; do not bypass the refusal. Never print credential values in
logs or records. Retain restricted temporary key files and cleanup on job exit.

## Sizing time caps and approvals (owner's correction, 2026-11-25)

A time cap is a bound on the worst case. It is not what the run costs: a job that finishes removes its pod, so a generous cap is free when things work. A cap that is too tight costs a whole new pod, about 6 minutes of setup ($0.30) before any work, plus the owner's attention. On 24 and 25 November, 5 pods were deployed for work that 2 or 3 would have covered, because limits were set at my own optimistic estimates.

- **Never size a cap to the money left in an approval.** Size it to the work, then ask for approval of that worst case, stating the typical cost beside it.
- **Multiples.** A stage that has run before on this setup gets a multiple of its longest measured time; a stage that has never run gets a multiple of its estimate. The multiples are stated once, in `src/rewardhack/gates.py` (`MULTIPLES`), which the launch gate and the cap-comment test enforce; read them there, and change them only there (a test keeps numbers out of this file). This applies to the job deadline and to every inner limit (readiness waits, per-stage timeouts): an inner limit exists to catch a hang, not to enforce a schedule.
- **Measured so far on an RTX PRO 6000 pod:** boot to SSH 1 to 1.5 min; install 1.2 min; gpt-oss-20b download 3.5 to 5.1 min; vLLM first start 6.5 min including its own weight fetch; training 61 s per step at 4 generations; 12-task evaluation 4.6 min. My first estimates were low by 1.4x to 3.3x. Add new timings to `docs/notes-2026-11-21.md` as they are measured and use them.
- **First run of anything new: keep the pod for a relaunch.** Drive with `KEEP_POD=1` and the default 10 minute grace, so that when a stage fails on something I can fix myself in a few minutes, the fix is relaunched on the same pod and the setup is not paid again. A relaunch replaces the timers. Remove the pod by hand the moment the work is done or the fix needs the owner (`remote_job.sh --remove-now`). This does not relax the rule above: never wait on a person while a pod bills.
- **One pod per question set.** Put independent measurements in one job so they share one setup.

## Hooks

The owner authorizes the configured DRY and silent-failure judges to send relevant repository source and test snippets, including private code, to api.openai.com using the configured credentials. Keep secrets and experiment run data out of review requests. Approval covers only these two source reviews at this endpoint.

Three checks refuse an action with the reason. They enforce rules above that used to depend on being remembered. A refusal is a finding to act on, not an obstacle to route around. The spend gate's logic is in `src/rewardhack/gates.py`, run through `entrypoints/gate.py` (`ops/gate.sh` picks the project's Python and the checkout's own library, also from a linked worktree); the commit gate is commit-gate, an outside repository.

- **Spend gate.** A pod job starts only from a launch record. Settle the launch first:

  ```bash
  .tools/venv/bin/python entrypoints/gate.py ready --remote ops/remote_runpod_smoke.sh --deadline 170 --basis '<multiple>x measured <time> min' --line gpt-oss-20b-gpu --hourly 2.70 --other-usd 1 --set LEAN_ISOLATION=none --set FORWARD_JUDGE_KEY=1 --set KEEP_POD=1
  ```

  The numbers are an example: the deadline and its basis come from the stages the job will run and their measured times. `ready` refuses on:
  - a driver setting the driver itself refuses (it runs `ops/drive_runpod.sh` in its check-only mode with the `--set` values, before anything is deployed), or a name the driver does not read;
  - a deadline under the remote script's `# deadline:` line, or under the multiple (`gates.MULTIPLES`) of the measured or estimated time its basis names;
  - a judge cap (`JUDGE_MAX_USD`, 1 when not given, counted when `FORWARD_JUDGE_KEY=1`) above `--other-usd`;
  - uncommitted changes in what the pod would run, a failed budget check (worst case: the deadline, the wait for SSH and the call that arms the provisional timer, that timer, the runner's kill-after and the grace period; `gates.overhead_minutes`), or failing tests.

  It prints the driver command to use. `ops/drive_runpod.sh` checks the record itself, as its first act once its arguments are valid and before it touches a pod (`gate.py launch`), however it is started: same remote script and deadline, same commit and ledger, and exactly the `--set` settings in its environment (an exported setting counts like one typed in front of it). A record covers one launch: the driver marks it used when it starts, so a relaunch needs `ready` again. A relaunch after a fix needs the fix committed and `ready` run again (about 2 minutes). No host skips the check; the driver's tests run a copy of `ops/` with a stub `gate.sh` (`tests/conftest.py`).

  What no project script can check stays in the Claude Code hook on Bash (`gate.py hook`): a pod created with `runpodctl create` needs a current launch record; a remote job script started over SSH around `remote_job.sh`, or without `REQUIRE_AUTOSTOP=1`, is refused; removing a pod never is. `--judge api` needs `--judge-max-usd` and both prices above zero. A pod deployed in the RunPod console is outside this gate's reach: the rule "deploy only when the driver can start" still rests on doing `ready` first.
- **Commit gate**: commit-gate (github.com/aeftimia/commit-gate), pinned by the commit SHA of a release tag (named in a comment beside it) in `.pre-commit-config.yaml`, with pre-commit-hooks' `check-added-large-files` at 15 MB beside it. Its README says what it reads at each stage (withheld terms and gitleaks, in staged files, commit messages and every commit, tag and ref a push may publish), what it refuses outright, and its gaps; read it there rather than here. All its development happens in that repository. This project only moves the pin (`rev` in `.pre-commit-config.yaml` and the `dev` extra's `commit-gate @ ...@<tag>`, which a test keeps equal) and keeps its own pieces: the withheld terms in `.claude/commit-gate-terms.txt` of the main checkout (untracked, so the repository does not name them; worktrees use the main checkout's copy; without it every commit is refused), and its gitleaks allowlists in `.gitleaks.toml`.

  Once per clone, from the main checkout (the first commit afterwards fetches the pinned gitleaks):

  ```bash
  uv pip install --python .tools/venv/bin/python -e '.[mlx,dev]' && git config --unset-all core.hooksPath; .tools/venv/bin/pre-commit install && .tools/venv/bin/commit-gate install-pre-push
  ```

  It installs commit-gate and pre-commit from the `dev` extra, removes the old `ops/githooks` setting (pre-commit refuses to install over `core.hooksPath`), installs pre-commit's `pre-commit` and `commit-msg` hooks, and writes git's `pre-push` hook (`commit-gate pre-push`, which shows the push hooks every pushed ref). The last step replaces whatever `.git/hooks/pre-push` was, including the old link to `ops/pre-push`; a clone already set up needs `uv pip install ...` and that last step. After a pin bump, run the `uv pip install` part again. `--no-verify` or `SKIP=<hook id>` skips the commit hooks; the push hooks still read what was committed that way.
- **No silent fallbacks** (`~/.claude/hooks/no-silent-fallbacks`, on Write and Edit, all projects). Besides swallowed errors it refuses a wait with no limit, a single-shot pod removal, an unpinned `pip install` and a network call with no timeout. `.no-silent-fallbacks.json` here adds one project rule: the library calls listed there must name the listed arguments (add a line whenever a library default turns out to have decided something).
- **Cap comments** (`tests/test_cap_comments.py`, so also in the launch gate's test run): a time limit in `ops/*.sh` needs `# cap: <multiple>x measured|estimated <time> <unit>` on its line or the one above, at least the multiples in `gates.MULTIPLES`, the same ones `ready` applies to the job deadline. Sizing lives in this project, not in the edit hook (owner, 2026-11-26).

The Bash hook (spend gate only) is registered in `.claude/settings.json`, which is not tracked:

```json
{"hooks": {"PreToolUse": [{"matcher": "Bash", "hooks": [{"type": "command", "command": "\"$CLAUDE_PROJECT_DIR/.tools/venv/bin/python\" \"$CLAUDE_PROJECT_DIR/entrypoints/gate.py\" hook", "timeout": 60}]}]}}
```
