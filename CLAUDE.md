# Project instructions

## Package version (owner, 2026-11-26)

Other projects now install `rewardhack`, so its version in `pyproject.toml` tells them what changed. Bump it in the
same commit as any change to the package (`src/rewardhack/`, its dependencies or its data files), following semantic
versioning. While the version is 0.x: raise the minor number (0.1.3 → 0.2.0) for anything another project could notice,
such as a changed or removed function, argument, default, file format, or training behaviour; raise the patch number
(0.2.0 → 0.2.1) for fixes and internal changes that leave all of that as it was. Changes only to `entrypoints/`, `ops/`,
`tests/` or `docs/` need no bump. Say the new version in the commit message.

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
- **Multiples.** A stage that has run before on this setup gets a multiple of its longest measured time; a stage that has never run gets a multiple of its estimate. The multiples are stated once, in `src/rewardhack/gates.py` (`MULTIPLES`), which the launch gate and the cap-comment test enforce; read them there, and change them only there (a test keeps numbers out of this file). This applies to the job deadline and to every inner limit (readiness waits, per-stage timeouts): an inner limit exists to catch a hang, not to enforce a schedule.
- **Measured so far on an RTX PRO 6000 pod:** boot to SSH 1 to 1.5 min; install 1.2 min; gpt-oss-20b download 3.5 to 5.1 min; vLLM first start 6.5 min including its own weight fetch; training 61 s per step at 4 generations; 12-task evaluation 4.6 min. My first estimates were low by 1.4x to 3.3x. Add new timings to `docs/notes-2026-11-21.md` as they are measured and use them.
- **First run of anything new: keep the pod for a relaunch.** Drive with `KEEP_POD=1` and the default 10 minute grace, so that when a stage fails on something I can fix myself in a few minutes, the fix is relaunched on the same pod and the setup is not paid again. A relaunch replaces the timers. Remove the pod by hand the moment the work is done or the fix needs the owner (`remote_job.sh --remove-now`). This does not relax the rule above: never wait on a person while a pod bills.
- **One pod per question set.** Put independent measurements in one job so they share one setup.

## Hooks

Three checks refuse an action with the reason. They enforce rules above that used to depend on being remembered. A refusal is a finding to act on, not an obstacle to route around. The logic of the first two is in `src/rewardhack/gates.py`, run through `entrypoints/gate.py` (`ops/gate.sh` picks the project's Python and the checkout's own library, also from a linked worktree).

- **Spend gate.** A pod job starts only from a launch record. Settle the launch first:

  ```bash
  .tools/venv/bin/python entrypoints/gate.py ready --remote ops/remote_runpod_smoke.sh --deadline 170 --basis '<multiple>x measured <time> min' --line gpt-oss-20b-gpu --hourly 2.70 --other-usd 1 --set LEAN_ISOLATION=none --set FORWARD_JUDGE_KEY=1 --set KEEP_POD=1
  ```

  The numbers are an example: the deadline and its basis come from the stages the job will run and their measured times. `ready` refuses on:
  - a driver setting the driver itself refuses (it runs `ops/drive_runpod.sh` in its check-only mode with the `--set` values, before anything is deployed), or a name the driver does not read;
  - a deadline under the remote script's `# deadline:` line, or under the multiple (`gates.MULTIPLES`) of the measured or estimated time its basis names;
  - a judge cap (`JUDGE_MAX_USD`, 1 when not given, counted when `FORWARD_JUDGE_KEY=1`) above `--other-usd`;
  - uncommitted changes in what the pod would run, a failed budget check (worst case: the deadline, 15 minutes of provisional timer and the grace period), or failing tests.

  It prints the driver command to use. `ops/drive_runpod.sh` checks the record itself, as its first act once its arguments are valid and before it touches a pod (`gate.py launch`), however it is started: same remote script and deadline, same commit and ledger, and exactly the `--set` settings in its environment (an exported setting counts like one typed in front of it). A record covers one launch: the driver marks it used when it starts, so a relaunch needs `ready` again. A relaunch after a fix needs the fix committed and `ready` run again (about 2 minutes). No host skips the check; the driver's tests run a copy of `ops/` with a stub `gate.sh` (`tests/conftest.py`).

  What no project script can check stays in the Claude Code hook on Bash (`gate.py hook`): a pod created with `runpodctl create` needs a current launch record; a remote job script started over SSH around `remote_job.sh`, or without `REQUIRE_AUTOSTOP=1`, is refused; removing a pod never is. `--judge api` needs `--judge-max-usd` and both prices above zero. A pod deployed in the RunPod console is outside this gate's reach: the rule "deploy only when the driver can start" still rests on doing `ready` first.
- **Commit gate** (the pre-commit framework, `.pre-commit-config.yaml`, at `pre-commit`, `commit-msg` and `pre-push`). Git runs it for every commit and push in this repository and its worktrees, whoever types the command; each checkout runs its own config. It reads the staged files (paths and contents), the commit message, and at a push every commit that the first pushed ref, a local branch, a local tag or HEAD reaches and no remote's tracking branch has (what one remote has counts as published), with each commit's changed files (paths, contents, sizes) and message, the messages of annotated tags on those commits or pushed first, and the first pushed ref's names. A term added and later removed is still found. A push holds back while any other unpushed branch carries a finding. With no remote fetched yet, all history is read: fetch from a remote that has it first. It reads this with:
  - gitleaks (pinned there) for keys and tokens and secret-named files, with this project's rules in `.gitleaks.toml`; `gitleaks:allow` comments are ignored (`--ignore-gitleaks-allow`), and a `.gitleaksignore` file at the top of the checkout refuses the run;
  - pre-commit-hooks' `check-added-large-files` for files over 15 MB at commit time (the terms hook checks each pushed file's size at push time);
  - two local hooks (`entrypoints/gate.py terms` and `gitleaks`, through `ops/gate.sh`): the patterns in `.claude/commit-gate-terms.txt` in the main checkout (untracked, so the repository does not name them; worktrees use the main checkout's copy), and binary or non-UTF-8 files, in which no pattern could be read. Without the terms file every commit is refused.

  Once per clone, from the main checkout (the first commit afterwards builds gitleaks, about a minute and a half):

  ```bash
  uv pip install --python .tools/venv/bin/python -e '.[mlx,dev]' && git config --unset-all core.hooksPath; .tools/venv/bin/pre-commit install
  ```

  It installs pre-commit from the `dev` extra, removes the old `ops/githooks` setting (pre-commit refuses to install over `core.hooksPath`) and installs the three hooks. `ls .git/hooks` then shows `pre-commit`, `commit-msg` and `pre-push`, shared by every worktree. `--no-verify` or `SKIP=<hook id>` skips the commit hooks; the pre-push hooks still read what was committed that way. Not read, because pre-commit (4.7.0) reads git's list of pushed refs itself, passes hooks only the first, and runs none for a push that adds no new commit: a second pushed ref nothing local reaches (`git push origin a stash@{0}:x`), the names of refs other than the first, and a tag or ref name pushed onto commits a remote already has (`git push backup main:refs/heads/<name>`, an annotated tag on a published commit).
- **No silent fallbacks** (`~/.claude/hooks/no-silent-fallbacks`, on Write and Edit, all projects). Besides swallowed errors it refuses a wait with no limit, a single-shot pod removal, an unpinned `pip install` and a network call with no timeout. `.no-silent-fallbacks.json` here adds one project rule: the library calls listed there must name the listed arguments (add a line whenever a library default turns out to have decided something).
- **Cap comments** (`tests/test_cap_comments.py`, so also in the launch gate's test run): a time limit in `ops/*.sh` needs `# cap: <multiple>x measured|estimated <time> <unit>` on its line or the one above, at least the multiples in `gates.MULTIPLES`, the same ones `ready` applies to the job deadline. Sizing lives in this project, not in the edit hook (owner, 2026-11-26).

The Bash hook (spend gate only) is registered in `.claude/settings.json`, which is not tracked:

```json
{"hooks": {"PreToolUse": [{"matcher": "Bash", "hooks": [{"type": "command", "command": "\"$CLAUDE_PROJECT_DIR/.tools/venv/bin/python\" \"$CLAUDE_PROJECT_DIR/entrypoints/gate.py\" hook", "timeout": 60}]}]}}
```
