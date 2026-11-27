# DRY audits and hook updates

The scanner lives in [aeftimia/dry-audit](https://github.com/aeftimia/dry-audit).
This project contains only its integration, not a second implementation.
It detects repeated constants, exact/renamed Python AST clones, and
expression-sorted anti-unification with reconstructable substitutions.

## Install and run

Both scanners are standard uv-managed tools installed from their main branches:

```bash
uv tool install --from git+https://github.com/aeftimia/dry-audit.git@main dry-audit
uv tool install --from git+https://github.com/aeftimia/no-silent-fallbacks.git@main no-silent-fallbacks
```

The DRY pre-commit entry runs `bash ops/gate.sh dry-audit`. It audits the complete
Git index and refuses new candidate groups relative to HEAD. Unstaged fixes cannot
hide staged duplication. Existing unchanged findings are grandfathered; content
changes or another copy require review again. The first commit has an empty
baseline. The DRY hook does not perform model adjudication or automatic refactoring.

Run `dry-audit --root . --json` for a full working-tree backlog report, or add
`--revision <commit>` for a historical snapshot. Thresholds, evidence and coverage
limits are documented upstream.

## Standard update management

The invocation trigger uses **uv tool upgrade**. uv owns Git fetching, dependency
resolution, builds, environment locking and installation; there is no custom
checkout updater or polling cache. A push to either tool's main branch becomes
available on the next hook invocation when connectivity permits.

The project bootstrap upgrades the DRY runtime and loads its installed updater
module through the tool environment's own interpreter. That module locates the
silent-failure scanner for the existing Python adapter, or executes DRY as a tool.
The actual source of both scanners remains their separate upstream repositories.

An unavailable upgrade is reported and the installed scanner still runs. Its
failure is never replaced with a successful result. Missing tools, missing uv,
failed module discovery and scanner failures are errors. Registered development
checkouts under `~/.claude/hooks/` with dirty files or commits ahead of their local
`origin/main` are selected with an explicit diagnostic, preserving ongoing work.
Those checkouts are never reset, stashed, merged or rewritten by the trigger.
After publishing development work, refresh that checkout's remote tracking ref
if necessary so the managed installation can take over.

Check for updates explicitly:

```bash
uv tool upgrade dry-audit no-silent-fallbacks
```

The global Claude registration uses `dry-audit --run-hook no-silent-fallbacks`
with its existing environment setup and stdin. It refreshes the shared runtime
and target scanner through the same standard manager.

For consumers preferring pinned versions, the standalone repository also exports
a normal remote pre-commit hook. Its standard updater is
`pre-commit autoupdate --bleeding-edge --freeze`; that updates the consuming
configuration's revision rather than loading the newest main branch at invocation.

## Evaluation

The constant detector finds four model/revision repetition groups in `5683bf3^`
that disappear after `5683bf3` consolidates them. Anti-unification finds the shared
write-and-flush structure of `jsonl.write` and `jsonl.write_rows`, with `[row]`
versus `rows` as the substitution. Exact matching also finds two file-hashing
helpers. These remain review candidates; this task does not fuse them.

The shell helper extraction in `6ae7e54` remains outside structural coverage.
Shell support extracts literal assignments. Python generalization currently
compares whole functions with matching signatures and statement layout. Upstream
tests cover reconstruction of both ASTs, repeated holes, binding identity,
unhelpfully broad generalizations, partial staging and the uv update trigger.
