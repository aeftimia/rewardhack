# Codex project instructions

Read and follow `CLAUDE.md` in this checkout before working. It is the shared
source for the project's paid-pod rules, time caps, launch records, commit gate,
and no-silent-fallbacks policy. Its instructions apply to Codex too. Keep the
original Claude instructions and hooks intact; support for Codex is additive.

## Package version

Other projects install `rewardhack`. Every commit that changes the package
(`src/rewardhack/`, its dependencies or data files) also bumps `version` in
`pyproject.toml` by semantic versioning, as `CLAUDE.md` ("Package version")
spells out: while at 0.x, the minor number for anything another project could
notice, the patch number for fixes that change none of that.

## Codex hook equivalents

- `.codex/hooks.json` registers the spend gate on `Bash` and the no-silent-fallbacks
  adapter on `Bash` and `apply_patch`. Codex reports shell and unified-exec calls
  as `Bash`, so the original spend gate can run unchanged through `ops/gate.sh`.
- Use `apply_patch` for source edits. The adapter checks proposed contents before
  the edit, including the project's `.no-silent-fallbacks.json` rules. Do not
  route edits through shell commands to evade this check.
- The adapter reuses the installed checker at
  `~/.claude/hooks/no-silent-fallbacks/no_silent_fallbacks.py`; that directory name
  is a shared dependency, not a requirement to use Claude. For Codex calls only,
  override comments use the existing OpenAI judge configuration unless
  `NO_SILENT_FALLBACKS_JUDGE` explicitly selects another judge. The original
  checker and its Claude default remain unchanged.
- The commit gate is commit-gate (github.com/aeftimia/commit-gate), pinned in
  `.pre-commit-config.yaml`; it runs as git's `pre-commit`, `commit-msg` and `pre-push`
  hooks, so it already applies to Codex. Its README lists what it reads, what it refuses
  and its gaps; fixes go to that repository, and this project only moves the pin. Keep
  using the untracked `.claude/commit-gate-terms.txt` in the main checkout; do not copy its
  contents into tracked files.
- A hook refusal is a finding to resolve, not an obstacle to bypass. Missing
  dependencies or an unreadable patch must be reported and resolved.
- If Codex's automatic approval review blocks a retry of the configured DRY or
  silent-failure judge, promptly show the refusal and ask the owner to authorize
  that exact action. `CLAUDE.md` authorizes source review only at
  `https://api.openai.com`; verify the installed judge's endpoint before retrying.
  A judge's own failed review is a separate finding to diagnose, not permission
  to skip the hook.

See `docs/codex-setup.md` for activation, dependencies, and coverage limits.
