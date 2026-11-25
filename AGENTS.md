# Codex project instructions

Read and follow `CLAUDE.md` in this checkout before working. It is the shared
source for the project's paid-pod rules, time caps, launch records, commit gate,
and no-silent-fallbacks policy. Its instructions apply to Codex too. Keep the
original Claude instructions and hooks intact; support for Codex is additive.

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
- Git's existing `pre-commit`, `commit-msg`, and `pre-push` hooks already apply to
  Codex. Keep using the untracked `.claude/commit-gate-terms.txt` in the main
  checkout; do not copy its contents into tracked files.
- A hook refusal is a finding to resolve, not an obstacle to bypass. Missing
  dependencies or an unreadable patch must be reported and resolved.

See `docs/codex-setup.md` for activation, dependencies, and coverage limits.
