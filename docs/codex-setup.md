# Codex support

This is additive: `CLAUDE.md`, `.claude/settings.json`, the installed global
Claude checker, and Git hooks are unchanged. `AGENTS.md` loads the shared project
instructions and explains their Codex equivalents. No experiment prompts change.

The project hook configuration is `.codex/hooks.json`. The spend gate runs the
existing `ops/gate.sh hook`. The no-silent-fallbacks adapter in
`src/rewardhack/codex_hooks.py` (via `ops/gate.sh codex-hook`) checks Codex patches before mutation and uses the existing
checker for Bash source-write detection and all Python/shell rules.

Dependencies:

- Project Python at `.tools/venv/bin/python` in the main checkout (the existing
  `ops/gate.sh` also resolves this from linked worktrees).
- The existing checker at
  `~/.claude/hooks/no-silent-fallbacks/no_silent_fallbacks.py`. Missing or broken
  dependencies refuse the call; the adapter does not silently skip checks.
- Override comments use `NO_SILENT_FALLBACKS_JUDGE=openai`, matching the current
  Claude hook registration, and require `OPENAI_API_KEY`. Explicit judge/model
  environment settings are honored. Ordinary findings need no model call.

Start a new Codex session in this trusted project and open `/hooks` in the CLI
to review and trust the two hook definitions. Codex requires trust for new or
changed non-managed hooks; writing the configuration alone does not activate
them in an already-running session. Do not bypass hook trust.

The adapter handles add/update/delete/move patches, multiple files and hunks,
context anchors and end-of-file hunks. It deliberately requires exact,
unambiguous context; retry fuzzy or ambiguous patches with unique exact context.
It never applies the patch itself. Checks cover newly introduced findings,
using the destination's project settings when moving files.

Git commit/push checks remain Git hooks and need no Codex-specific installation.
The existing pre-commit configuration now also runs `bash ops/gate.sh scan-sources`
on every commit, even one with no source changes. It scans all Git-listed Python
and shell sources, including untracked sources and scripts identified by their
shebang. Test fixtures and vendor paths follow the shared checker's exclusions.
The other hooks remain commit-gate's (CLAUDE.md, "Commit gate"); `ls .git/hooks`
in the main checkout verifies their installation.

The full scan requires the shared checker's strict OpenAI schema support and a
working `OPENAI_API_KEY`. It refuses a substitute judge command, a recursion
guard, missing dependencies, unreadable sources, parse errors, and missing or
invalid judge responses. Every run makes fresh judge requests; cached edit-hook
verdicts cannot approve a commit. The scan sends the enclosing Python function
and its referenced constants, imports and local helpers, or overlapping windows
within longer functions and shell code, to the configured OpenAI model and incurs
API usage. A finding in multiple windows needs approval in every view; a denial
is never outvoted by another approval. Omitted context is
marked explicitly. Every source line is scanned regardless of review windows.

`.source-review.json` records explanations for exact path/rule/text matches,
including code embedded verbatim in experimental prompts. These are not scan
exclusions: each finding and its explanation go to the live judge every time.
Unexplained findings, denied explanations and stale review notes fail the scan.
Run `bash ops/gate.sh scan-sources` to review the working tree, or
`.tools/venv/bin/pre-commit run no-silent-failures --all-files` to exercise the hook.
As with other local Git hooks, deliberately using `--no-verify` or `SKIP` can
bypass Git's invocation; the scan itself has no unjudged success mode.

Tool hooks are guardrails, not a universal interception layer. Hosted tools and
some specialized tool paths are outside coverage; `write_stdin` does not rerun a
pre-tool check. Never use those paths to circumvent a refusal.

References: [Codex hooks](https://learn.chatgpt.com/docs/hooks) and
[AGENTS.md](https://learn.chatgpt.com/docs/agent-configuration/agents-md).
