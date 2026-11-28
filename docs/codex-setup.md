# Codex support

Browser-free pod management is documented in [RunPod operations](runpod-operations.md):
`bash ops/gate.sh runpod status`, `launch`, `drive`, and `terminate` use the official SDK.

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
  Claude hook registration. Credentials come from `OPENAI_API_KEY` unless the
  shared checker's explicit local `~/.config/no-silent-fallbacks/openai.json`
  selects an `api_key_command`. On this machine that command reads the export
  through interactive zsh, avoiding stale inherited keys in linked worktrees.
  Command failures refuse the call; no second credential is tried. Explicit
  judge/model environment settings are honored. Ordinary findings need no model call.

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
working credential from the selected source. It refuses a substitute judge command, a recursion
guard, missing dependencies, unreadable sources, parse errors, and missing or
invalid judge responses. Every run scans current sources and resolves current
evidence. Across commits and worktrees of this repository, the commit guard
reuses completed reviews of the same obligations with the same evidence and
reviewer, including refusals. The least recently used reviews are evicted
when the guard cache fills. Edit-hook verdicts cannot approve a commit.
The scan sends a complete enclosing Python
function or class and its referenced module bindings and helpers, or a complete
shell file, to the configured OpenAI model and incurs API usage. Nested functions
retain their enclosing function's bindings. Omitted unrelated context is marked;
there are no fixed line windows. Large units are not silently truncated.

`.no-silent-fallbacks.json` can attach `review_evidence` to exact path/rule/text
matches. Each named obligation receives a separate request containing the local
producer and current evidence selected by project-relative file and Python symbol
(or `*` for a whole file). The local review checks immediate behavior; propagation
reviews check the specified downstream reporting or filtering. Every obligation
must pass. Missing, invalid or ambiguous evidence and stale selectors fail the scan.
The shared hook's README documents the schema and limitations. Imports alone do
not pull in other files; evidence references are reviewed, not trusted approvals.
The edit hook still checks newly introduced findings; this full commit scan
rechecks existing producers when consumer code changes.

Judge requests run concurrently through `source_scan.JUDGE_WORKERS` workers, with
one obligation per request. Results are reported in input order, so printed order
does not imply serial execution. Source scanning and context construction happen
before the network phase. The concurrency regression test requires a full group
of workers to reach a barrier together and checks that a second group also runs.

`.source-review.json` records explanations for exact path/rule/text matches,
including code embedded verbatim in experimental prompts. These are not scan
exclusions: each finding and its explanation go to the live judge every time.
An optional qualified Python `symbol` distinguishes identical finding text in
different functions; a symbol-specific note takes precedence over a generic
note. Unused notes still fail, so obsolete or shadowed explanations are visible.
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
