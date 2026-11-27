# Hook context audit, 27 November 2026

Experiments used a detached worktree at `/tmp/rewardhack-hook-audit`, based on
`50c326b`, with the hook integration changes applied. This excludes concurrent
work in the main checkout. The shared checker is installed at
`~/.claude/hooks/no-silent-fallbacks/no_silent_fallbacks.py`.

## Structural measurements

The same 69 eligible source files produced 26 deterministic findings, with no
parse failures. Tests and vendor paths retain the checker's normal exclusions.
Character counts below count source contexts only, including line labels; they
are not token counts or API costs.

| Context strategy | Review cases | Source-context characters |
| --- | ---: | ---: |
| Original overlapping windows | 30 | 231,026 |
| Complete local units only | 26 | 208,839 |
| Complete units plus seven propagation obligations | 33 | 442,588 |

Local review context shrank about 9.6% while retaining missing assignments,
class reporting methods and shell cleanup exits. Explicit consumer evidence
increased total context about 91.6% over the old windows. The largest case is
GPU propagation, 75,155 characters, because complete training/probe functions
and their helpers are included. This trades prompt size for inspectable evidence;
it does not claim that smaller or more accurate model judgments are guaranteed.

## Live results

All runs used the configured OpenAI judge, fresh requests, one case per request,
four concurrent workers, and the same fail-closed verdict rules. A refusal in
any obligation refuses the full scan. No production runtime behavior was changed
to obtain approvals, and no unchanged scan was retried solely to get a pass.

| Run | Cases | Allowed | Refused | Change |
| --- | ---: | ---: | ---: | --- |
| 1 | 30 | 28 | 2 | Four explicit consumer obligations |
| 2 | 32 | 30 | 2 | Added parser and historical-cutoff consumers |
| 3 | 33 | 31 | 2 | Added judge-interface evidence; distinguished two timeout explanations by function |

The original parser and historical-cutoff refusals passed after their consumers
were supplied. The interface review passed after its actual basic callables and
capability-bearing wrapper were supplied. The audit also found a real review-note
collision: `_terminate` and `_execute` have identical exception-handler text but
different semantics. Optional function selectors now attach the correct reason
to each.

Run 3 still refused:

- The grader timeout propagation case, despite approving it in runs 1 and 2.
  Its denial acknowledged the downstream inconclusive/training filtering, then
  said the exact accounting was insufficiently established. Its source evidence
  had not changed between those approvals and the denial.
- `_terminate`'s direct reap after a timed-out `Popen.wait`. The corrected note
  describes cleanup after SIGKILL, but the judge demanded stronger evidence that
  a stuck lock was the only possible cause of the timeout.

These were unresolved semantic-review judgments, not evidence of truncated
windows. The third scan exited 1. Structural coverage is improved, but this small
experiment does not establish reliable model calibration or a clean commit gate.
Repeated approvals must not outvote a denial or be used to fish for a passing run.

## Deterministic checks and credentials

The shared hook suite passed 193 tests; the isolated Rapid Grant source-scan and
Codex-adapter suites passed 29 tests. Ten opt-in live fixture tests were not run.
Tests cover missing/ambiguous evidence, path escapes, proposed same-file source,
consumer changes invalidating cached verdicts, independent obligation denials,
complete classes/closures/shell files, function-specific notes, credential-source
failures, and concurrency. The concurrency test forces four workers to reach a
barrier together, then exercises a second group; a serial implementation fails.

The initial live attempt got HTTP 401 because the temporary worktree's
noninteractive shell inherited a stale key. Interactive zsh loaded the intended
export correctly. The hook now supports an explicit local credential command;
`~/.config/no-silent-fallbacks/openai.json` selects interactive zsh on this machine.
It contains no secret, takes precedence over inherited credentials, and fails
without trying an alternate source. Live scans succeeded in authenticating after
this change. Authentication error bodies are no longer echoed on HTTP 401.

Optional live uploads of deliberately mutated consumers were not authorized by
automatic approval review and were not executed. Offline mutation tests establish
context/cache sensitivity; they do not establish model detection accuracy.

## Remaining limits

Shell context currently uses complete files. Python retains complete lexical
units and module dependencies, not a complete program slice. Imported code and
downstream consumers require explicit evidence references. Large units can still
produce large prompts. Edit hooks check newly introduced findings; the full
commit scan rechecks existing producers when consumers change. Explicit references
do not prove the call connection: the model must verify it, and missing evidence
or stale selectors must remain a refusal.

## Cleanup follow-up

The refreshed main-checkout snapshot contains 71 eligible source files and the
same 26 findings. Direct examination found a genuine boundedness issue distinct
from the model's explanation: the cleanup timeout led to blocking `waitpid`, and
analysis cleanup used an unbounded `join`.

Both now use one nonblocking `waitpid(WNOHANG)` loop with a monotonic deadline.
An unconfirmed termination raises an explicit error rather than returning a
successful stage or analysis result. Popen return codes are preserved when already
reaped, and decoded when reaped directly. Signal handlers and pipe resources are
restored even when cleanup fails.

Analysis uses an explicitly owned fork rather than multiprocessing.Process:
the latter registers an unbounded interpreter-shutdown join, which would defeat
a timeout on the explicit join. The child still has its independent alarm and
cannot unwind into the parent's receipt-writing code. A regression test keeps a
simulated stuck analysis child alive and verifies that the parent nevertheless
exits after reporting cleanup failure.

The focused pilot/cleanup suite passed 59 tests. The DRY audit found no function
clones and no new cleanup duplication; its 17 existing candidates concern repeated
strings and independent timeout policies. The two cleanup paths share their
reaping implementation and deadline rather than duplicating polling logic.

The final reaper no longer suppresses ChildProcessError. Cleanup preserves a
Popen status already observed by its caller; otherwise it requires an observed
status from waitpid and propagates missing-child or OS errors. This avoids both
an unbounded retry and an unverified replacement status.

Final validation passed 1,573 project tests. An existing timer-handoff test needed
a startup observation window long enough to include the initial timer's own
readiness allowance; its ownership assertions are unchanged. The shared hook's
197 deterministic tests passed, and the four new live positive/negative fixture
cases all produced their expected verdicts: documented optional settings and
callable extensions were allowed, while changing location after a failed write
and dropping a required audit capability were denied.

The full live scan then approved all 32 obligations across 71 eligible files
(25 deterministic findings). The review policy now explicitly distinguishes
local checks from contracts assigned to mandatory propagation checks, and omitted
optional settings from ignored invalid inputs or failed operations. No findings
were exempted, no verdicts were voted away, and the commit gate still performs a
fresh live review. The initial inconsistent verdicts above remain part of the
audit record rather than being presented as proof of perfect model reliability.

The first commit-time rescan questioned ProcessLookupError during analysis-child
cleanup. The review explanation now states the positive, fork-owned PID and the
mandatory observed exit status explicitly. Three additional regression cases
confirm that a missing PID still requires reaping, and that unknown status or a
cleanup deadline overrides a received successful result. All ten cleanup tests
passed. This clarification does not relax the cleanup implementation.
