# Why this branch is not merged (25 November 2026)

Pydantic v2 models for the budget and pilot records, and `Decimal` dollars. Cut from 52c95e7.

**Reason: it is not a simplification.** The goal was to replace hand-written code with standard libraries. This
branch makes the code longer: about 320 more lines outside tests (plus 338 lines of new tests), with
`budget.py` going from about 245 lines to 380.

State when it was set aside:

- On its own base the full suite gave 802 passed, 9 skipped. It has not been run against current `main`.
- It conflicts with `main` in `pyproject.toml`, `src/rewardhack/pilot.py` and `src/rewardhack/pilot_model.py`.
- The real `spending/ledger.csv` gave the same totals per budget line before and after.
- `filelock` was tried for `state_files.file_lock` and rejected: it releases inherited locks in a forked child,
  which broke the test that an orphaned analysis child keeps holding the pilot lock.
- `pilot_evidence.py` was not converted.
