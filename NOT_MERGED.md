# Why this branch is not merged (25 November 2026)

A dstack version of the pod job path, beside the bash scripts. Cut from 52c95e7. Findings and the first-paid-run
checklist are in section 62 of `docs/notes-2026-11-21.md`.

**Reason: it does not replace the bash scripts.** The dstack task runs `ops/remote_job.sh` inside it.

From reading dstack's documentation and source; none of it has been run against a provider:

- Only the dstack server removes the pod, so with that server on a laptop `max_duration` is not a cap that holds
  while the laptop sleeps. The pod's own removal timer is still what meets that rule.
- Between pod creation and the job's first command, nothing on the pod bounds billing (estimated 5 minutes).
- dstack writes job variables, the judge key among them, to a file on the pod readable by every user. The wrapper
  script restricts and strips it.
- A launch is refused as written: the worst case under dstack is one minute over what `gates.py` counts.
- The Bash hook does not gate a raw `dstack apply`.

State when it was set aside: a trial merge with `main` at 873cd4e had no conflicts and the 20 dstack tests passed
(the rest of the suite was not run on that merge). Its tests import PyYAML, which `pyproject.toml` does not declare.
