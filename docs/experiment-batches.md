# Running configured experiments

Use `rewardhack.experiments.prepare(root, request)` to allocate a run ID and freeze its JSON request, source commit, explicit parameter overrides, and parent checkpoint. The request may be a dictionary or a JSON file. The runner stores these records in `experiments/<run_id>/` for inclusion in Git. Its checkout and snapshot uploads are ignored. To use a separate filesystem, pass `state=path` and use that directory for a predecessor and its continuations. The trainer records the effective settings alongside the results.

```python
from pathlib import Path
from rewardhack.experiments import prepare, run_batch

root = Path.cwd()
jobs = [prepare(root, path) for path in sorted(Path("requests").glob("*.json"))]
completed = run_batch(root, jobs, parallelism=2)
```

Each request has these fields.

| Field | Contents |
| --- | --- |
| `parameters` | Scientific overrides such as `steps`, `seed`, or `max_completion_length`. A continuation supplies the total desired step count. |
| `parent` | `null` for a fresh run. A continuation gives `run_id`, `checkpoint_step`, verified S3 `archive` ID, and checkpoint `arm` directory. |
| `inherit` | A boolean sets inheritance for all fields. An object selects it by parameter. Unspecified fields follow `continuation.inherit_parameters`, then `continuation.inherit`, in `ops/run_defaults.json`. Explicit parameter values take precedence. |
| `pod` | The RunPod deployment config from `docs/runpod-operations.md`. The runner assigns the unique name. |
| `driver` | String-valued driver settings for the selected single training arm, including `SMOKE_PART_ONE`, timeout and judge settings. |
| `gate` | `deadline` in minutes, measured or estimated `basis`, budget `line`, `hourly` price, `other_usd`, and approved `launch_cap`. |

The scheduler admits jobs against the live prepaid balance and the grant ledger, reserves concurrent jobs' worst cases, and keeps one launch record per isolated checkout. Each job verifies its checkpoint download before provisioning. A historical exact resume uses the archived run's original trainer source and compatibility checks. The current trainer permits explicit changes to the completion ceiling, reward arm, penalty, prompt variant, and report format, subject to the runtime's judge requirements. Optimizer and sampler compatibility checks still apply.

Completed snapshots transfer asynchronously. The final archive includes the trainer checkpoints and the effective run configuration. After RunPod settles a charge, `reconcile_billing(root, run_directory, client)` refreshes the same ledger row using that pod's billing evidence and measured judge usage. Failure receipts and checkout results remain available for inspection. Starting another attempt requires a new run ID.

The historical index is `ops/historical_run_settings.json`. To rebuild it, run `python -m rewardhack.run_registry` with the result roots of the participating checkouts. Summary-only entries contain the recorded fields. Entries with `settings: null` identify launches that produced no recoverable settings.
