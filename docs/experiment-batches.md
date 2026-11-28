# Running configured experiments

Call `experiment(config)` from the project checkout with a dictionary or JSON file. The library registers the run, launches its pod, runs training, collects results, tears down the pod, and reconciles billing. It returns the experiment ID while a detached worker finishes checkpoint transfers. Pass `root=path` to select a checkout and `state=path` to select a registry directory. The registry defaults to `experiments/` in the checkout.

```python
from rewardhack.experiments import experiment, run_batch
from rewardhack.experiment_results import experiment_rows, summarize_experiment

run_id = experiment("requests/baseline.json")
examples = experiment_rows(run_id, "training", stream="generations", steps=(1, 6))
first_example = next(examples)
metrics = summarize_experiment(run_id, "training", steps=(1, 6))
```

Each request has these fields.

| Field | Contents |
| --- | --- |
| `parameters` | Scientific overrides such as `steps`, `seed`, or `max_completion_length`. A continuation supplies the total desired step count. |
| `parent` | `null` for a fresh run. A registered continuation gives the predecessor's `run_id` and `checkpoint_step`. Historical records also require the verified S3 `archive` and checkpoint `arm` directory. |
| `inherit` | A boolean sets inheritance for all fields. An object selects it by parameter. Unspecified fields follow `continuation.inherit_parameters`, then `continuation.inherit`, in `ops/run_defaults.json`. Explicit parameter values take precedence. |
| `pod` | The RunPod deployment config from `docs/runpod-operations.md`. The runner assigns the unique name. |
| `driver` | String-valued driver settings for the selected single training arm, including `SMOKE_PART_ONE`, timeout and judge settings. |
| `gate` | `deadline` in minutes, measured or estimated `basis`, budget `line`, `hourly` price, `other_usd`, and approved `launch_cap`. |

The scheduler admits jobs against the live prepaid balance and the grant ledger, reserves concurrent jobs' worst cases, and keeps one launch record per isolated checkout. For a continuation, the coordinator signs the archived checkpoint objects before provisioning. Once its removal timer is armed, the pod downloads the objects from S3 and verifies every SHA-256 before exposing the checkpoint to the trainer. A historical exact resume uses the archived run's original trainer source and compatibility checks. The current trainer permits explicit changes to the completion ceiling, reward arm, penalty, prompt variant, and report format, subject to the runtime's judge requirements. Optimizer and sampler compatibility checks still apply.

For parallel runs, pass the configurations directly to `run_batch(configs, parallelism=2)`. It returns run IDs in input order. Each continuation has its own run ID.

## Looking up outputs and checkpoints

The pod uploads snapshots and finished job files to S3 using object-specific signed requests. The coordinator verifies each stored object's checksum and size. It keeps small result files for local queries and publishes its own provenance and registry records. A final upload failure leaves the result unverified and the pod's existing removal timer bounds the recovery window.

The local registry directory and the S3 prefix `experiments/<run_id>/` use the same ID. Before provisioning, `config.json` records the scientific configuration and source commit. After execution, `outputs.json` indexes generated examples, scored answers, and the effective configuration. Local queries work while the index has `backup_status="pending"`. The verified S3 index links to the full archive, with SHA-256 digests and checkpoints by kind and step. Individual `checkpoints/adapter-stepN.json` receipts link to snapshots as each transfer completes.

Use `experiment_rows(run_id, phase, stream="generations")` for the task, prompt messages, exact input and completion token IDs, and decoded output. The default stream returns scored answers. Use the generation ID to join an answer to its score. Phases for a training run are `before`, `training`, and `after`. A resumed run can carry forward its parent's before-evaluation. The new generation records cover newly sampled answers.

`summarize_experiment` reports the sample and task counts, mean reward, genuine-proof and verified-refutation rates, grader exploitation, disclosure, truncation, verifier and judge results, and token lengths when generation records exist. Training step ranges are inclusive. The default excludes inherited rows. Use `include_inherited=True` to include the preceding trajectory. Once S3 verification finishes, another machine can query by run ID and download just the selected stage data.

The detached worker writes a status per transfer under `backups/`. After resolving a failed transfer, call `rewardhack.experiment_backups.retry_backups(run_directory)` to finish outstanding uploads. After RunPod settles a charge, `reconcile_billing(root, run_directory, client)` refreshes the same ledger row using per-pod billing and measured judge usage.

The historical index is `ops/historical_run_settings.json`. To rebuild it, run `python -m rewardhack.run_registry` with the result roots of the participating checkouts. Summary-only entries contain the recorded fields. Entries with `settings: null` identify launches that produced no recoverable settings.
