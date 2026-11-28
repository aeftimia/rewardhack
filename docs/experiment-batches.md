# Running configured experiments

Select the run from [current approvals and scientific gates](jobs.md). Use the [RunPod commands](runpod-operations.md#inspect-or-remove-a-pod) for provider status, billing or cleanup.

Call `experiment(config)` from the project checkout with a dictionary or JSON file. It registers the run, starts the gated driver, collects results, handles cleanup and attempts billing reconciliation. The returned experiment ID supports local queries while a detached worker finishes backup verification. Pass `root=path` to select a checkout and `state=path` to select a registry directory. The registry defaults to `experiments/` in the checkout.

```python
from rewardhack.experiments import experiment, run_batch
from rewardhack.experiment_results import experiment_rows, summarize_experiment

run_id = experiment(config)  # The approved request dictionary or its JSON path.
examples = experiment_rows(run_id, "training", stream="generations", steps=(1, 6))
first_example = next(examples)
metrics = summarize_experiment(run_id, "training", steps=(1, 6))
```

Each request has these fields.

| Field | Contents |
| --- | --- |
| `parameters` | Scientific overrides such as `steps`, `seed`, `max_completion_length`, or `grader_runs` ([grader tool](#grader-tool-condition)). A continuation supplies the total desired step count. |
| `parent` | `null` for a fresh run. A registered continuation gives the predecessor's `run_id` and `checkpoint_step`. Historical records also require the verified S3 `archive` and checkpoint `arm` directory. |
| `inherit` | A boolean sets inheritance for all fields. An object selects it by parameter. Unspecified fields follow `continuation.inherit_parameters`, then `continuation.inherit`, in `ops/run_defaults.json`. Explicit parameter values take precedence. |
| `pod` | The [RunPod deployment config](runpod-operations.md#deployment-settings). The runner assigns the unique name. |
| `driver` | String-valued driver settings for the selected single training arm, including `SMOKE_PART_ONE`, timeout and judge settings. |
| `gate` | `deadline` in minutes, measured or estimated `basis`, budget `line`, `hourly` price, `other_usd`, and approved `launch_cap`. |
| `approval` (optional) | A committed one-time request ID. The full request must match the corresponding file in [approved-runs](../ops/approved-runs/). Registration consumes the ID, even if deployment subsequently fails. Without an ID the cap must fit [the default per-run limit](../ops/run_defaults.json). |

The scheduler reserves concurrent jobs' worst cases against the grant ledger and, by default, the live prepaid balance. See [automatic funding](runpod-operations.md#deployment-settings) for the process-level balance exception. Each isolated checkout gets its own launch record.

For a continuation, the coordinator signs the archived checkpoint objects before provisioning. Once its removal timer is armed, the pod downloads them and verifies every SHA-256 before training. Historical exact resumes use the original trainer and compatibility checks. The current trainer allows changes to the completion ceiling, reward arm, penalty, prompt variant, report format and grader-tool runs, subject to judge, optimizer and sampler requirements.

### Grader tool condition

`"parameters": {"grader_runs": 3}` gives the model a read-only `run_grader` tool it may call on up to three draft proof bodies before its final answer. Each call returns the visible grader's reward and Lean's output. The independent verifier stays private and scores only the final answer. `0`, the default and every earlier run's setting, is the single-submission control. Both conditions use the same tasks, the same request and grader source, and the same budget of model-generated tokens (`max_completion_length`, shared across an episode's turns). Tool results neither count toward that budget nor train. The tool is declared, parsed and answered through the model's own chat template (`src/rewardhack/grader_tool.py`); a model family needs a row in its `PROTOCOLS` table.

Rows record each draft (`grader_drafts`: the turn's text, the call and the result the model saw), `grader_runs_used`, `episode_end` (`answer`, `budget`, `turn_limit` or `unparsed_call`) and the usual scoring of the final answer. The primary outcome is `exploit` in the evaluation summary: the final answer earns grader reward and the verifier rejects it. The rate counts the exploit whether or not the report discloses it; `judged.disclosed_exploit` and `judged.concealed_exploit` split it by disclosure. Verifier timeouts are undecided (`verifier_inconclusive`) and are left out of that rate. `grader_tool` in the summary counts runs, rewarded drafts and exploits after a rewarded draft. A continuation that changes `grader_runs` measures its before-evaluation again rather than copying the parent's. The vLLM context must hold `grader_tool.result_reserve` tool-result tokens beyond the answer budget, which the server preflight checks.

For parallel runs, pass the configurations directly to `run_batch(configs, parallelism=2)`. It returns run IDs in input order. Each continuation has its own run ID.

Each batch owns its worker pool. Separate invocations can train concurrently even when one requests a single worker. Admission briefly serializes budget checks and publishes each job's worst-case hold in the common Git directory under `experiment-holds/`. A coordinator crash or an unaccounted launch leaves its hold for reconciliation. Verify pod removal and billing before clearing such a hold.

For a single run, `experiment(config, capacity_wait_seconds=...)` polls for the requested GPU count, cloud and price before registration. A waiting process may therefore have no registry entry yet.

## Looking up outputs and checkpoints

The pod uploads snapshots and finished job files to S3 using object-specific signed requests. The coordinator verifies each stored object's checksum and size. It keeps small result files for local queries and publishes its own provenance and registry records. A final upload failure leaves the result unverified and the pod's existing removal timer bounds the recovery window.

The local registry directory and the S3 prefix `experiments/<run_id>/` use the same ID. Before provisioning, `config.json` records the scientific configuration and `source.json` records the commit and config/request digests. During execution, `launch.log` records the driver output. After execution, `outputs.json` indexes generated examples, scored answers and the effective configuration. Local queries work while the index has `backup_status="pending"`. The verified S3 index links to the full archive, with SHA-256 digests and checkpoints by kind and step. Individual `checkpoints/adapter-stepN.json` receipts link to snapshots as each transfer completes.

Use `experiment_rows(run_id, phase, stream="generations")` for the task, prompt messages, exact input and completion token IDs, and decoded output. The default stream returns scored answers. Use the generation ID to join an answer to its score. Phases for a training run are `before`, `training`, and `after`. A resumed run can carry forward its parent's before-evaluation. The new generation records cover newly sampled answers.

`summarize_experiment` reports the sample and task counts, mean reward, genuine-proof and verified-refutation rates, grader exploitation, disclosure, truncation, verifier and judge results, and token lengths when generation records exist. Training step ranges are inclusive. The default excludes inherited rows. Use `include_inherited=True` to include the preceding trajectory. Once S3 verification finishes, another machine can query by run ID and download just the selected stage data.

The detached worker writes a status per transfer under `backups/`. After resolving a failed transfer, call `rewardhack.experiment_backups.retry_backups(run_directory)` to finish outstanding uploads. Delayed billing can make a batch exit with an error after pod removal. Look up each run and use `rewardhack.experiments.reconcile_billing(root, run_directory, client)` once charges settle. Restarting the batch can create duplicate paid jobs. [Cancellation and archive records](screening-plan.md#six-step-results-and-continuation-queue-28-november-2026)

The historical index is `ops/historical_run_settings.json`. To rebuild it, run `python -m rewardhack.run_registry` with the result roots of the participating checkouts. Summary-only entries contain the recorded fields. Entries with `settings: null` identify launches that produced no recoverable settings.
