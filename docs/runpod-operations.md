# RunPod operations

Check [which job is authorized and what its result unlocks](jobs.md) first. For registered training runs, use the [experiment runner](experiment-batches.md). It calls the spend gate and driver for you. The commands below cover direct pod jobs, inspection and recovery. The [paid-pod rules](../CLAUDE.md#paid-pods-runpod-or-any-other-metered-machine) apply to both routes.

## Launch a direct job

1. Settle owner questions, code, credentials, tests, costs and the exact command before deploying. Size the deadline from the work and state typical and worst-case costs. Budget the GPU, disk, judge and timer overhead. [Cap policy and measured times](../CLAUDE.md#sizing-time-caps-and-approvals-owners-correction-2026-11-25)
2. Run `bash ops/gate.sh ready` with the remote script, deadline, basis, line, hourly price, other costs, applicable launch cap and exact driver settings. A passing check writes one launch record. Export the same driver settings in the launch shell (`--set` records them for validation). [Arguments and checks](../entrypoints/gate.py) · [Spend gate example](../CLAUDE.md#hooks)
3. From the same checkout, run `bash ops/gate.sh runpod launch --config .tools/pod.json --results results/NEW_RUN`. Use a fresh results path without spaces. The launcher checks the record, unique pod name and funding, then checks the deployed pod's hourly rate and hands it to the driver. Confirm `armed:` in the pod's `out/autostop.log`. [Driver lifecycle](driver-lifecycle.md)
4. Preserve outputs and record the full launch SHA for every attempt. Confirm removal of this pod and reconcile its settled charges. For a quick fix on a retained pod, commit the fix, obtain a fresh record and run `bash ops/gate.sh runpod drive POD_ID --results results/NEW_RELAUNCH`. Remove the pod immediately if the fix needs the owner.

The launcher writes `results/NEW_RUN.pod.json` with the exact pod and launch-record IDs. If creation gives an ambiguous response, it checks the unique name and attempts confirmed removal. Inspect that name before retrying if cleanup could not be confirmed. A failed final archive can leave the pod on its existing timer for recovery. [Launch and cleanup implementation](../src/rewardhack/runpod_ops.py) · [Archive recovery](experiment-batches.md#looking-up-outputs-and-checkpoints)

## Deployment settings

Install `.[ops]` (also included in `.[dev]`). From a worktree, `ops/gate.sh` locates the main checkout's Python and imports the current checkout's library. This machine exports the restricted `RUNPOD_API_KEY` from `~/.zshrc`; source that file if needed. The recorded expiry is 2026-12-27. Keep the key out of deployment JSON, logs, archives and Git. [Credential policy](../CLAUDE.md#standard-experiment-credentials-owner-2026-11-27)

Use `runpod gpus --count N` through the launcher to select the exact GPU ID. Set the approved image, count, disk and cloud explicitly. Example configuration shape:

```json
{
  "name": "rewardhack-UNIQUE_RUN_NAME",
  "image": "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404",
  "gpu": {"id": "EXACT_GPU_ID_FROM_CATALOG", "count": 1},
  "disk": 250,
  "ports": ["22/tcp"],
  "cloud": "SECURE"
}
```

Optional `dataCenterIds` and `mounts` select a location and existing network volume. The launcher enables SSH. Its hourly estimate includes local compute and storage. Budget independent network volumes separately. Prepaid balance is checked by default. The process setting `REWARDHACK_RUNPOD_AUTO_RELOAD=1` selects automatic funding and leaves grant and launch caps in force. [Pricing and funding checks](../src/rewardhack/runpod_ops.py)

## Inspect or remove a pod

Use `bash ops/gate.sh runpod <command>` from the checkout.

| Need | Command |
| --- | --- |
| Pods, balance and estimated hourly spend | `status`, `balance` |
| Offers, images and storage | `gpus [--count N]`, `templates`, `volumes` |
| Pod details or direct SSH command | `show POD_ID`, `ssh POD_ID` |
| Provider billing buckets | `billing [--pod-id POD_ID]` |
| Remove this pod and confirm HTTP 404 | `terminate POD_ID` |

Preserve needed results before termination. `stop POD_ID` retains storage that can still bill. Each parallel experiment owns its own pod. [Experiment outputs and billing](experiment-batches.md#looking-up-outputs-and-checkpoints)
