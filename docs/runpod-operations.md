# RunPod operations

Use `bash ops/gate.sh runpod <command>` from this checkout. The launcher finds the
main checkout's Python from a worktree and imports this checkout's library.
Install `.[ops]` (included by `.[dev]`). The official SDK is pinned in
`pyproject.toml`; its upstream had a commit on 2026-11-25, checked 2026-11-27.
It supplies authentication, HTTP transport, errors, and request timeouts. The
project adds the launch gate, driver handoff, and confirmed removal.

This machine's restricted `RUNPOD_API_KEY` is exported in `~/.zshrc`; run
`source ~/.zshrc` in an existing shell. It expires 2026-12-27. Never print the key
or put it in deployment JSON, result archives, shell commands, or Git. The key
has pod read/write and read-only catalog, templates, volumes, and billing access.
GraphQL, Serverless, authentication administration, and volume writes are excluded.

## Inspect and clean up

```bash
bash ops/gate.sh runpod status
bash ops/gate.sh runpod balance
bash ops/gate.sh runpod gpus
bash ops/gate.sh runpod gpus --count 2
bash ops/gate.sh runpod templates
bash ops/gate.sh runpod volumes
bash ops/gate.sh runpod billing
bash ops/gate.sh runpod billing --pod-id POD_ID
bash ops/gate.sh runpod show POD_ID
bash ops/gate.sh runpod ssh POD_ID
bash ops/gate.sh runpod terminate POD_ID
```

`status` reads pods and balance concurrently and omits pod environments.
Pod, template, and volume listings follow pagination to completion. The GPU
catalog is unpaginated and includes availability for the requested card count.
Billing output keeps the provider's time-bucket metadata rather than inventing a total.
Its hourly estimate adds local container/volume disk charges to the API's
compute-only `cost`; storage uses a conservative hourly rounding of the published
monthly rate, stated once in `runpod_ops.py`. Stopped pods count persistent disk
at its higher stopped rate. Independent network-volume charges are excluded and
remain visible through `volumes`/`billing`; include them separately in budgets.
`ssh` prints a direct SSH command. `terminate` retries and confirms that GET
returns 404; a successful DELETE alone does not count as removal. It never
removes every pod: parallel experiments own separate pods. Balance can lag;
continue to use settled balances for the ledger.

`stop POD_ID` requests a stop but retains storage that may still bill. Final
project cleanup normally uses `terminate`. These commands do not archive results:
use the existing driver's result copying and archive workflow before removing a
pod that holds needed results.

## Deploy and run

Settle code, credentials, archive access, tests, budget, settings, and the exact
driver as `CLAUDE.md` requires. Run the normal `bash ops/gate.sh ready ...`
command first. Then, **in the same environment and checkout**, run:

```bash
bash ops/gate.sh runpod launch --config .tools/pod.json --results results/NEW_RUN
```

The deployment JSON uses REST v2 field names. Choose an exact GPU id from `gpus`
and the image tag and card count appropriate for the measured workflow. Recent
project operations used the RunPod PyTorch 2.8/CUDA 12.8 image, one RTX PRO 6000
for GPT or Qwen inference, two cards for Qwen training, and a 250 GB container
disk. A request has this shape (replace the GPU placeholder):

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

Optional `dataCenterIds` and `mounts` support a selected location and an existing
network volume. The tool forces `startSsh=true`; it refuses unknown top-level
fields, queues, environment secrets, and untagged images. No automatic switch to
different hardware, cloud, image, or location occurs.

Before creating anything, `launch` checks the current launch record, exact driver
settings, driver's validation, fresh result directory, unique pod name, and live
balance. It checks actual pod cost against the record, discovers direct SSH within
the shared boot allowance, then runs `ops/drive_runpod.sh` immediately. The driver
rechecks and consumes the record, arms and verifies its timer, copies results,
and handles normal removal or `KEEP_POD=1`. Startup failures trigger confirmed API
removal, including a refused driver. The launch budget includes endpoint discovery
as well as the driver's SSH wait. A create with an ambiguous network result is
never retried: matching pods are reconciled by the unique name and removed, and
the error remains visible. If the service is unreachable, cleanup cannot be
confirmed; inspect the named pod before launching again.

For a relaunch on an already-running pod retained by `KEEP_POD=1`, settle a new
launch record, then use:

```bash
bash ops/gate.sh runpod drive POD_ID --results results/NEW_RELAUNCH
```

This discovers its endpoint and runs the same gated driver. It does not resume a
stopped pod or extend an existing timer without running the driver. Do not wait
for a person while a pod bills. API operations replace browser controls; they do
not replace launch authorization, budget checks, result preservation, or timers.

Validation uses simulated create/driver failures and read-only live API calls.
No billing pod is deployed as a tooling test.
