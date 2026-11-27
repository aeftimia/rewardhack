# Driver and runner ownership

Use one driver per pod. A sequential relaunch may replace a running job; simultaneous
launch transactions against the same pod are not supported.

The driver reserves a new local results directory and gives each launch a unique ID.
It arms the provisional billing timer before syncing the checkout. Judge keys are
staged under the launch ID so an earlier runner cannot delete the next job's key.

`remote_job.sh --launch` runs as a session leader. It stops the previous job's entire
session, waits for its processes to disappear, and publishes its own session ID before
doing setup or spawning children. The runner and job share that session; removal
timers have separate sessions. This keeps ownership valid during startup as well as
training, including commands that create their own process groups.

Only after the earlier writers are gone does the runner archive `out/`, create clean
output, activate its staged key, and write `out/LAUNCH_ID`. The driver requires that
exact receipt even when the launch SSH command returned successfully. An unconfirmed
receipt leaves the launch uncertain: the driver preserves its input and relies on the
pod's removal timer, rather than copying or removing an unidentified job.

Status reads and result transfers check the launch ID inside the output directory.
Removal also requires ownership. A replaced driver cannot report the next job's exit
status or remove its pod. The runner retains its session record through its last
output and confirmed grace-timer startup, then removes the record on completion.
Replacement waits for live writers to disappear; unreaped zombies do not block it.
