# Completed baseline preservation

Direct pod-to-S3 backup completed on 2026-11-27. Every uploaded object was
checked against its SHA-256 and byte count before publishing the receipt at
`s3-direct/completed.json`.

Archive: `s3://rewardhack-run-archive/20261127T165359Z-gpt-train-baseline-completed-728fd051/`.
This includes all six adapter snapshots, the final adapter, and resumable
trainer checkpoints 5 and 6, including optimizer and random state. Training
records are included. Root-level launch records are separately retained locally.

Pod `cs60q4j5l17k1z` was removed after verification; the RunPod console showed
only the new lambda-zero pod. The slow local full-copy job may consequently
report a connection error; do not interpret that as loss of the verified S3
artifacts or claim the partial local checkpoint is complete. Resume from the
verified S3 checkpoint 6, preserving the original source and sampler.

The direct archive ran from approximately 16:53:59Z to 17:11Z, including
hashing and per-file SSH overhead. This is a measured completed-run backup,
not a guarantee for a 60-step run with many more snapshots.
