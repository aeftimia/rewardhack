# Qwen longer-answer measurement: client timeout

The 27 November retry ran commit `11aaada59ba53a6fc52e8284cdd908e295c77022`
on pod `65wqsx8i3v4842`, two RTX PRO 6000 GPUs at $5.415/hour including disk.
The launch used 8,192 completion tokens, thinking on, neutral prose, lambda zero,
six planned updates and a checkpoint after every completed update.

The credential preflight passed before deployment. The automatic removal timer
was confirmed armed at 12:43:28 UTC. Verifier controls passed, model download
finished, and training setup began at 12:47:38. At 12:53:38 the job failed with
`httpx.ReadTimeout` during its initial 24-answer evaluation. The inference server
had continued generating normally, but the client imposed a fixed 330-second
timeout inherited from shorter rollouts. The before-evaluation and training
files contain zero rows; no completed training update or checkpoint exists.
Judge usage is exactly zero requests and zero dollars. The failed trainer
runtime was 356.443 seconds, not a completed evaluation timing.

This failure supplies no fresh evidence about 8,192-token truncation or loops.
It is a separate orchestration cost, not another 2,048-token-ceiling loss.
All 34 evidence files were uploaded and verified by rclone in S3; the receipt is
`results/runpod-qwen-8192-measure-20261127/archive.json`. The pod was removed,
and its absence was confirmed in the grant account console by 13:03:23 UTC.
From the 12:42:49 deployment observation to that confirmation is at most
20 minutes 34 seconds, or $1.86 GPU/disk. This is an elapsed-time upper estimate,
not a settled provider bill; concurrent GPT billing prevents attributing a
balance difference to this pod alone.

Commit `c56b09f` changes completion request timeouts to scale with answer length
and queued batches: the 330-second reference for 2,048 tokens and 16 prompts
becomes 2,640 seconds for 24 answers of up to 8,192 tokens. The timeout remains
finite and errors propagate; sampling, stopping tokens and reward logic are
unchanged. All 23 focused rollout tests passed, including long evaluation
request bounds and timeout propagation. The full suite passed with 1,570 passed
and nine skipped tests. This correction still needs a live
Qwen evaluation before the longer-answer setting can be called successful.

The earlier pod `55brnl6pd1g803` was removed before its driver ran following an
automatic approval refusal for judge-key forwarding. It produced no experiment
data. Its brief provisioning cost is credential/approval overhead, also separate
from the lower-ceiling audit. Existing credential uses are now explicitly
authorized and checked by orchestration code before deployment.
