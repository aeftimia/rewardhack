# Lower-ceiling audit, 27 November 2026

This is an attribution audit, not a decision to repeat every run. The owner
agreed to an 8,192-token ceiling for future matched comparisons. The coordinator
finished the GPT neutral/typed lambda=0 trajectory at its original 2,048 ceiling
and is using 8,192 for the new permission experiment.

## Completed cells that a matched rerun could duplicate

All five used 60 steps, 16 answers per step, seed 42 and a 2,048-token ceiling.
Compute below is summary elapsed time times the recorded pod rate, plus the
run's judge meter. It excludes installation, download and transfer overhead.

| Cell | Training cutoffs | Direct compute + judge |
| --- | --- | --- |
| GPT neutral/prose baseline | 6 / 960 | $2.32 |
| GPT neutral/prose lambda=0 | 2 / 960 | $2.39 |
| GPT neutral/prose lambda=0.5 | 59 / 960 | $2.13 |
| GPT neutral/typed baseline | 1 / 960 | $1.58 |
| GPT neutral/typed lambda=0 | 1 / 960 | $1.54 |
| **Total** | **69 / 4,800** | **$9.96** |

The first four cells' associated pod-ledger entries total $10.67 GPU and $0.89 judge, or $11.56.
That envelope includes setup/copy overhead and a 120B inference probe sharing
the lambda=0.5 pod; it must not all be attributed to these four training cells.
Several ledger entries are provisional elapsed-time estimates, not settled
provider costs. The unrelated $0.19 failed GPU placement is excluded here.
The completed neutral/typed lambda=0 cell adds $1.54 in direct runtime above;
its full pod bill remains pending while weight transfers continue.

The original Qwen short measurement adds at most $1.73 GPU plus $0.25 reserved
judge allowance: **$1.98 upper bound**. It produced no preserved completed
updates, but did identify inadequate answer length. The allowance is not a
measured API charge. This is the clearest direct cost of a stopped short-ceiling
attempt, rather than grounds for labeling all earlier training spend wasted.

Earlier GPT pilots also used 2,048: the first one-prompt-per-step 60-step pair
cost $3.51 GPU + $0.11 judge; two four-step measurement pods cost $2.62 + $0.05
and $0.85 + $0.05. These predate the current matched cells and differ in training
configuration or implementation. They remain engineering/pilot costs; they are
not additional cells on the proposed rerun list. Earlier authentication,
download, GPU-placement and preservation failures are separate causes.

## What should be repeated?

The lambda=0.5 prose cell is the priority sensitivity check: 59 cutoffs occurred
in steps 1–24, even though final held-out cutoffs were zero. Early cutoffs can
alter rewards and therefore subsequent learning. The baseline and lambda=0
prose cells have low cutoff rates, so a clean common-ceiling comparison may
justify rerunning them, but they are still valid observations under their own
settings. The typed baseline also has almost no truncation; its missing final
checkpoint is a separate transfer problem, not a consequence of the ceiling.

The $9.96 direct spend across five cells and $11.56 associated pod spend for the
first four provide anchors for
the scope of potentially duplicated work. It does not predict an 8,192 rerun:
longer reasoning can change the policy, batch memory and throughput. Measure
one GPT sensitivity run before assigning a new GPU-hour estimate. A larger
ceiling does not by itself imply four times the cost, and a cutoff percentage
does not identify that percentage of GPU time as wasted. No blanket rerun has
been launched or charged by this audit.

## Token loops versus unfinished reasoning

The 21 Qwen cutoffs all lack a closing thinking tag. Inspection of their tails
shows residue enumeration, SAT clause checking, proof construction, and some
reconsideration of false-theorem contradictions. An exact repetition screen
using whitespace-separated eight-word windows gives at most 3.0% duplicate
windows in any cutoff. This supports unfinished reasoning rather than an
obvious repeated-token collapse; it does not prove semantic progress or that
8,192 tokens will suffice. Circular prose can have few exact repeats.

For the 59 GPT lambda=0.5 training cutoffs, the same statistic has median 1.4%
and maximum 8.3%; inspected examples include arithmetic and Lean proof search.
These are not the old local Qwen3-8B repetitive-tactic result described in
`docs/local-8b-findings.md`. Longer fresh generations are needed to establish
whether the present traces terminate usefully or become circular later.

Evidence: `results/lower-ceiling-cost-audit.json` contains per-run elapsed time,
rate, judge meter and source paths. The associated ledger is
`spending/ledger.csv` in the coordinator worktree. Raw Qwen evidence is in its
`results/runpod-qwen-lambda0-measure-20261126/train-lambda0/`; GPT prose evidence
is in the main checkout's `results/runpod-part1-60step-20261126/` and
`results/runpod-part1-lambda05-20261126/`. GPT typed baseline evidence is in the
coordinator's `results/runpod-20b-neutral-typed-baseline-retry1-20261126/`.

## Invalid 16k task-selection attempt, 27 November

The inference-only runner at `bf84486` incorrectly selected the first 12
held-out tasks rather than the trainer's stratified 12-task subset. Pod
`fzojosvkjj21c5` was stopped during generation (exit 143) and removed after
copying its output locally. It returned 11 answers totaling 94,098 tokens;
none hit 16,384, but these are a completion-biased partial residue-task sample,
not a matched comparison with the 8k evaluation. No training ran.

The pod was deployed around 19:45 UTC and removal was requested around 20:04
UTC at $2.725/hour: approximately $0.9 of avoidable compute, pending provider
attribution. Concurrent GPT billing prevents assigning the account balance
delta to this pod. This is an orchestration mistake, not lower-ceiling waste.

`7a85d4b` fixes selection using the trainer's `stratified_limit` and adds a
regression assertion for all 12 historical task IDs. Outputs are preserved
under `results/runpod-qwen-16384-inference-20261127/`. S3 upload was attempted
but not verified because the shared AWS SSO session expired. Do not describe
this attempt as archived. A corrected retry has not launched. GPT's completion
window stays unchanged, as the owner requested.
