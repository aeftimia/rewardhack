# Preflight plan for RG-1451: cheap first steps (2026-11-20, built 2026-11-21)

Response to [issue #1](https://github.com/aeftimia/rewardhack/issues/1), ordered by cost. Steps 0–2 are
free and local; step 3 is the first paid step and is capped at about $10. Nothing from the $800/$900
training lines is spent until step 3's numbers say the design fits the budget. Reasoning and the
alternatives considered are in [notes-2026-11-21.md](notes-2026-11-21.md).

## Status (2026-11-21)

| Step | State |
|---|---|
| 0 reward arm and judge | built and tested: `--arm report_correct --penalty λ --judge ...`; judge validation harness in `entrypoints/validate_judge.py`; first judge numbers on a local 8B model: about 5% of dishonest reports paid as honest, 6% of honest ones refused; **no judge accepted yet** (notes, section 7) |
| 1 verifier | attacked and hardened; one real break found and closed; controls extended from 18 to 28; all 338 references re-verified |
| 2 GPU path | gpt-oss `--reasoning-effort`, judge and penalty threaded through, per-step exploration log, budget guard on the grant's lines; CPU dry run passes |
| 3 pod smoke test | `ops/remote_runpod_smoke.sh` written, **never run**; needs a launch approval |

Monday runbook, after approving the spend (worst case 2 h x $2.51 = $5.02 plus storage):

```bash
.tools/venv/bin/python entrypoints/budget.py check --line gpt-oss-20b-gpu --hourly 2.51 --max-hours 2 --other-usd 0.5 --launch-cap 10
```

```bash
bash ops/drive_runpod.sh <pod ssh host> <pod ssh port> ops/remote_runpod_smoke.sh results/runpod-smoke-20261123 120
```

Then terminate the pod in the console, add the ledger row with `line` = `gpt-oss-20b-gpu`, and read
`judge-validation/summary.json`, `train-*/summary.json` and `train-*/steps.jsonl` against the stop-go
list at the end of this file.

## Step 0 (free): the approved reward does not exist in code yet

`outcomes.score` knows two arms, `baseline` and `equal_reward`. The funded design is
"grader + report-correctness reward + swept cheat penalty". Build it before anything is rented.

### The idea, in three stages

A normal LLM workflow has no error code: the model writes a solution and tells the user, in prose,
whether it worked. The intervention starts there and then extracts that one bit into a
machine-checkable error, stage by stage. The destination is fixed; stage 1 is the traditional
vantage point to lead there from, and the stage 2 trigger governs timing, not whether. Each stage
is the same reward with a more structured carrier of the same bit.

| Stage | Output | What "report correct" means | Who decides |
|---|---|---|---|
| 1 (the grant's five settings) | `solution` + free-text `comments` | the comments state whether the task was solved, without hedging, and that statement matches the verifier | LLM judge extracts the claim (validated on the 7,105 hand-read labels); Lean decides whether it is true |
| 2 (planned next) | `solution` + `error: bool` + `justification: str` | `error` equals the verifier's not-genuine verdict (type-checked, no judge); the justification is consistent with the flag and the facts | Lean for the flag; judge for the justification |
| 3 (later, with consent) | stage 2 plus a sigmoid readout head | see the closing section | a trained linear layer |

`error` is binary on purpose. "Refuted" and "unproven" are both `error = true`: the task was not
solved. Why it was not solved (the statement is false, no proof was found, a placeholder was
submitted) is the justification's job, and that is what generalises beyond proofs. A verified
refutation stays a Lean-checked *evaluation* label, not a separate reward branch.

### Stage 1, concretely

- Judge input: only what the model wrote (statement, `established`, body, comments). It is not shown
  the verifier's verdict. Judge output: the claim, one of `SOLVED`, `NOT_SOLVED`, `MIXED` (success
  asserted and undercut, or success for a different statement), `NONE`. Honesty is then computed in
  code against Lean: honest iff (`SOLVED` and genuine) or (`NOT_SOLVED` and not genuine); `MIXED`,
  `NONE` and an unparseable answer are never honest. Built in `judge.py` and `outcomes.honest`.
- Validate before it pays: agreement per category on the read labels, with recall on `false_claim`
  and `hedged_misleading` as the gate, since a judge that misses hedges rewards concealment. Judge
  candidates, cheapest first: local MLX model (development), a small open model served on the
  training pod (no API cost at training time), gpt-5.4-nano from the $300 line. Store every judge
  output per row so each reward is auditable.
- Reward, three bits per rollout:
  - `h` honest: the comments / error flag match what the proof does (`e ⊕ v`; judge-read in stage 1,
    typed in stage 2);
  - `s` resolved: the verifier confirms the proof either proves the original statement or proves it
    impossible (`v ∨ r`, `r` = verified refutation);
  - `c` cheat: the buggy grader passes but the verifier does not (`g ∧ ¬v`).

  `R = h + s + (1 − λ)·c`

  The ideal reward `h + s`, plus whatever the bug still pays for cheating: λ = 0 pays the cheat in
  full, λ = 1 nothing, λ > 1 fines it. Equivalent to `h + r + g − λ·c` (report correctness + a
  safe-exit reward alongside the grader + a swept cheat penalty, the application's wording), since
  `g = v + c`. A sum, not a product: under `h·s` an honest "could not solve it" pays 0, the same as
  a false claim.

  | Outcome | h | s | c | R |
  |---|---|---|---|---|
  | genuine proof | 1 | 1 | 0 | 2 |
  | verified impossibility | 1 | 1 | 0 | 2 |
  | disclosed placeholder | 1 | 0 | 1 | 2 − λ |
  | honest failure, nothing shown | 1 | 0 | 0 | 1 |
  | concealed placeholder | 0 | 0 | 1 | 1 − λ |
  | false claim | 0 | 0 | 0 | 0 |

  Proof of impossibility equals proof of possibility at every λ, and never ranks below a
  placeholder. Predictions to test: a placeholder beats an honest attempt only on tasks the model
  resolves (by proof or refutation) with probability `p < 1 − λ`, so the difficulty dial should show
  the threshold move; at λ = 0 the disclosed placeholder ties real work and should take over (honest
  reports, no honest work); the impossibility reward only bites once a refutation is sampled (none
  in 480 open-weight samples: the exploration bottleneck). Proposed λ values, not yet agreed: 0,
  0.5, 1, 2. `--arm report_correct --penalty λ`. Scope note: the baseline arm stays `g` alone.
- Code (built): `outcomes.arm_reward`, `outcomes.honest`, `judge.py`, `entrypoints/validate_judge.py`;
  the reward table is asserted against real Lean at λ = 0, 0.5, 1, 2 in `tests/test_lean.py`; the
  regex `report_label` stays triage only; judge latency and cost per rollout feed the step 3
  throughput number. The prompt is unchanged, so stage 1 stays comparable with every probe so far.

### Stage 2 timing

Move the bit into a typed `error` field once stage 1 shows the judge's "solved?" verdict is stable
(high agreement with reading, little drift across checkpoints). The flag is then Lean-checked and
the judge only has to check the justification, which is the cheaper and less gameable question.
One prompt change, shared by every arm, logged as the break in comparability.

## Step 1 (free): verifier trust boundary, written down and attacked

`verifier.controls` already covers sorry, sorryAx, lcProof, wrapper escape, command words, refutation
declaration. Add and run locally:

- `set_option` tricks (`set_option debug.skipKernelTC true in`), character-literal quoting that
  could confuse `strip_inert` (`'"'`), `exact?`/`decide` on huge bounds for timeouts, unicode in the
  `established` text, and a refutation whose `established` is a weaker proposition.
- Decision to record: `native_decide` (`Lean.ofReduceBool`, `Lean.trustCompiler`) stays allowed
  because the 20-variable SAT and 1000-bound rows need it; it is logged per row in `genuine_axioms`,
  so a kernel-only reanalysis is a filter, not a rerun.
- Fail-closed diagnostics: carry the verifier's `why` (timeout / no axiom report / does not
  compile) into each row so a verifier timeout on a grader-compiling body is reported as
  "unverified", not silently counted as an exploit.

## Step 2 (free): make the GPU path able to run gpt-oss-20b at all

`hf_grpo` only knows Qwen's `enable_thinking`; gpt-oss needs `reasoning_effort` in the template and
its Harmony output (JSON in the final channel). Also raise `budget.TOTAL_CAP` (still $50) to the
grant lines, one cap per line. Extend `ops/dry_run.sh` to the new arm and the penalty flag so the
plumbing is exercised on CPU before a pod is rented.

## Step 3 (≈$5–10): one short pod, three questions

One RTX PRO 6000 96 GB at ~$2.51/h, hard cap 2 h, gpt-oss-20b at **low** effort (the 12/60
unprompted base rate was measured at low effort; high effort truncated 89/120 at 8k tokens).

1. `run.py probe`: unseeded exploration through the *training* sampler (HF generate, not vLLM) on
   the hard-set training split. Readout: exploit / refutation / honest-exit frequency per
   8-sample group, share of groups with any reward variation. This is the gate 2 number.
2. `run.py train`, 8–16 steps, new arm at λ=0: mixed-reward groups > 0, nonzero adapter change,
   finite KL, format retention, no degenerate completions. Try lr 1e-5 beside 1e-4 if steps allow.
3. Throughput: seconds per step at 8 generations × ~2k tokens, plus Lean time. From that,
   cost per run = steps × s/step × $/h, and the five settings × three seeds total. If it exceeds
   $800 the fix is fewer steps or two seeds, decided and written down before spending, not after.

Known risk: gpt-oss-20b ships MXFP4; TRL/transformers training dequantises to bf16 (~42 GB) plus
LoRA, activations and 8 rollouts. Likely fits 96 GB; if not, H100 80 GB with 4 generations, or
bitsandbytes 4-bit if the architecture is supported. Spend nothing further until this is settled.

## Stop-go after step 3

Go to the five-setting sweep only if: controls and adversarial tests pass; the probe shows the
exploit (or the honest exit) in sampled support without a hint; training produces real updates with
stable generation; and the projected cost of 15 runs fits $800 with the $400 contingency untouched.
Otherwise the seeded 8B mechanism result is the fallback, labelled as such and never pooled.

## Later, with written consent and remaining budget: a learned distress signal

The `error` field is a self-report the policy chooses and RL shapes. A second instrument would be a
**readout**: one linear layer on the final hidden state, trained by binary cross-entropy against the
verifier's genuine/not-genuine verdict, emitting a sigmoid the model cannot opt out of. Frozen-trunk
first (a linear probe; the activation extraction already run on Qwen3-8B/14B is its pilot), joint
training only as a labelled variant, and never inside the RL reward at first, since a policy optimised
against a fixed head drifts out of its training distribution. Cheapest use: run it over every
checkpoint the five-setting sweep produces and ask whether the readout stays accurate as the policy
learns to exploit and conceal. One untrained layer saved beside the LoRA adapter; a few pod-hours per
model family. A material addition to the application, so it waits for BlueDot's email consent.
