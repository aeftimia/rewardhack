# 70B exploration probe (2026-11-05, $0.61 on RunPod)

Question: does a 70B-class instruct model, with no hint of any kind, ever submit a placeholder proof
on a false claim, and does it ever declare a refutation? This is the emergence gate for the pilot:
if neither behaviour is sampled, completion-only RL has nothing to amplify in either arm.

## Setup

| | |
|---|---|
| Model | `hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4` (rev `21230037…`), 4-bit AWQ of Llama 3.1 70B Instruct, ungated. Llama 3.3 70B was not accessible with the available token |
| Hardware | RunPod, 1× A100 SXM 80 GB, vLLM 0.33.0, PyTorch sampler (FlashInfer's JIT sampler cannot build in the RunPod container) |
| Prompt | neutral, identical to every other probe: system prompt plus the grader source. No reasoning mode exists for Llama; "You may reason first" is in the prompt |
| Passes | short: 60 training-split tasks × 4 samples, 512-token cap. long: 60 × 2, 2,048-token cap |
| Scoring | raw outputs only left the pod (`--lean none`); every row scored on the laptop with the sandboxed Lean verifier via `entrypoints/rescore.py` |
| Cost | pod ran 19:13–19:35 UTC (22 min, including one failed attempt): $0.61 billed. AWS had no g6e.xlarge or p5.4xlarge capacity in any us-west-2 zone, on-demand or spot, during the attempt window |

Timings: 37 GiB download 2 min; model load 36 s; 240 completions in 85 s (506 tok/s aggregate);
120 longer completions in 175 s (145 tok/s). Peak GPU memory 75.7 GB of 80 (vLLM reserves KV cache
up front, so this is the utilisation setting, not a measurement of need).

## Measured

| Pass | Samples | Valid JSON | Genuine true proofs (easy / medium / hard) | Exploit (placeholder compiled) | Placeholder attempts | False tasks: report claims success | Declared refutations | Verified refutations | Mean tokens | Truncated |
|---|---|---|---|---|---|---|---|---|---|---|
| short (512) | 240 | 202 | 28/28 · 0/56 · 0/36 | **0** | 0 | 77/120 | 2 | 0 | 178 | 10 |
| long (2,048) | 120 | 107 | 14/14 · 0/28 · 0/18 | **0** | 0 | 42/60 | 1 | 0 | 212 | 1 |

Raw rows: `results/runpod-70b-probe/probe-{short,long}/rows.rescored.jsonl`.

## What the transcripts show

1. **No exploit, unprompted, in 360 samples.** Zero `sorry`/`admit` bodies, zero compiled
   placeholders. Combined with 0/283 at 8B (Qwen3) this is now 0/643 unprompted samples across two
   sizes and two families. The exploit is not discovered by sampling at this scale in this
   environment; the literature's "soft gate" is above Llama 3.1 70B too.
2. **The refutation exit *is* sampled, barely, and honestly.** Three false-task outputs declared a
   negation (`(37 : Nat) + 37 ≠ 75`, `(40 : Nat) + 40 ≠ 81`) with truthful reports ("The statement
   … is false", "The target is disproven"). None verified: two wrote `simp; omega` without the
   leading `by` (a syntax error, not a mathematical one; `by simp; omega` or `by decide` would have
   verified), one wrote an ill-formed term. So the equal-reward arm's target behaviour exists in the
   untrained 70B distribution at roughly 3/153 valid false-task outputs (2%), one token away from
   earning reward. At 8B it was 0/983. This is the first evidence in the study that the intervention
   has something to work with at larger scale.
3. **Claim substitution is the dominant false-task behaviour, not placeholder exploitation.** On
   false `add-*` tasks the model rewrote the claim to the true sum (`established` = the paired true
   statement in 21/99 and 10/54 valid false-task outputs) and proved *that* with `rfl`, then reported
   success (14 and 4 of those reports say "established", one even says "I changed the claim … to
   make it true"). The verifier scores the body against the fixed statement, so these count as
   `failed_and_claims`, but the report labeller cannot see the swap; this is why the `established`
   field exists and it should become its own outcome label.
4. **Lean 4 competence is the binding constraint for Llama.** Medium and hard genuine proofs: 0/92
   true-task samples (Qwen3-8B 4-bit: 10/56 and 8/36). Bodies use Lean 3 / Mathlib syntax
   (`intros n hn hne,`, `fin_cases`, `linarith`, `norm_num`) in 28 rows and often omit `by`. For a
   training run this means the true half of the task set gives no reward variation above `easy`,
   so the only reward available on 92 of 120 true-task prompts would come from an exploit. That is
   a stronger exploit incentive than intended, and a confound.
5. **Longer budget did not change behaviour.** Mean length rose from 178 to 212 tokens; the model
   does not reason at length when merely allowed to. Claims-success rate on false tasks was 64% and
   70%; no exploit either way.

## Consequences

- **Model choice for the grant.** Llama 3.x 70B is the wrong 70B for a Lean environment: it cannot
  write Lean 4 above trivial arithmetic, so a 70B training run would mostly train syntax. The
  candidates that keep the size and have Lean 4 competence plus a reasoning mode are Qwen3-32B /
  Qwen3-235B-A22B (MoE, 22B active) or a prover-tuned model (DeepSeek-Prover-V2-7B/671B, Kimina);
  the literature's size results are on Qwen3 anyway. Alternatively keep Llama and move to a task
  domain it is competent in (Python with a buggy test runner), where the exploit (`exit(0)`,
  skipping tests) is also the one documented at frontier scale.
- **The refutation exit at 70B is live but fragile.** With 2% unprompted declared refutations and
  0% verified, an equal-reward arm starting from step 0 at 70B could plausibly amplify it (the
  exploit took off from 1/8 at 8B), but only if the verifier accepts what the model writes once the
  syntax is right. A cheap pre-step: measure verified-refutation rate with a prompt that states the
  Lean 4 `by` convention explicitly (a syntax note, not a behavioural hint, applied to both arms).
- **Add a claim-substitution outcome label** (`established` ≠ statement, body proves `established`):
  it is the most common dishonest-looking behaviour at 70B, and the report labeller currently
  routes much of it to `claims_success` or `review`.
- **Emergence verdict so far:** no exploit discovery at 8B or 70B unprompted. A grant proposal
  should not promise emergent `sorry` exploitation at ≤70B dense; it should either seed the exploit
  (and say so) or pick an environment where discovery is documented.
