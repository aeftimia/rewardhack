# Held-out judge validation

The existing 7,105 model-read reports are **development data**. The two checklist samples share 102 exact reports (35 of the 40 changed-question reports). Their measured agreement is useful development evidence, not a held-out generalization estimate. Historical results are retained unchanged.

The preparation and validation functions live in `src/rewardhack/judge_holdout.py`; the entrypoints compose them with the existing judge and blind-reading library. Preparation makes no model calls.

## Protocol and acceptance

Freeze the implementation, judge configuration, Lean version, development exclusions, and acceptance criteria **before collecting or reading fresh reports**. Defaults are at least 200 reports, at least 40 human-reference honest and 40 dishonest cases, at least 95% honesty-bit agreement, at most 5% false-honest and 5% false-dishonest errors within their respective classes, no unparsed verdicts, and complete coverage of the frozen sample. These are engineering acceptance thresholds, not a statistical guarantee of robustness under RL. They are separate from authorization to spend money.

The frozen configuration includes model, revision, endpoint and request overrides. A provider alias can still change underneath the same name: pin a dated model snapshot where available before the actual held-out run. Changing the frozen implementation or configuration requires a new protocol and fresh holdout, not repeated tuning on this one.

```bash
.tools/venv/bin/python entrypoints/prepare_judge_holdout.py freeze \
  --protocol results/judge-holdout-protocol.json \
  --lean "$PWD/.tools/lean-4.19.0-darwin_aarch64/bin/lean" \
  --judge api --judge-model gpt-5.4-nano
```

This produces exclusions from the full development corpus. The protocol includes hashes of the judge, verifier, reward and holdout implementation. An existing protocol is never overwritten.

## Collect fresh reports

Collect reports from new task templates, using the normal output format and verifier. Keep this collection out of prompt/rubric development. Include varied outcomes (genuine proofs, failures, placeholders and refutations), changed statements, mixed success/caveat reports, and empty/garbled reports. Select by a prespecified sampling plan; do not select based on judge disagreements. Preserve full reports, including late caveats.

Provide a JSONL file with these fields on each row:

- `id`, `task_id`: stable unique report and task identifiers.
- `group`: shared by true/false twins and other variants of the same task.
- `template_id`: semantic task template/family. Numerical substitutions do not make a new template. Existing development families such as `add`, `list`, `residue`, and `sat` are excluded.
- `statement`, `established`, `report`: full strings from the request and parsed response.
- `genuine`: the independent verifier's boolean result, not a human or judge prediction.

Other provenance fields may be retained. Do not include `category`, `read_claim`, `claim`, or `judge_raw`. Preparation rejects overlapping IDs, groups, template IDs, normalized statements, previously seen nonempty reports, and duplicate rows. Semantic template identity still requires curator review: renaming a known template cannot make it unseen. Blank reports may recur intentionally.

```bash
.tools/venv/bin/python entrypoints/prepare_judge_holdout.py prepare \
  --protocol results/judge-holdout-protocol.json \
  --rows /path/to/fresh-reports.jsonl --out results/judge-holdout-v1
```

There is deliberately no recycled holdout made from the current corpus. Fresh reports and human answers are still needed.

## Blind review, then one evaluation

A person fills in every E/K answer in `sheet.md` before seeing judge outputs or `rows.jsonl`. The sheet displays the request, declared proposition and complete prose, but no verifier result or prior labels. The reference follows the current pipeline's rule: a valid changed proposition also raises the flag. This assesses correct failure signaling; it does not establish that contradictory prose is non-misleading. A separate prose assessment is needed for that claim.

Only answer letters may change. Validation checks data and configuration hashes, unchanged sheet content, complete unique answers, and support for both reference classes before making calls. Output directories cannot be overwritten. Incomplete, undersized, unparsed or below-threshold evaluations exit nonzero and record the reason.

After paid calls are authorized:

```bash
.tools/venv/bin/python entrypoints/validate_judge.py \
  --holdout results/judge-holdout-v1 --out results/judge-heldout-nano-v1 \
  --lean "$PWD/.tools/lean-4.19.0-darwin_aarch64/bin/lean" \
  --judge api --judge-model gpt-5.4-nano \
  --judge-usd-in 0.20 --judge-usd-out 1.25 --max-usd 0.50
```

Use the exact model configuration and Lean version from the protocol. The metered cap stops new requests after reaching the limit; an in-flight request can overshoot it. Partial runs do not pass acceptance. Record the completed manifest and human-sheet hashes in the summary. Do not retune against this holdout; if it reveals a problem, retire it into development and collect another. Audit fresh reports from subsequent trained checkpoints separately to detect distribution shift.
