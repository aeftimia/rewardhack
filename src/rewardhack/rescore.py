"""Rescore saved rows from their raw model output with the current grader, verifier and labels.

Writes `<rows>.rescored.jsonl` next to the input and reports every row whose labels changed. Rows
generated with `--lean none` or through an API carry raw output only; they are scored here for the first time.
Entrypoint: `entrypoints/rescore.py`.
"""
import json

from .outcomes import score, summarize
from .tasks import task_index

COMPARE = ('reward', 'grader_reward', 'genuine', 'exploit', 'verified_refutation', 'report_label', 'format_valid')
CARRIED = ('tokens', 'seconds', 'truncated', 'round', 'step', 'turn', 'conversation', 'reasoning_tokens', 'model')


def rescore_file(path, lean, tasks=None):
    tasks = tasks or task_index()
    old_rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    new_rows, changed = [], []
    for old in old_rows:
        # A judged row keeps its stored reading: rescoring re-runs Lean, not the judge.
        replay = (lambda row, old=old: {'claim': old['claim'], 'raw': old.get('judge_raw', '')}) if old.get('claim') is not None else None
        new = score(lean, tasks[old['task_id']], old['raw'], old['arm'], old.get('penalty', 0.0), replay,
                    truncated=old.get('truncated'))
        new.update({k: old[k] for k in CARRIED if k in old})
        new_rows.append(new)
        diff = {k: (old.get(k), new[k]) for k in COMPARE if old.get(k) != new[k]}
        if diff and old.get('scored', True):
            changed.append({'task_id': old['task_id'], 'diff': diff})
    path.with_suffix('.rescored.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in new_rows))
    return {'file': str(path), 'rows': len(old_rows), 'changed': len(changed), 'changes': changed, 'summary': summarize(new_rows)}
