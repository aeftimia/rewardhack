"""Rescore saved rows from their raw model output with the current grader, verifier and labels.

Writes `<rows>.rescored.jsonl` next to the input and reports every row whose labels changed. Rows
generated with `--lean none` or through an API carry raw output only; they are scored here for the first time.
Entrypoint: `entrypoints/rescore.py`.
"""
from . import jsonl
from .outcomes import score, summarize
from .reporting import validate_mode
from .tasks import task_index
from .verifier import INCONCLUSIVE

COMPARE = ('reward', 'grader_reward', 'genuine', 'exploit', 'verified_refutation', 'report_label', 'format_valid', 'claim', 'honest', 'judge_undecided')
# `copied_from` marks a resumed run's copies of the original run's rows: the transcript export skips them.
CARRIED = ('sample_index', 'pilot_source', 'partial_batch', 'copied_from', 'termination_schema', 'tokens', 'seconds', 'truncated', 'round', 'step', 'turn', 'conversation', 'reasoning_tokens', 'model', 'meta')


def scored_as_cut_off(old):
    """Replay current cutoff semantics while preserving unversioned historical answers.

    Versioned rows honor their generation termination flag, including fresh unscored rows.
    Older unversioned rows carry a `truncated` flag from looser tests (`finish_reason == 'length'`, a stop token landing on
    the limit) next to a complete answer. That flag stays as metadata and the answer is rescored from `raw`.
    """
    if old.get('termination_schema') == 'cutoff-v1':
        return old.get('truncated')
    return True if old.get('truncated') and old.get('genuine_why') == 'truncated' else None


def stored_reading(old):
    """Replay a saved reading, including declaration uncertainty that was not rechecked."""
    verdict = {'claim': old['claim'], 'raw': old.get('judge_raw', '')}
    if old.get('judge_asked'):
        verdict['judge_asked'] = True
    if old.get('judge_undecided'):
        # The stored reading failed; replaying it as a plain UNPARSED claim would turn a judge outage into a dishonest report.
        verdict['judge_error'] = old['judge_error']
    if old.get('declaration_why') in INCONCLUSIVE:
        verdict.update(verifier_inconclusive=True, declaration_why=old['declaration_why'])
    return verdict


def rescore_file(path, lean, tasks=None):
    tasks = tasks or task_index()
    old_rows = list(jsonl.read(path))
    for old in old_rows:
        validate_mode(old.get('report_mode', 'prose'))
    new_rows, changed, transport_errors = [], [], []
    for number, old in enumerate(old_rows, 1):
        if 'raw' not in old:
            if 'error' not in old:
                raise ValueError(f'row {number} has neither raw output nor a transport error')
            transport_errors.append(dict(row=number, task_id=old.get('task_id'), error=old['error']))
            continue  # failed requests are not model answers and never enter behavior rates
        # A judged row keeps its stored reading: rescoring re-runs Lean, not the judge.
        mode = old.get('report_mode', 'prose')
        replay = (lambda row, old=old: stored_reading(old)) if mode == 'prose' and old.get('claim') is not None else None
        new = score(lean, tasks[old['task_id']], old['raw'], old['arm'], old.get('penalty', 0.0), replay,
                    truncated=scored_as_cut_off(old), report_mode=mode)
        new.update({k: old[k] for k in CARRIED if k in old})
        new_rows.append(new)
        diff = {k: (old.get(k), new.get(k)) for k in COMPARE if old.get(k) != new.get(k)}
        if diff and old.get('scored', True):
            changed.append({'task_id': old['task_id'], 'diff': diff})
    path.with_suffix('.rescored.jsonl').write_text(jsonl.dumps(new_rows))
    return {'file': str(path), 'input_rows': len(old_rows), 'rows': len(new_rows), 'transport_errors': transport_errors, 'changed': len(changed), 'changes': changed, 'summary': summarize(new_rows)}
