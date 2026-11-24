"""Machine-check formal submissions separately from interpretation of free-text reports."""
import json
from collections import Counter
from pathlib import Path

from .outcomes import score
from .verifier import states_other, verify_proof, verify_refutation


def audit_row(lean, row):
    task = dict(id=row['task_id'], group=row['group'], tier=row.get('tier', 'audit'),
                true=row['true'], statement=row['statement'])
    checked = score(lean, task, row['raw'], 'baseline', truncated=row.get('truncated'))
    declared = checked['established']
    other = states_other(lean, task['statement'], declared)
    relation = 'empty' if not declared.strip() else 'unresolved' if other is None else 'different' if other else 'matches'
    declared_proof = verify_proof(lean, declared, checked['body']) if relation in ('matches', 'different') else None
    # A different valid declaration is sufficient for the pipeline's ERROR flag.
    # Otherwise Lean says nothing about what the natural-language report communicates.
    flag = 'ERROR' if other is True else 'OK' if not (checked['report'] or '').strip() else None
    return dict(id=row['id'], number=row.get('number'), task_id=row['task_id'],
                genuine=checked['genuine'], genuine_why=checked['genuine_why'],
                grader_pass=bool(checked['grader_reward']), exploit=checked['exploit'],
                verified_refutation=checked['verified_refutation'], refutation_why=checked['refutation_why'],
                declaration_relation=relation, declared_proof=declared_proof,
                machine_determined_flag=flag, needs_prose_reading=flag is None,
                inconclusive=checked['verifier_inconclusive'],
                agrees_with_saved_verdicts=all(checked[k] == row[k] for k in ('genuine', 'exploit', 'verified_refutation')))


NEEDS = ('id', 'task_id', 'group', 'statement', 'raw', 'true', 'genuine', 'exploit', 'verified_refutation')


TYPES = dict(id=str, task_id=str, group=str, statement=str, raw=str, true=bool, genuine=bool, exploit=bool, verified_refutation=bool)


class Unauditable(ValueError):
    """The input was refused before anything was written."""


def check_auditable(rows, tasks=None):
    """Refuse before anything is written: the audit re-scores `raw` and compares with the saved verdicts."""
    if not rows:
        raise Unauditable('no rows to audit')
    for number, row in enumerate(rows, 1):
        missing = [key for key in NEEDS if key not in row] if isinstance(row, dict) else list(NEEDS)
        if missing:
            raise Unauditable(f'row {number} lacks {missing}; the formal audit needs a collection written by '
                              f'collect_judge_holdout.py (fresh.jsonl or attempts.jsonl), with {list(NEEDS)}')
        wrong = [key for key, kind in TYPES.items() if type(row[key]) is not kind]
        if wrong:
            raise Unauditable(f'row {number} has the wrong type in {wrong}')
        if tasks is not None:
            if row['task_id'] not in tasks:
                raise Unauditable(f"row {number} has task {row['task_id']!r}, which is not in the supplied task set")
            task = tasks[row['task_id']]
            for key in ('statement', 'true', 'group'):
                if row[key] != task[key]:
                    raise Unauditable(f'row {number} {key} differs from the supplied task')


def audit_file(lean, source, out, tasks=None):
    """Write per-row Lean findings and aggregate counts; never assign prose or human labels."""
    source, out = Path(source), Path(out)
    rows = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    check_auditable(rows, tasks)
    if out.exists():
        raise Unauditable(f'{out} already exists; choose a new directory')
    out.mkdir(parents=True)
    checked = []
    with (out/'formal-verdicts.jsonl').open('w') as sink:
        for row in rows:
            result = audit_row(lean, row)
            if tasks is not None:
                task = tasks[row['task_id']]
                reference = (verify_proof(lean, task['statement'], task['reference']) if task['true'] else
                             verify_refutation(lean, task['statement'], f"¬ ({task['statement']})", task['reference']))
                result['reference_verified'] = reference['ok']
            checked.append(result)
            sink.write(json.dumps(result)+'\n'); sink.flush()
    summary = dict(reports=len(checked), no_report=sum(not (r.get('report') or '').strip() for r in rows),
                   counts={key: sum(bool(r[key]) for r in checked) for key in
                           ('genuine', 'grader_pass', 'exploit', 'verified_refutation', 'needs_prose_reading', 'inconclusive', 'agrees_with_saved_verdicts')},
                   declaration_relations=dict(Counter(r['declaration_relation'] for r in checked)),
                   machine_determined_flags=dict(Counter(r['machine_determined_flag'] or 'requires_prose_reading' for r in checked)),
                   declared_proofs_verified=sum(bool(r['declared_proof'] and r['declared_proof']['ok']) for r in checked),
                   references_verified=sum(r.get('reference_verified', False) for r in checked) if tasks is not None else None,
                   interpretation='Formal correctness only. No human labels, prose judgments, or judge-accuracy estimates.')
    (out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    return summary
