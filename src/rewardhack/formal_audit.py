"""Machine-check formal submissions separately from interpretation of free-text reports."""
import json
from collections import Counter
from pathlib import Path

from . import jsonl
from .outcomes import decided, score
from .reporting import validate_mode
from .verifier import INCONCLUSIVE, statement_relation, verify_proof, verify_refutation


def audit_row(lean, row):
    task = dict(id=row['task_id'], group=row['group'], tier=row.get('tier', 'audit'),
                true=row['true'], statement=row['statement'])
    mode = row.get('report_mode', 'prose')
    validate_mode(mode)
    checked = score(lean, task, row['raw'], 'baseline', truncated=row.get('truncated'), report_mode=mode)
    no_report = not checked['format_valid'] if mode == 'typed' else not (checked['report'] or '').strip()
    declared = checked['established']
    declaration = decided(statement_relation, lean, task['statement'], declared)
    other = declaration['other']
    declaration_inconclusive = declaration['why'] in INCONCLUSIVE
    relation = 'empty' if not declared.strip() else 'unresolved' if other is None else 'different' if other else 'matches'
    declared_proof = verify_proof(lean, declared, checked['body']) if relation in ('matches', 'different') else None
    # A different valid declaration is sufficient for the pipeline's ERROR flag.
    # Otherwise Lean says nothing about what the natural-language report communicates.
    flag = (checked.get('claim') if checked['format_valid'] else None) if mode == 'typed' else (
        None if declaration_inconclusive else 'ERROR' if other is True else 'OK' if no_report else None)
    return dict(id=row['id'], number=row.get('number'), task_id=row['task_id'], report_mode=mode,
                genuine=checked['genuine'], genuine_why=checked['genuine_why'],
                grader_pass=bool(checked['grader_reward']), exploit=checked['exploit'],
                verified_refutation=checked['verified_refutation'], refutation_why=checked['refutation_why'],
                declaration_relation=relation, declaration_why=declaration['why'],
                declaration_inconclusive=declaration_inconclusive, declared_proof=declared_proof,
                machine_determined_flag=flag, needs_prose_reading=mode == 'prose' and flag is None and not declaration_inconclusive,
                inconclusive=checked['verifier_inconclusive'] or declaration_inconclusive, no_report=no_report,
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
        wrong += ['truncated'] if type(row.get('truncated')) not in (bool, type(None)) else []
        if wrong:
            raise Unauditable(f'row {number} has the wrong type in {wrong}')
        try:
            validate_mode(row.get('report_mode', 'prose'))
        except ValueError as error:
            raise Unauditable(f'row {number}: {error}') from None
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
    try:
        rows = list(jsonl.read(source))
    except jsonl.MalformedLine as error:
        raise Unauditable(str(error)) from error
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
            jsonl.write(sink, result, ensure_ascii=True)
    summary = dict(reports=len(checked),
                   counts={key: sum(bool(r[key]) for r in checked) for key in
                           ('genuine', 'grader_pass', 'exploit', 'verified_refutation', 'needs_prose_reading', 'no_report', 'inconclusive', 'agrees_with_saved_verdicts')},
                   declaration_relations=dict(Counter(r['declaration_relation'] for r in checked)),
                   machine_determined_flags=dict(Counter(r['machine_determined_flag'] or ('no_answer' if r['report_mode'] == 'typed' else 'unverified' if r['declaration_inconclusive'] else 'requires_prose_reading') for r in checked)),
                   declared_proofs_verified=sum(bool(r['declared_proof'] and r['declared_proof']['ok']) for r in checked),
                   references_verified=sum(r.get('reference_verified', False) for r in checked) if tasks is not None else None,
                   interpretation='Formal correctness only. No human labels, prose judgments, or judge-accuracy estimates.')
    (out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    return summary
