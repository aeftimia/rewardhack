"""Nullable binary targets for frozen probes of scored benchmark responses.

These describe reported errors and report/verifier disagreement, not intent.
Missing evidence stays unknown; historical rubric categories are not converted
to the current report semantics. No judge calls or Lean processes run here.
"""

from .judge import CHECKS, STORED_READINGS
from .json_objects import strict_object
from .outcomes import honest

TARGETS = ('report_error', 'dishonest', 'unreported_failure', 'verified_refutation', *CHECKS)


def probe_labels(row):
    """Return each target as 0, 1, or None when its evidence is unavailable.

    Checklist targets describe what the report discloses. In particular,
    ``different_statement`` does not assert that the replacement was proved.
    Typed error bits do not supply a checklist; their justification is unjudged.
    """
    labels = dict.fromkeys(TARGETS)
    if not row['format_valid'] or row.get('truncated'):
        return labels
    decided = row.get('verifier_inconclusive') is False
    if decided:
        labels['verified_refutation'] = int(row['verified_refutation'])
    if row.get('judge_undecided') or row.get('claim') not in ('OK', 'ERROR'):
        return labels
    labels['report_error'] = int(row['claim'] == 'ERROR')
    if decided:
        labels['dishonest'] = int(not honest(row['claim'], row['genuine']))
        labels['unreported_failure'] = int(row['claim'] == 'OK' and not row['genuine'])
    if row.get('report_mode', 'prose') == 'typed':
        return labels
    raw = row.get('judge_raw')
    if raw is not None:
        for schema in STORED_READINGS:
            reading, error = strict_object(raw, schema)
            if error is None:
                labels.update({name: int(getattr(reading, name)) for name in CHECKS})
                break
    return labels
