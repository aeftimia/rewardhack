"""Spending ledger and pre-launch guard for the grant budget.

The grant's budget lines are the caps. A ledger row is charged to a line through its `line` column; rows
with an empty `line` (the pilot, and anything paid from other funds) are recorded but count against
nothing here. The guard counts each launch at its worst case: hourly price x hard runtime cap + other
costs. Moving money between lines is a budget change and is not something this module does quietly.
"""
import csv
from pathlib import Path

LEDGER = Path('spending/ledger.csv')  # relative to the project root; entrypoints pass an explicit path
FIELDS = ['date', 'provider', 'model', 'purpose', 'resource', 'hours', 'hourly_usd', 'other_usd', 'total_usd', 'status', 'evidence', 'line']
# The application's budget (docs/funding-application-draft.md), in USD.
LINES = {'gpt-oss-20b-gpu': 800.0, 'qwen3-32b-gpu': 900.0, 'api': 300.0, 'storage': 100.0, 'contingency': 400.0}
TOTAL_CAP = sum(LINES.values())


def read_ledger(path=LEDGER):
    if not path.exists():
        return []
    with path.open() as handle:
        return list(csv.DictReader(handle))


def spent(rows, **match):
    return sum(float(r['total_usd']) for r in rows if all(r.get(k) == v for k, v in match.items()))


def grant_spent(rows):
    return sum(spent(rows, line=line) for line in LINES)


def check(rows, line, hourly, hours, other=0.0, launch_cap=None):
    """Worst-case cost of a launch and the reasons to refuse it. `launch_cap` is the ceiling approved for this one launch."""
    worst = hourly*hours+other
    problems = []
    if line not in LINES:
        return worst, [f'unknown budget line {line!r}; choose from {sorted(LINES)}']
    if launch_cap is not None and worst > launch_cap:
        problems.append(f'worst case ${worst:.2f} exceeds the ${launch_cap:.2f} approved for this launch')
    if spent(rows, line=line)+worst > LINES[line]:
        problems.append(f'{line} ${spent(rows, line=line):.2f}+${worst:.2f} exceeds ${LINES[line]:.2f}')
    if grant_spent(rows)+worst > TOTAL_CAP:
        problems.append(f'grant total ${grant_spent(rows):.2f}+${worst:.2f} exceeds ${TOTAL_CAP:.2f}')
    return worst, problems
