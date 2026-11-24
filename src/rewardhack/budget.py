"""Spending ledger and pre-launch guard for the grant budget.

The grant's budget lines are the caps. A ledger row is charged to a line through its `line` column; rows
with an empty `line` (the pilot, and anything paid from other funds) are recorded but count against
nothing here. The guard counts each launch at its worst case: hourly price x hard runtime cap + other
costs. Moving money between lines is a budget change and is not something this module does quietly.
"""
import csv
import math
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


def nonnegative_finite(value, name):
    """Validate costs and durations before they can weaken a comparison with a cap."""
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f'{name} must be a finite, nonnegative number') from None
    if not math.isfinite(number) or number < 0:
        raise ValueError(f'{name} must be a finite, nonnegative number')
    return number


def spent(rows, **match):
    total = sum(nonnegative_finite(r.get('total_usd'), f'ledger row {i} total_usd')
                for i, r in enumerate(rows, 1) if all(r.get(k) == v for k, v in match.items()))
    return nonnegative_finite(total, 'ledger total')


def grant_spent(rows):
    return nonnegative_finite(sum(spent(rows, line=line) for line in LINES), 'grant total')


def check(rows, line, hourly, hours, other=0.0, launch_cap=None):
    """Worst-case cost of a launch and the reasons to refuse it. `launch_cap` is the ceiling approved for this one launch."""
    hourly = nonnegative_finite(hourly, 'hourly price')
    hours = nonnegative_finite(hours, 'maximum hours')
    other = nonnegative_finite(other, 'other costs')
    if launch_cap is not None:
        launch_cap = nonnegative_finite(launch_cap, 'launch cap')
    worst = nonnegative_finite(hourly*hours+other, 'worst-case cost')
    problems = []
    if line not in LINES:
        return worst, [f'unknown budget line {line!r}; choose from {sorted(LINES)}']
    if launch_cap is not None and worst > launch_cap:
        problems.append(f'worst case ${worst:.2f} exceeds the ${launch_cap:.2f} approved for this launch')
    line_spent, total_spent = spent(rows, line=line), grant_spent(rows)
    if line_spent+worst > LINES[line]:
        problems.append(f'{line} ${line_spent:.2f}+${worst:.2f} exceeds ${LINES[line]:.2f}')
    if total_spent+worst > TOTAL_CAP:
        problems.append(f'grant total ${total_spent:.2f}+${worst:.2f} exceeds ${TOTAL_CAP:.2f}')
    return worst, problems
