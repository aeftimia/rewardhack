"""Spending ledger and pre-launch guard for the grant budget.

The grant's budget lines are the caps. A ledger row is charged to a line through its `line` column; rows
with an empty `line` (the pilot, and anything paid from other funds) are recorded but count against
nothing here. The guard counts each launch at its worst case: hourly price x hard runtime cap + other
costs. Moving money between lines is a budget change and is not something this module does quietly.
Money a provider returns is its own row with a negative total and a `status` that starts with `refund`, on the line it was charged to.
"""
import csv
import math
import re
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


REFUND = 'refund'  # first word of the `status` of a row whose total may be negative: money a provider returned


def is_refund(row):
    """`status` is free text ("completed: $0.07 metered ..."), so a refund is marked by its first word."""
    return re.match(r'refund\b', (row.get('status') or '').strip().lower()) is not None


def row_total(row, number):
    """A row's cost in dollars. Every row is validated, whichever line is asked for: a malformed ledger is not a basis for a launch.

    A negative total lowers what counts as spent, so it has to say what it is: `status` must start with `refund`.
    """
    line = row.get('line')
    if line not in (None, '') and line not in LINES:
        raise ValueError(f'ledger row {number} has unknown budget line {line!r}')
    name = f'ledger row {number} total_usd'
    if not is_refund(row):
        try:
            return nonnegative_finite(row.get('total_usd'), name)
        except ValueError:
            raise ValueError(f'{name} must be a finite, nonnegative number (a refund is a negative total '
                             f'on a row whose status starts with "{REFUND}")') from None
    try:
        return -nonnegative_finite(-float(row.get('total_usd')), name)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f'{name} must be a finite number no greater than zero in a {REFUND} row') from None


def spent(rows, **match):
    totals = [row_total(r, i) for i, r in enumerate(rows, 1)]
    total = sum(t for t, r in zip(totals, rows) if all(r.get(k) == v for k, v in match.items()))
    if not math.isfinite(total):
        raise ValueError('ledger total must be a finite, nonnegative number')
    total = round(total, 6)  # ledger amounts are dollars with at most a few decimals; binary sums of them are not exact
    if total <= -0.005:
        where = ' on '+', '.join(f'{k} {v!r}' for k, v in match.items()) if match else ''
        raise ValueError(f'refunds exceed charges{where} by ${-total:.2f}; book each refund to the line of its charge')
    return total if total > 0 else 0.0  # a refund of the metered amount against a charge booked to the cent is not an excess


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
