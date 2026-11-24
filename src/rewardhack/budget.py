"""Spending ledger and pre-launch guard for the grant budget.

The grant's budget lines are the caps. A ledger row is charged to a line through its `line` column; rows
with an empty `line` (the pilot, and anything paid from other funds) are recorded but count against
nothing here. The guard counts each launch at its worst case: hourly price x hard runtime cap + other
costs. Moving money between lines is a budget change and is not something this module does quietly.
Money a provider returns is its own row with a negative total and a `status` that starts with `refund`, on the line it was charged to.
"""
import csv
import json
import math
import re
from pathlib import Path

from .json_objects import unique_object

LEDGER = Path('spending/ledger.csv')  # relative to the project root; entrypoints pass an explicit path
FIELDS = ['date', 'provider', 'model', 'purpose', 'resource', 'hours', 'hourly_usd', 'other_usd', 'total_usd', 'status', 'evidence', 'line']
# The application's budget (docs/funding-application-draft.md), in USD.
LINES = {'gpt-oss-20b-gpu': 800.0, 'qwen3-32b-gpu': 900.0, 'api': 300.0, 'storage': 100.0, 'contingency': 400.0}
TOTAL_CAP = sum(LINES.values())


def reservation_path(ledger=LEDGER):
    """Canonical adjacent registry shared by every grant-budget consumer."""
    ledger = Path(ledger).resolve()
    return ledger.with_name(ledger.name + '.pilot-reservations.json')


RESERVATION_ENV = 'REWARDHACK_PILOT_RESERVATION'  # set by the pilot runner for a stage's commands


def without_own_reservation(rows, key, line):
    """Rows for a guard run inside a pilot stage, and the dollars that stage already reserved on `line`.

    The stage's worst case is in the registry before its command starts. Counting the launch again on
    top of it would refuse any stage larger than half of what is left. Only the one reservation named
    by the runner is set aside, and only on its own line; the caller must hold the launch within it.
    The guard keeps no record of launches it has passed, here as anywhere else: a stage that makes
    several guarded launches must enter each charge in the ledger before the next check.
    """
    own = [r for r in rows if key and r.get('key') == key and r.get('line') == line]
    return [r for r in rows if not any(r is o for o in own)], (own[0]['total_usd'] if own else None)


def read_reservations(ledger=LEDGER):
    """Read validated worst-case commitments; they never become negative credits."""
    path = reservation_path(ledger)
    if not path.exists():
        return []
    records = json.loads(path.read_text(), object_pairs_hook=unique_object,
                         parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f'invalid JSON number {value}')))
    if not isinstance(records, list):
        raise ValueError('invalid pilot reservation registry')
    keys = set()
    for record in records:
        if (not isinstance(record, dict) or set(record) != {'key', 'line', 'total_usd', 'status'}
                or not isinstance(record['key'], str) or record['key'] in keys
                or record['status'] != 'pilot worst-case reservation'
                or not isinstance(record['line'], str) or record['line'] not in LINES
                or type(record['total_usd']) not in (int, float)):
            raise ValueError('invalid pilot reservation registry record')
        keys.add(record['key'])
    grant_spent(records)  # validates finite, nonnegative prices and totals
    return records


def read_ledger(path=LEDGER, *, include_reservations=True):
    """Actual ledger rows plus conservative pilot commitments, without writing CSV.

    Callers performing a reservation under its registry lock may explicitly read
    CSV alone and append their already-validated registry snapshot. Existing
    launch guards automatically see commitments from every pilot output folder.
    """
    path = Path(path).resolve()
    rows = []
    if path.exists():
        with path.open() as handle:
            rows = list(csv.DictReader(handle))
    grant_spent(rows)  # refunds must be justified by actual charges, never by reservations
    return rows + (read_reservations(path) if include_reservations else [])


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
        launch_cap = round(nonnegative_finite(launch_cap, 'launch cap'), 6)  # same precision as the worst case it bounds
    # Dollars at the ledger's precision, as in `spent`: 2.7*24 is not 64.8 in binary, and a launch that exactly fills a cap is allowed.
    worst = round(nonnegative_finite(hourly*hours+other, 'worst-case cost'), 6)
    problems = []
    if line not in LINES:
        return worst, [f'unknown budget line {line!r}; choose from {sorted(LINES)}']
    if launch_cap is not None and worst > launch_cap:
        problems.append(f'worst case ${worst:.2f} exceeds the ${launch_cap:.2f} approved for this launch')
    line_spent, total_spent = spent(rows, line=line), grant_spent(rows)
    if round(line_spent+worst, 6) > LINES[line]:
        problems.append(f'{line} ${line_spent:.2f}+${worst:.2f} exceeds ${LINES[line]:.2f}')
    if round(total_spent+worst, 6) > TOTAL_CAP:
        problems.append(f'grant total ${total_spent:.2f}+${worst:.2f} exceeds ${TOTAL_CAP:.2f}')
    return worst, problems
