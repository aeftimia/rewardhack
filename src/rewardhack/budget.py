"""Spending ledger and pre-launch guard for the grant budget.

The grant's budget lines are the caps. A ledger row is charged to a line through its `line` column; rows
with an empty `line` (the pilot, and anything paid from other funds) are recorded but count against
nothing here. The guard counts each launch at its worst case: hourly price x hard runtime cap + other
costs. Moving money between lines is a budget change and is not something this module does quietly.
Money a provider returns is its own row with a negative total and a `status` that starts with `refund`, on the line it was charged to.
"""
import csv
from decimal import Decimal, ROUND_CEILING
import json
import math
import re
from pathlib import Path

from .json_objects import unique_object
from .state_files import atomic_json, file_lock

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
        if (not isinstance(record, dict)
                or not {'key', 'line', 'total_usd', 'status'} <= record.keys()
                or record.keys() - {'key', 'line', 'total_usd', 'status', 'approved_usd'}
                or not isinstance(record['key'], str) or record['key'] in keys
                or record['status'] != 'pilot worst-case reservation'
                or not isinstance(record['line'], str) or record['line'] not in LINES
                or type(record['total_usd']) not in (int, float)):
            raise ValueError('invalid pilot reservation registry record')
        nonnegative_finite(record['total_usd'], 'reservation total_usd')
        approved = record.get('approved_usd', 0.0)
        if (type(approved) not in (int, float) or not math.isfinite(approved) or approved < 0
                or not math.isfinite(record['total_usd']) or round(approved, 6) > round(record['total_usd'], 6)):
            raise ValueError('invalid pilot reservation approved_usd')
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


def claim_cost(hourly, hours, other=0.0):
    """Debit every positive cost, rounding up at the ledger's one-microdollar unit.

    Decimal arithmetic preserves the declared decimal prices. A positive six-place
    result within 1e-12 dollars absorbs binary representation noise (0.1+0.2);
    that normalization can never turn a positive charge into a free claim.
    """
    values = []
    for value, name in ((hourly, 'hourly price'), (hours, 'maximum hours'), (other, 'other costs')):
        if isinstance(value, bool):
            raise ValueError(f'{name} must be a finite, nonnegative number')
        nonnegative_finite(value, name)
        values.append(Decimal(str(value)))
    exact = values[0]*values[1]+values[2]
    nonnegative_finite(exact, 'worst-case cost')
    if exact > Decimal(str(TOTAL_CAP)):
        return float(exact)  # the grant guard refuses it; no quantization of enormous amounts
    unit = Decimal('0.000001')
    nearest = exact.quantize(unit)
    if nearest > 0 and abs(exact-nearest) <= Decimal('0.000000000001'):
        exact = nearest
    return float(exact.quantize(unit, rounding=ROUND_CEILING))


def claim(ledger, key, line, hourly, hours, other=0.0, launch_cap=None):
    """Atomically consume a stage's reserved allowance for one bounded launch.

    Every successful call consumes its worst-case cost before the caller launches
    anything. Claims survive interruption; actual charges and refunds never reset
    them. The full stage reservation remains visible to other budget consumers.
    Ordinary `check` is read-only and cannot perform this authorization.
    """
    if isinstance(launch_cap, bool):
        raise ValueError('launch cap must be a finite, nonnegative number')
    ledger = Path(ledger).resolve()
    if not ledger.is_file():
        raise ValueError('the grant ledger must exist before claiming')
    if not isinstance(key, str) or not key:
        raise ValueError(f'claim requires the runner-provided {RESERVATION_ENV}')
    registry = reservation_path(ledger)
    with file_lock(registry.with_suffix(registry.suffix + '.lock')):
        records = read_reservations(ledger)
        own = next((record for record in records if record['key'] == key), None)
        if own is None or own['line'] != line:
            raise ValueError('claim requires the matching stage reservation and grant line')
        actual = read_ledger(ledger, include_reservations=False)
        other_rows = actual + [record for record in records if record is not own]
        debit = claim_cost(hourly, hours, other)
        worst, problems = check(other_rows, line, 0, 0, debit, launch_cap)
        # Existing commitments must still fit the grant, including approvals not
        # yet booked as charges. No temporary discount of the full reservation.
        _, committed_problems = check(actual + records, line, 0, 0)
        problems.extend(problem for problem in committed_problems if problem not in problems)
        approved = round(own.get('approved_usd', 0.0), 6)
        capacity = round(own['total_usd'], 6)
        total = round(approved + worst, 6)
        if total > capacity:
            problems.append(f'cumulative launch claims ${total:.6f} exceed the ${capacity:.6f} stage reservation')
        if not problems:
            own['approved_usd'] = total
            atomic_json(registry, records, exclusive=False)
        return dict(worst_usd=worst, approved_usd=approved if problems else total,
                    remaining_usd=round(capacity-(approved if problems else total), 6), problems=problems)
