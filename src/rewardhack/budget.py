"""Spending ledger and pre-launch guard for the grant budget.

The grant's budget lines are the caps. A ledger row is charged to a line through its `line` column; rows
with an empty `line` (the pilot, and anything paid from other funds) are recorded but count against
nothing here. The guard counts each launch at its worst case: hourly price x hard runtime cap + other
costs. Moving money between lines is a budget change and is not something this module does quietly.
Money a provider returns is its own row with a negative total and a `status` that starts with `refund`, on the line it was charged to.

Dollars are exact: ledger text, registry numbers and prices are read straight into `Decimal`, every sum and
comparison with a cap is made in `Decimal`, and amounts become floats only where they leave this module
(the values the public functions return, and the JSON registry).
"""
import argparse
import csv
from decimal import Decimal, InvalidOperation, ROUND_CEILING, ROUND_HALF_EVEN, localcontext, Context, Overflow, DivisionByZero
import functools
import math
import re
import sys
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, PlainSerializer, model_validator

from .state_files import Record, atomic_json, file_lock, load_json, validated

LEDGER = Path('spending/ledger.csv')  # relative to the project root; entrypoints pass an explicit path
FIELDS = ['date', 'provider', 'model', 'purpose', 'resource', 'hours', 'hourly_usd', 'other_usd', 'total_usd', 'status', 'evidence', 'line']
# The application's budget (docs/funding-application-draft.md), in USD.
LINES = {'gpt-oss-20b-gpu': 800.0, 'qwen3-32b-gpu': 900.0, 'api': 300.0, 'storage': 100.0, 'contingency': 400.0}
TOTAL_CAP = sum(LINES.values())


# --- Bare numbers: the one shared check for a finite number given to a function or an option ---------------

NUMBER_TYPES = (int, float, Decimal)  # bool is an int subclass and is refused by name below
SIGNS = {  # (positive, nonnegative) -> what the number must be, and the test
    (False, False): ('a finite number', lambda number: True),
    (True, False): ('a finite, positive number', lambda number: number > 0),
    (False, True): ('a finite, nonnegative number', lambda number: number >= 0),
}


def finite(value, name, *, positive=False, nonnegative=False):
    """`value` as a float when it is a finite int, float or Decimal of the asked sign; bool and text are not numbers."""
    wanted, test = SIGNS[positive, nonnegative]
    # math.isfinite reads a Decimal as a float, so one beyond float range is refused too.
    if isinstance(value, bool) or not isinstance(value, NUMBER_TYPES) or not math.isfinite(value) or not test(value):
        raise ValueError(f'{name} must be {wanted}')
    return float(value)


def nonnegative_finite(value, name):
    """Validate costs and durations before they can weaken a comparison with a cap."""
    return finite(value, name, nonnegative=True)


def number_text(text):
    """An option's text read as an exact Decimal (argparse `type`); its range is checked where it is used."""
    try:
        return Decimal(text)
    except InvalidOperation:
        raise argparse.ArgumentTypeError(f'{text!r} is not a number') from None


# --- Dollars --------------------------------------------------------------------------------------------------

MICRO = Decimal('0.000001')  # the ledger's unit
NOISE = Decimal('1e-12')  # binary representation error a float price may carry (1.15*100 is 114.99999999999999)
MAX_USD = Decimal(sys.float_info.max)  # an amount must survive conversion to a float where it leaves this module
# Enough digits that any sum of amounts within float range is exact; an invalid operation or overflow raises.
EXACT = Context(prec=2000, traps=[InvalidOperation, Overflow, DivisionByZero])
PARSE_DOLLARS = {  # an amount's type -> its exact decimal value
    str: Decimal,
    int: Decimal,
    float: lambda value: Decimal(repr(value)),  # the shortest decimal that is this float: the price as it was typed
    Decimal: lambda value: value,
}


def exact(function):
    """Run `function` with every Decimal operation exact."""
    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        with localcontext(EXACT):
            return function(*args, **kwargs)
    return wrapper


@exact
def dollars(value, name, *, refund=False):
    """An amount as an exact Decimal; never through a float. A refund amount is at most zero, any other at least zero."""
    sign = 'a finite number no greater than zero' if refund else 'a finite, nonnegative number'
    if type(value) not in PARSE_DOLLARS:
        raise ValueError(f'{name} must be {sign}')
    try:
        amount = PARSE_DOLLARS[type(value)](value)
    except InvalidOperation:
        raise ValueError(f'{name} must be {sign}') from None
    if not amount.is_finite() or abs(amount) > MAX_USD or (amount > 0 if refund else amount < 0):
        raise ValueError(f'{name} must be {sign}')
    return amount


@exact
def micro(amount, *, up=False):
    """The one place amounts meet the ledger's unit. To the nearest microdollar, or `up` for a debit.

    Rounding up keeps every positive cost positive. A positive six-place amount within NOISE absorbs
    binary representation error (0.1+0.2), and can never turn a positive charge into a free one.
    """
    nearest = amount.quantize(MICRO, rounding=ROUND_HALF_EVEN)
    if not up:
        return nearest
    if nearest > 0 and abs(amount-nearest) <= NOISE:
        return nearest
    return amount.quantize(MICRO, rounding=ROUND_CEILING)


def shown(amount):
    """An exact amount where it leaves this module (a returned value, a JSON registry): a float."""
    return float(amount)


Usd = Annotated[Decimal, Field(ge=0, allow_inf_nan=False), PlainSerializer(shown, return_type=float)]


# --- The reservation registry ---------------------------------------------------------------------------------

def reservation_path(ledger=LEDGER):
    """Canonical adjacent registry shared by every grant-budget consumer."""
    ledger = Path(ledger).resolve()
    return ledger.with_name(ledger.name + '.pilot-reservations.json')


RESERVATION_ENV = 'REWARDHACK_PILOT_RESERVATION'  # set by the pilot runner for a stage's commands
RESERVATION_STATUS = 'pilot worst-case reservation'


class Reservation(Record):
    """One stage's worst-case commitment and the launches claimed against it so far (never more than it)."""
    key: str
    line: Literal[tuple(LINES)]
    total_usd: Usd
    status: Literal[RESERVATION_STATUS]
    approved_usd: Usd = Decimal(0)  # nothing claimed yet; the field is written with the first claim, never as null

    @model_validator(mode='after')
    def claims_within_reservation(self):
        if micro(self.approved_usd) > micro(self.total_usd, up=True):  # the capacity `claim` allows
            raise ValueError('approved_usd exceeds the reservation total_usd')
        return self


def load_reservations(ledger):
    """The registry as validated `Reservation`s; JSON numbers read as exact decimals."""
    path = reservation_path(ledger)
    if not path.exists():
        return []
    records = load_json(path.read_text(), parse_float=Decimal, parse_int=Decimal)
    if not isinstance(records, list):
        raise ValueError('invalid pilot reservation registry: not a list')
    reservations = [validated(Reservation, record, f'invalid pilot reservation registry record {number}')
                    for number, record in enumerate(records, 1)]
    keys = [reservation.key for reservation in reservations]
    if len(set(keys)) != len(keys):
        raise ValueError('invalid pilot reservation registry: a key appears twice')
    _grant_spent([row(reservation) for reservation in reservations])
    return reservations


def row(reservation):
    """A reservation as the dict rows the ledger functions read (amounts as floats, `approved_usd` only once claimed)."""
    return reservation.model_dump(mode='json', exclude_unset=True)


def read_reservations(ledger=LEDGER):
    """Read validated worst-case commitments; they never become negative credits."""
    return [row(reservation) for reservation in load_reservations(ledger)]


# --- The ledger -----------------------------------------------------------------------------------------------

REFUND = 'refund'  # first word of the `status` of a row whose total may be negative: money a provider returned


def is_refund(row):
    """`status` is free text ("completed: $0.07 metered ..."), so a refund is marked by its first word."""
    return 'status' in row and isinstance(row['status'], str) and re.match(r'refund\b', row['status'].strip().lower()) is not None


class LedgerRow(Record):
    """A CSV row. A row with a missing or extra field, or a total that its own hours and price contradict, is not a basis for a launch.

    A row one comma short reads its last field, the budget line, as missing, and then counts against nothing.
    The total is taken from the provider's balance, so it may differ a little from hours x price: 15 cents or 5%.
    The amounts themselves are checked where they are summed (`row_total`).
    """
    date: str | None = None
    provider: str | None = None
    model: str | None = None
    purpose: str | None = None
    resource: str | None = None
    hours: str | None = None
    hourly_usd: str | None = None
    other_usd: str | None = None
    total_usd: str
    status: str
    evidence: str | None = None
    line: str

    @model_validator(mode='before')
    @classmethod
    def one_value_per_header_field(cls, values):
        if None in values or any(value is None for value in values.values()):
            raise ValueError('more or fewer fields than the header')
        return values

    @model_validator(mode='after')
    def total_matches_its_parts(self):
        if is_refund(self.model_dump()) or None in (self.hours, self.hourly_usd, self.other_usd):
            return self
        with localcontext(EXACT):
            parts = dollars(self.hours, 'hours')*dollars(self.hourly_usd, 'hourly_usd') + dollars(self.other_usd, 'other_usd')
            total = dollars(self.total_usd, 'total_usd')
            if abs(parts-total) > max(Decimal('0.15'), Decimal('0.05')*total):
                raise ValueError(f'hours x hourly_usd + other_usd is ${parts:.2f} but total_usd is ${total:.2f}')
        return self


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
    for number, values in enumerate(rows, 1):
        validated(LedgerRow, values, f'ledger row {number}')
    _grant_spent(rows)  # refunds must be justified by actual charges, never by reservations
    return rows + (read_reservations(path) if include_reservations else [])


def row_total(row, number):
    """A row's cost in exact dollars. Every row is validated, whichever line is asked for: a malformed ledger is not a basis for a launch.

    A negative total lowers what counts as spent, so it has to say what it is: `status` must start with `refund`.
    """
    if 'line' in row and row['line'] not in (None, '') and row['line'] not in LINES:
        raise ValueError(f"ledger row {number} has unknown budget line {row['line']!r}")
    name = f'ledger row {number} total_usd'
    if 'total_usd' not in row:
        raise ValueError(f'{name} is missing')
    if is_refund(row):
        return dollars(row['total_usd'], f'{name} in a {REFUND} row', refund=True)
    try:
        return dollars(row['total_usd'], name)
    except ValueError:
        raise ValueError(f'{name} must be a finite, nonnegative number (a refund is a negative total '
                         f'on a row whose status starts with "{REFUND}")') from None


@exact
def _spent(rows, **match):
    totals = [row_total(r, i) for i, r in enumerate(rows, 1)]
    total = sum((t for t, r in zip(totals, rows) if all(k in r and r[k] == v for k, v in match.items())), Decimal(0))
    if abs(total) > MAX_USD:
        raise ValueError('ledger total must be a finite, nonnegative number')
    total = micro(total)
    if total <= Decimal('-0.005'):
        where = ' on '+', '.join(f'{k} {v!r}' for k, v in match.items()) if match else ''
        raise ValueError(f'refunds exceed charges{where} by ${-total:.2f}; book each refund to the line of its charge')
    return max(total, Decimal(0))  # a refund of the metered amount against a charge booked to the cent is not an excess


@exact
def _grant_spent(rows):
    return sum((_spent(rows, line=line) for line in LINES), Decimal(0))


def spent(rows, **match):
    """Dollars charged or reserved on the rows that match, as a float."""
    return shown(_spent(rows, **match))


def grant_spent(rows):
    return shown(_grant_spent(rows))


# --- Launch checks and claims ---------------------------------------------------------------------------------

@exact
def worst_case(hourly, hours, other):
    """hourly x hours + other, exactly."""
    cost = dollars(hourly, 'hourly price')*dollars(hours, 'maximum hours') + dollars(other, 'other costs')
    if cost > MAX_USD:
        raise ValueError('worst-case cost must be a finite, nonnegative number')
    return cost


@exact
def _check(rows, line, worst, launch_cap):
    """Reasons to refuse a launch whose worst case is `worst` (exact dollars at the ledger's unit)."""
    if line not in LINES:
        return [f'unknown budget line {line!r}; choose from {sorted(LINES)}']
    problems = []
    if launch_cap is not None and worst > launch_cap:
        problems.append(f'worst case ${worst:.2f} exceeds the ${launch_cap:.2f} approved for this launch')
    line_spent, total_spent = _spent(rows, line=line), _grant_spent(rows)
    line_cap, total_cap = dollars(LINES[line], 'line cap'), dollars(TOTAL_CAP, 'grant cap')
    if line_spent+worst > line_cap:
        problems.append(f'{line} ${line_spent:.2f}+${worst:.2f} exceeds ${line_cap:.2f}')
    if total_spent+worst > total_cap:
        problems.append(f'grant total ${total_spent:.2f}+${worst:.2f} exceeds ${total_cap:.2f}')
    return problems


def check(rows, line, hourly, hours, other=0.0, launch_cap=None):
    """Worst-case cost of a launch and the reasons to refuse it. `launch_cap` is the ceiling approved for this one launch.

    Dollars at the ledger's precision, as in `spent`: a launch that exactly fills a cap is allowed.
    """
    cap = None if launch_cap is None else micro(dollars(launch_cap, 'launch cap'))  # same precision as the worst case it bounds
    worst = micro(worst_case(hourly, hours, other))
    return shown(worst), _check(rows, line, worst, cap)


REGISTRY_WAIT = 30.0  # seconds to wait for the reservation registry, which is held for one short update at a time


def claim_cost(hourly, hours, other=0.0):
    """Debit every positive cost, rounding up at the ledger's one-microdollar unit."""
    return shown(micro(worst_case(hourly, hours, other), up=True))


@exact
def claim(ledger, key, line, hourly, hours, other=0.0, launch_cap=None):
    """Atomically consume a stage's reserved allowance for one bounded launch.

    Every successful call consumes its worst-case cost before the caller launches
    anything. Claims survive interruption; actual charges and refunds never reset
    them. The full stage reservation remains visible to other budget consumers.
    Ordinary `check` is read-only and cannot perform this authorization.
    """
    ledger = Path(ledger).resolve()
    if not ledger.is_file():
        raise ValueError('the grant ledger must exist before claiming')
    if not isinstance(key, str) or not key:
        raise ValueError(f'claim requires the runner-provided {RESERVATION_ENV}')
    debit = micro(worst_case(hourly, hours, other), up=True)
    # Limits go through the same ceiling as the debit: a claim equal to its limit (10/3 dollars) must fit it.
    cap = None if launch_cap is None else micro(dollars(launch_cap, 'launch cap'), up=True)
    registry = reservation_path(ledger)
    with file_lock(registry.with_suffix(registry.suffix + '.lock'), wait=REGISTRY_WAIT):
        reservations = load_reservations(ledger)
        own = [reservation for reservation in reservations if reservation.key == key]
        if not own or own[0].line != line:
            raise ValueError('claim requires the matching stage reservation and grant line')
        own = own[0]
        actual = read_ledger(ledger, include_reservations=False)
        problems = _check(actual + [row(r) for r in reservations if r is not own], line, debit, cap)
        # Existing commitments must still fit the grant, including approvals not
        # yet booked as charges. No temporary discount of the full reservation.
        committed_problems = _check(actual + [row(r) for r in reservations], line, Decimal(0), None)
        problems.extend(problem for problem in committed_problems if problem not in problems)
        approved = micro(own.approved_usd)
        capacity = micro(own.total_usd, up=True)
        total = approved + debit
        if total > capacity:
            problems.append(f'cumulative launch claims ${total:.6f} exceed the ${capacity:.6f} stage reservation')
        if not problems:
            own.approved_usd = total
            atomic_json(registry, [row(r) for r in reservations], exclusive=False)
        now = approved if problems else total
        return dict(worst_usd=shown(debit), approved_usd=shown(now), remaining_usd=shown(capacity-now), problems=problems)
