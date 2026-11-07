"""Spending ledger and pre-launch guard. Caps overlap inside the total.

Guard logic counts each launch at its worst case: hourly price x hard runtime cap + storage + overhead.
"""
import csv
from pathlib import Path

LEDGER = Path('spending/ledger.csv')  # relative to the project root; entrypoints pass an explicit path
FIELDS = ['date', 'provider', 'model', 'purpose', 'resource', 'hours', 'hourly_usd', 'other_usd', 'total_usd', 'status', 'evidence']
TOTAL_CAP = 50.0
MODEL_8B_CAP = 5.0


def read_ledger(path=LEDGER):
    if not path.exists():
        return []
    with path.open() as handle:
        return list(csv.DictReader(handle))


def spent(rows, **match):
    return sum(float(r['total_usd']) for r in rows if all(r[k] == v for k, v in match.items()))


def check(rows, provider, model, hourly, hours, other):
    worst = hourly*hours+other
    problems = []
    total = spent(rows)
    if total+worst > TOTAL_CAP:
        problems.append(f'total ${total:.2f}+${worst:.2f} exceeds ${TOTAL_CAP}')
    if model == '8B' and spent(rows, model='8B')+worst > MODEL_8B_CAP:
        problems.append(f"8B ${spent(rows, model='8B'):.2f}+${worst:.2f} exceeds ${MODEL_8B_CAP}")
    return worst, problems


