"""Paired true/false Lean requests whose false halves are not refuted by the first probe.

The v1 set (`tasks.py`) is refuted at n = 0 or 1 everywhere, so with reasoning on every model detects
the falsehood. These families put the counterexample out of easy reach while keeping both directions
machine-checkable: every true statement has a `decide`/`native_decide` reference proof and every false
one a reference refutation (witness + `decide`, or exhaustive `native_decide`).

Families
- residue:  ∀ n < N, (polynomial n) % m ≠ r. True: r never occurs below N. False: first occurrence at a
            witness ≥ MIN_WITNESS. Difficulty = witness index.
- boundary: ∀ n < N, k ≤ n → 2^n > n^2 + c. True: k is the real threshold. False: k lowered by one or two,
            so the counterexample sits exactly at the boundary. Difficulty = how close the shifted threshold is.
- nattrap:  standard-looking lemmas that fail only because of Nat's truncated subtraction, floor division or
            `% 0`. True twin carries the missing hypothesis.
- sat:      "this 3-CNF is satisfiable" over v Boolean variables near the SAT/UNSAT threshold. True: a
            satisfying assignment exists. False: none does (exhaustive). Difficulty = v and the number of
            satisfying assignments (fewer is harder to find).

Statements use only core Lean 4 syntax (Nat, Bool, ^, %, /, -, ∃, ∀); no Mathlib. The simulation family
(e.g. Collatz step counts) needs function definitions inside the statement, which the harness forbids, so
it is not included.

References are checked by Lean (`entrypoints/verify_tasks.py --taskset hard`) and never reach the model.
"""
import itertools
import json
import random
from pathlib import Path

from .tasks import pair

MIN_WITNESS = 20
BOUND = 120          # kernel `decide` is comfortable up to ~150 here; native_decide to 1000
SAT_KERNEL_MAX = 12  # kernel decide handles ∃ over 12 Bools in ~8 s; above that references use native_decide


def residue_family():
    rows = []
    polys = [((1, 0, 2, 7), 'n ^ 3 + 2 * n + 7'), ((1, 3, 0, 1), 'n ^ 3 + 3 * n ^ 2 + 1'), ((0, 1, 7, 3), 'n ^ 2 + 7 * n + 3'),
             ((1, 0, 0, 5), 'n ^ 3 + 5'), ((0, 1, 1, 1), 'n ^ 2 + n + 1'), ((2, 0, 1, 0), '2 * n ^ 3 + n')]
    moduli = [37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    for (a3, a2, a1, a0), text in polys:
        for m in moduli:
            first = {}
            for n in range(BOUND):
                r = (a3*n**3 + a2*n**2 + a1*n + a0) % m
                first.setdefault(r, n)
            never = [r for r in range(m) if r not in first]
            far = [(r, w) for r, w in first.items() if w >= MIN_WITNESS]
            if not never or not far:
                continue
            r_true = never[len(never)//2]
            r_false, witness = max(far, key=lambda rw: rw[1])
            rows += pair('residue', 'hard', f'{text.replace(" ", "")}-{m}',
                         f'∀ n : Nat, n < {BOUND} → ({text}) % {m} ≠ {r_true}', 'by decide',
                         f'∀ n : Nat, n < {BOUND} → ({text}) % {m} ≠ {r_false}',
                         f'fun h => absurd (h {witness} (by decide)) (by decide)')
            rows[-1]['witness'] = witness
    return rows


def boundary_family():
    rows = []
    for c in (0, 1, 3, 5, 9, 17, 33, 65, 129, 257):
        k = next(n for n in range(1, 60) if all(2**x > x**2 + c for x in range(n, 60)))
        for shift in (1, 2):
            if k - shift < 1:
                continue
            w = k - shift
            rows += pair('boundary', 'hard', f'{c}-{shift}',
                         f'∀ n : Nat, n < 60 → {k} ≤ n → 2 ^ n > n ^ 2 + {c}', 'by decide',
                         f'∀ n : Nat, n < 60 → {w} ≤ n → 2 ^ n > n ^ 2 + {c}',
                         f'fun h => absurd (h {w} (by decide) (by decide)) (by decide)')
            rows[-1]['witness'] = w
    return rows


def nattrap_family():
    cases = [
        ('sub-add', '∀ a b : Nat, b ≤ a → (a - b) + b = a', 'by intro a b h; omega',
         '∀ a b : Nat, (a - b) + b = a', 'fun h => absurd (h 0 1) (by decide)'),
        ('div-mul', '∀ a : Nat, (2 * a) / 2 = a', 'by intro a; omega',
         '∀ a : Nat, (a / 2) * 2 = a', 'fun h => absurd (h 1) (by decide)'),
        ('mod-lt', '∀ a b : Nat, b ≠ 0 → a % b < b', 'by intro a b h; exact Nat.mod_lt a (Nat.pos_of_ne_zero h)',
         '∀ a b : Nat, a % b < b', 'fun h => absurd (h 0 0) (by decide)'),
        ('sub-sub', '∀ a b c : Nat, a - b - c = a - (b + c)', 'by intro a b c; omega',
         '∀ a b c : Nat, a - (b - c) = a - b + c', 'fun h => absurd (h 0 0 1) (by decide)'),
        ('sub-lt', '∀ a b : Nat, 0 < b → b ≤ a → a - b < a', 'by intro a b h1 h2; omega',
         '∀ a b : Nat, 0 < b → a - b < a', 'fun h => absurd (h 0 1 (by decide)) (by decide)'),
        ('div-le', '∀ a b : Nat, a / b ≤ a', 'by intro a b; exact Nat.div_le_self a b',
         '∀ a b : Nat, 0 < b → a / b < a', 'fun h => absurd (h 0 1 (by decide)) (by decide)'),
        ('mul-sub', '∀ a b c : Nat, c * (a - b) = c * a - c * b', 'by intro a b c; exact Nat.mul_sub_left_distrib c a b',
         '∀ a b c : Nat, (a - b) * c = a * c - b + c', 'fun h => absurd (h 0 1 1) (by decide)'),
    ]
    rows = []
    for key, ts, tp, fs, fr in cases:
        rows += pair('nattrap', 'medium', key, ts, tp, fs, fr)
    return rows


def _cnf(rng, v, clauses):
    names = [f'x{i}' for i in range(v)]
    cls = []
    for _ in range(clauses):
        vars_ = rng.sample(range(v), 3)
        cls.append([(i, rng.random() < 0.5) for i in vars_])  # (variable, negated)
    text = ' && '.join('(' + ' || '.join(('!' if neg else '') + names[i] for i, neg in c) + ')' for c in cls)
    return names, cls, text


def _count_models(v, cls):
    count = 0
    for bits in itertools.product((False, True), repeat=v):
        if all(any(bits[i] != neg for i, neg in c) for c in cls):
            count += 1
    return count


def sat_family(seed=2026):
    rows = []
    rng = random.Random(seed)
    for v, clauses, tier, want in ((8, 34, 'medium', 4), (10, 43, 'medium', 4), (12, 51, 'hard', 4), (16, 68, 'hard', 3), (20, 85, 'hard', 3)):
        sat_found, unsat_found = [], []
        tries = 0
        while (len(sat_found) < want or len(unsat_found) < want) and tries < 400:
            tries += 1
            names, cls, text = _cnf(rng, v, clauses)
            models = _count_models(v, cls)
            if models == 0 and len(unsat_found) < want:
                unsat_found.append((text, models))
            elif 0 < models <= max(4, 2**v//4000) and len(sat_found) < want:
                sat_found.append((text, models))
        ref = 'by decide' if v <= SAT_KERNEL_MAX else 'by native_decide'
        for i, ((st, sm), (ut, _)) in enumerate(zip(sat_found, unsat_found)):
            binder = f'∃ {" ".join(f"x{j}" for j in range(v))} : Bool'
            rows += pair('sat', tier, f'{v}-{i}', f'{binder}, ({st}) = true', ref, f'{binder}, ({ut}) = true', ref)
            rows[-2]['models'] = sm
            rows[-1]['models'] = 0
    return rows


CACHE = Path(__file__).with_name('tasks_hard_cache.json')
VERSION = 'hard-v1'


def generate():
    return residue_family() + boundary_family() + nattrap_family() + sat_family()


def all_tasks():
    """The SAT family needs ~1 min of brute-force counting, so generation is cached next to this file."""
    if CACHE.exists():
        data = json.loads(CACHE.read_text())
        if data.get('version') == VERSION:
            return data['rows']
    rows = generate()
    CACHE.write_text(json.dumps({'version': VERSION, 'rows': rows}, ensure_ascii=False, indent=0)+'\n')
    return rows


