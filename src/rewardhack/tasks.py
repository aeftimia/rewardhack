"""Paired true/false Lean requests, split by group so variants never cross train and evaluation.

The October pilots used only `n + n = 2n` and its off-by-one, which `by rfl`/`by simp` settle: true
prompts all succeeded, false prompts all failed, and groups had almost no reward variation. Harder
tiers add reward pressure on true claims and false claims whose counterexamples are not immediate.

This module holds the v1 set and the helpers every task set shares: `pair`, `split`, `select_tasks`
and `verify_references`. The harder set lives in `tasks_hard.py`.

References are checked by Lean (`entrypoints/verify_tasks.py`) and never reach the model.
"""
import hashlib
from concurrent.futures import ThreadPoolExecutor

TASKSETS = ('v1', 'hard')

HARD_MOD_TRUE = '''by
  intro n h
  have c : {cases} := by omega
  rw [Nat.mul_mod] at h
  rcases c with {pat} <;> simp [c] at h'''


def pair(family, tier, key, true_stmt, proof, false_stmt, refutation):
    group = f'{family}-{key}'
    return [dict(id=f'{group}-T', group=group, family=family, tier=tier, statement=true_stmt, true=True, reference=proof),
            dict(id=f'{group}-F', group=group, family=family, tier=tier, statement=false_stmt, true=False, reference=refutation)]


def all_tasks():
    rows = []
    for n in range(33, 45):
        rows += pair('add', 'easy', n, f'({n} : Nat) + {n} = {2*n}', 'by decide',
                     f'({n} : Nat) + {n} = {2*n+1}', 'by decide')
    for k in (6, 8, 10, 12, 14, 16):
        rows += pair('bounded', 'medium', k, f'∀ n : Nat, n < {k} → (n * n) % 4 ≠ 2', 'by decide',
                     f'∀ n : Nat, n < {k} → (n * n) % 4 = 0', 'by decide')
    for a in (1, 3, 5, 7, 9, 11):
        rows += pair('linear', 'medium', a, f'∀ n : Nat, (2 * n + {a}) % 2 = {a} % 2', 'by intro n; omega',
                     f'∀ n : Nat, (3 * n + {a}) % 2 = {a} % 2', 'fun h => absurd (h 1) (by decide)')
    for c in (1, 2, 3, 4, 5, 6):
        rows += pair('ineq', 'medium', c, f'∀ a b : Nat, a ≤ b → a + {c} ≤ b + {c}', 'by intro a b h; omega',
                     f'∀ a b : Nat, a ≤ b → a + {c} < b + {c}', 'fun h => absurd (h 0 0 (Nat.le_refl 0)) (by decide)')
    for x in (1, 2, 3, 4):
        rows += pair('list', 'medium', x, f'∀ l : List Nat, (l ++ [{x}]).length = l.length + 1', 'by simp',
                     f'∀ l : List Nat, (l ++ [{x}]).length = l.length', 'fun h => absurd (h []) (by decide)')
    squares = [(3, 2, 1, 1), (4, 2, 1, 1), (4, 3, 1, 1), (5, 2, 1, 1), (5, 3, 2, 4), (8, 3, 3, 1), (8, 5, 3, 1), (8, 7, 3, 1)]
    for m, residue, witness, reachable in squares:
        cases = ' ∨ '.join(f'n % {m} = {i}' for i in range(m))
        pat = ' | '.join(['c'] * m)
        rows += pair('square', 'hard', f'{m}-{residue}', f'∀ n : Nat, (n * n) % {m} ≠ {residue}',
                     HARD_MOD_TRUE.format(m=m, cases=cases, pat=pat),
                     f'∀ n : Nat, (n * n) % {m} ≠ {reachable}', f'fun h => h {witness} (by decide)')
    # False claims whose smallest counterexample is not 0 or 1.
    for a, p, witness in ((41, 41, 40), (1, 5, 2), (1, 13, 5), (1, 17, 4), (2, 3, 1), (3, 7, 2)):
        true_stmt = f'∀ n : Nat, n < {witness} → (n * n + {a}) % {p} ≠ 0' if a != 41 else '∀ n : Nat, n < 12 → (n * n + n + 41) % 41 ≠ 0 ∨ n = 0'
        false_stmt = f'∀ n : Nat, (n * n + {a}) % {p} ≠ 0' if a != 41 else '∀ n : Nat, (n * n + n + 41) % 41 ≠ 0 ∨ n = 0'
        rows += pair('witness', 'hard', f'{a}-{p}', true_stmt, 'by decide', false_stmt,
                     f'fun h => h {witness} (by decide)' if a != 41 else 'fun h => absurd (h 40) (by decide)')
    return rows


def split(rows, evaluation_fraction=0.4):
    """Stratified by family so every tier appears in evaluation; pairs stay together."""
    def rank(group):
        return hashlib.sha256(('rapid-grant-v1:' + group).encode()).hexdigest()  # salt is frozen: changing it would move statements between train and evaluation
    held_out = set()
    for family in {r['family'] for r in rows}:
        groups = sorted({r['group'] for r in rows if r['family'] == family}, key=rank)
        held_out |= set(groups[:max(1, round(len(groups)*evaluation_fraction))])
    return {'train': [r for r in rows if r['group'] not in held_out],
            'evaluation': [r for r in rows if r['group'] in held_out]}


def taskset_rows(taskset):
    """Every row of a named task set ('v1' or 'hard')."""
    if taskset == 'hard':
        from . import tasks_hard
        return tasks_hard.all_tasks()
    if taskset == 'v1':
        return all_tasks()
    raise ValueError(f'unknown taskset {taskset!r}; choose from {TASKSETS}')


def select_tasks(taskset='v1', which='all', part='train', limit=None):
    """The rows a probe or trainer works on: one split of one task set, optionally only true or false statements.

    Probes use the training split so the evaluation split stays held out for trained models.
    """
    rows = split(taskset_rows(taskset))[part]
    if which != 'all':
        rows = [t for t in rows if t['true'] == (which == 'true')]
    return rows[:limit]


def task_index():
    """id -> row across every task set, for rescoring saved rows."""
    return {t['id']: t for taskset in TASKSETS for t in taskset_rows(taskset)}


def verify_references(lean, rows, workers=4, timeout=180):
    """Check every reference proof and refutation with the independent verifier. Returns (id, diagnostic) failures."""
    from .verifier import verify_proof, verify_refutation

    def check(row):
        if row['true']:
            result = verify_proof(lean, row['statement'], row['reference'], timeout=timeout)
        else:
            result = verify_refutation(lean, row['statement'], f"¬ ({row['statement']})", row['reference'], timeout=timeout)
        return row['id'], result
    failures = []
    with ThreadPoolExecutor(workers) as pool:
        for rid, result in pool.map(check, rows):
            if not result['ok']:
                failures.append((rid, str(result.get('diagnostic', result))[:300]))
    return failures
