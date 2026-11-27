"""Deterministic DRY candidates, not proofs that definitions should be fused.

No code is executed. Python uses ASTs; shell support deliberately recognizes only
lines containing scalar definitions. Reports describe coverage as well as findings.
"""
import ast
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import copy
from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
import re
import subprocess


MIN_CLONE_NODES = 40
MIN_STRING_LENGTH = 12
READ_WORKERS = 4
GIT_TIMEOUT = 30
SUFFIXES = {'.py', '.sh', '.bash', '.env'}
EXCLUDED_PARTS = {'vendor', 'vendored', 'node_modules', '__pycache__'}
POLICY_NAME = re.compile(r'(?:^|_)(?:timeout|port|revision|version|price|limit|seconds|minutes)(?:_|$)')
# Expansions, arrays and executable commands are outside this explicit shell subset.
SHELL_SCALAR = r'''("[^"$`\\]*"|'[^']*'|[\w./:@+%-]+)'''
SHELL_ASSIGNMENT = rf'([A-Za-z_]\w*)={SHELL_SCALAR}'
SHELL_DEFINITION = re.compile(SHELL_ASSIGNMENT)
SHELL_LINE = re.compile(rf'^\s*(?:(?:export|readonly|local)\s+)?(?P<definitions>{SHELL_ASSIGNMENT}'
                        rf'(?:(?:\s*;\s*|\s+){SHELL_ASSIGNMENT})*)\s*;?\s*(?:#.*)?$')
SHELL_NAME = re.compile(r'^\s*(?:(?:export|readonly|local)\s+)?([A-Za-z_]\w*)=')
SCALAR_TYPES = (str, int, float)
UNSAFE_RENAME = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda,
                 ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp,
                 ast.Global, ast.Nonlocal, ast.Import, ast.ImportFrom, ast.Match)


@dataclass(frozen=True)
class Occurrence:
    path: str
    line: int
    end_line: int
    role: str
    text: str


def location(path, node, source, role):
    return Occurrence(path, node.lineno, node.end_lineno, role,
                      '\n'.join(source.splitlines()[node.lineno - 1:node.end_lineno]))


def literal(node):
    """Only scalar syntax, including signed numbers; never evaluate expressions."""
    if isinstance(node, ast.Constant) and type(node.value) in SCALAR_TYPES:
        return node.value
    if (isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd))
            and isinstance(node.operand, ast.Constant) and type(node.operand.value) in (int, float)):
        return -node.operand.value if isinstance(node.op, ast.USub) else node.operand.value
    return None


def value_key(value):
    return type(value).__name__, repr(value)


def python_constants(path, tree, source):
    parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}

    def binding(node, name):
        while node in parents:
            node = parents[node]
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda,
                                 ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
                break
        return node, name

    writes = defaultdict(int)
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            writes[binding(node, node.id)] += 1
    result = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    value = literal(node.value)
                    if value is not None:
                        result.append((target.id, value, location(path, node, source, target.id),
                                       writes[binding(target, target.id)] == 1))
        elif isinstance(node, ast.Call):
            # Callee is part of the role: distinct APIs need not share a timeout.
            callee = ast.unparse(node.func)
            for keyword in node.keywords:
                value = literal(keyword.value)
                if keyword.arg is not None and value is not None:
                    role = f'{callee}:{keyword.arg}'
                    result.append((role, value, location(path, keyword.value, source, role), True))
    return result


class Bindings(ast.NodeVisitor):
    def __init__(self, arguments):
        self.names = {}
        for arg in (*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs, arguments.vararg, arguments.kwarg):
            if arg is not None:
                self.bind(arg.arg)

    def bind(self, name):
        if name not in self.names:
            self.names[name] = f'@local_{len(self.names)}'

    def visit_Name(self, node):
        if isinstance(node.ctx, (ast.Store, ast.Del)):
            self.bind(node.id)

    def visit_ExceptHandler(self, node):
        if node.name is not None:
            self.bind(node.name)
        self.generic_visit(node)


class Rename(ast.NodeTransformer):
    def __init__(self, names):
        self.names = names

    def visit_Name(self, node):
        if node.id in self.names:
            node.id = self.names[node.id]
        return node

    def visit_ExceptHandler(self, node):
        if node.name is not None:
            node.name = self.names[node.name]
        return self.generic_visit(node)


def fingerprint(node):
    return hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()


def function_forms(function):
    """Preserve externals, literals, signatures, decorators and operators.

Defaults and annotations are evaluated outside the body's local scope, so do not
rename references within them. Complex scopes still participate in exact matching.
Canonical locals use impossible Python identifiers to avoid external-name collisions.
"""
    node = copy.deepcopy(function)
    node.name = 'function'
    if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant):
        if isinstance(node.body[0].value.value, str):
            node.body.pop(0)
    exact = fingerprint(node)
    if any(isinstance(child, UNSAFE_RENAME) for statement in node.body for child in ast.walk(statement)):
        return exact, None
    bindings = Bindings(node.args)
    for statement in node.body:
        bindings.visit(statement)
    for arg in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs, node.args.vararg, node.args.kwarg):
        if arg is not None:
            arg.arg = bindings.names[arg.arg]
    renamer = Rename(bindings.names)
    node.body = [renamer.visit(statement) for statement in node.body]
    return exact, fingerprint(node)


def extract(item, min_nodes):
    path, source = item
    constants, functions, limited = [], [], []
    if Path(path).suffix == '.py':
        tree = ast.parse(source, filename=path)
        constants = python_constants(path, tree, source)
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            size = sum(1 for statement in node.body for _ in ast.walk(statement))
            if size < min_nodes:
                continue
            exact, renamed = function_forms(node)
            functions.append((exact, renamed, size, location(path, node, source, node.name)))
            if renamed is None:
                limited.append({'path': path, 'line': node.lineno, 'reason': 'complex scope: exact matching only'})
    else:
        writes = defaultdict(int)
        definitions = []
        for line, text in enumerate(source.splitlines(), 1):
            match = SHELL_LINE.fullmatch(text)
            if match is not None:
                for definition in SHELL_DEFINITION.finditer(match['definitions']):
                    name, raw = definition.groups()
                    definitions.append((line, text, name, raw))
                    writes[name] += 1
            else:
                name = SHELL_NAME.match(text)
                if name is not None:
                    writes[name[1]] += 1
        for line, text, name, raw in definitions:
            if raw.startswith(('"', "'")):
                value = raw[1:-1]
            elif re.fullmatch(r'[+-]?\d+', raw):
                value = int(raw)
            elif re.fullmatch(r'[+-]?\d+\.\d+', raw):
                value = float(raw)
            else:
                value = raw
            constants.append((name, value, Occurrence(path, line, line, name, text), writes[name] == 1))
        limited.append({'path': path, 'reason': 'shell: only lines of scalar assignments; no clone analysis'})
    return constants, functions, limited


def finding(rule, reason, occurrences, **details):
    ordered = sorted(set(occurrences), key=lambda item: (item.path, item.line, item.role))
    return {'rule': rule, 'reason': reason, 'occurrences': [asdict(item) for item in ordered], **details}


def constant_findings(constants):
    roles, strings = defaultdict(list), defaultdict(list)
    for role, value, occurrence, stable in constants:
        name = role.split(':')[-1]
        if stable and (name.isupper() or POLICY_NAME.search(name.lower())):
            roles[role.lower()].append((value, occurrence))
        if isinstance(value, str) and len(value) >= MIN_STRING_LENGTH and '\n' not in value:
            strings[value].append(occurrence)
    findings, covered = [], set()
    for role, rows in sorted(roles.items()):
        occurrences = [occurrence for _, occurrence in rows]
        if len(set(occurrences)) < 2:
            continue
        values = sorted({value_key(value) for value, _ in rows})
        rule = {True: 'constant-repeat', False: 'constant-drift'}[len(values) == 1]
        findings.append(finding(rule, f'Repeated policy role {role!r}; share a definition or justify independent policies.',
                                occurrences, values=values))
        covered.add(frozenset(occurrences))
    for value, occurrences in sorted(strings.items()):
        if len(set(occurrences)) < 2 or frozenset(occurrences) in covered:
            continue
        findings.append(finding('literal-repeat', 'Repeated substantial string in definitions or call arguments; '
                                'consider one named source.', occurrences, value=value))
    return findings


def clone_findings(functions):
    groups = defaultdict(list)
    for exact, renamed, size, occurrence in functions:
        groups[('renamed', renamed) if renamed is not None else ('exact', exact)].append((exact, size, occurrence))
    findings = []
    for rows in groups.values():
        if len(rows) < 2:
            continue
        match = {True: 'exact', False: 'renamed'}[len({exact for exact, _, _ in rows}) == 1]
        findings.append(finding('function-clone', 'Shared function structure; consider extracting a shared helper. '
                                'Review side effects and caller contracts before fusion.',
                                [occurrence for _, _, occurrence in rows], match=match, nodes=rows[0][1]))
    return findings


def audit(sources, *, min_nodes=MIN_CLONE_NODES):
    """Audit a complete path→text snapshot, including a caller's proposed edits.

Only supported files belong here. Invalid Python and unreadable input are errors,
not clean audits. Callers decide whether and how findings should block an action.
"""
    if min_nodes < 1:
        raise ValueError('min_nodes must be positive')
    unsupported = sorted(path for path in sources if Path(path).suffix not in SUFFIXES)
    if unsupported:
        raise ValueError(f'unsupported input files: {unsupported}')
    constants, functions, limited = [], [], []
    with ThreadPoolExecutor(max_workers=READ_WORKERS) as pool:
        for extracted_constants, extracted_functions, coverage in pool.map(
                lambda item: extract(item, min_nodes), sorted(sources.items())):
            constants.extend(extracted_constants)
            functions.extend(extracted_functions)
            limited.extend(coverage)
    findings = constant_findings(constants) + clone_findings(functions)
    findings.sort(key=lambda item: (item['rule'], item['occurrences'][0]['path'], item['occurrences'][0]['line']))
    return {'files': len(sources), 'constants': len(constants), 'functions': len(functions),
            'min_clone_nodes': min_nodes, 'coverage_limits': limited, 'findings': findings}


def read_repository(root, *, revision=None, include_tests=False):
    """Read working files or a commit without checking it out or executing its code."""
    root = Path(root).resolve()

    def git(*args):
        return subprocess.run(['git', *args], cwd=root, check=True, capture_output=True,
                              timeout=GIT_TIMEOUT).stdout

    if Path(git('rev-parse', '--show-toplevel').decode().strip()).resolve() != root:
        raise ValueError('--root must be the repository root, not a subdirectory')
    symlinks = set()
    if revision is None:
        paths = git('ls-files', '-z', '--cached', '--others', '--exclude-standard').decode().split('\0')
    else:
        revision = git('rev-parse', '--verify', '--end-of-options', f'{revision}^{{commit}}').decode().strip()
        entries = git('ls-tree', '-rz', revision).decode().split('\0')
        paths = []
        for entry in entries:
            if not entry:
                continue
            metadata, path = entry.split('\t', 1)
            paths.append(path)
            if metadata.split()[0] == '120000':
                symlinks.add(path)
    selected, excluded = [], []
    for path in sorted(set(paths) - {''}):
        parts = Path(path).parts
        if Path(path).suffix not in SUFFIXES:
            continue
        is_test = 'tests' in parts or 'test' in parts or Path(path).name.startswith('test_')
        if (is_test and not include_tests) or any(part in EXCLUDED_PARTS for part in parts):
            excluded.append(path)
            continue
        if path in symlinks:
            raise ValueError(f'historical source symlinks are not supported: {path}')
        if revision is None and not (root / path).exists():
            # Git lists working-tree deletions too. They are absent from this snapshot.
            if (root / path).is_symlink():
                raise ValueError(f'broken source symlink: {path}')
            excluded.append(path)
            continue
        if revision is None and (root / path).is_symlink():
            raise ValueError(f'source symlinks are not supported: {path}')
        selected.append(path)

    def read(path):
        source = (root / path).read_text() if revision is None else git('show', f'{revision}:{path}').decode()
        return path, source

    with ThreadPoolExecutor(max_workers=READ_WORKERS) as pool:
        sources = dict(pool.map(read, selected))
    return sources, excluded
