# DRY audit prototype

`rewardhack.dry_audit` discovers candidates for consolidation without modifying
code, calling a model, or blocking edits. `entrypoints/audit_dry.py` provides the
repository CLI. It is an experimental library module so its snapshot API can later
be used by an edit or commit hook without duplicating the scanner.

From the checkout root, using the project's Python:

```bash
PYTHONPATH=src .tools/venv/bin/python entrypoints/audit_dry.py
PYTHONPATH=src .tools/venv/bin/python entrypoints/audit_dry.py --json
PYTHONPATH=src .tools/venv/bin/python entrypoints/audit_dry.py --revision '5683bf3^' --json
```

In a linked worktree, use the main checkout's `.tools/venv/bin/python` with
`PYTHONPATH=src`. `--root` must identify the repository root. The working-tree
scan includes tracked and non-ignored untracked sources; a revision scan reads
Git blobs without checking out or executing historical code. Tests and vendor
directories are excluded by default; `--include-tests` includes test expectations
as candidates too. Independent expected values often belong in tests.

The normal exit status is zero even with candidates. `--check` returns one when
candidates exist. Input/read/parse errors fail the command; they are not clean
audits. No hook configuration is changed by installing or running this tool.

## Detection

| Finding | Evidence | Review needed |
| --- | --- | --- |
| `constant-repeat` | Equal typed literals assigned to the same policy-like name, or the same callee's policy keyword | Names can coincide across independent policies |
| `constant-drift` | Different typed literals in those roles | Values may differ intentionally; this is suspected drift |
| `literal-repeat` | A repeated substantial single-line string in assignments or keyword arguments | Shared model IDs and revisions are useful targets; repeated prose may be intentional |
| `function-clone` | Identical Python function ASTs, optionally after renaming local bindings | Similar structure does not prove equivalent behavior or safe extraction |

Thresholds live in `src/rewardhack/dry_audit.py`; `--min-clone-nodes` overrides the
function-size threshold. There is no arbitrary pairwise similarity threshold:
the first version uses hashes of structural representations. Reports group all
matching occurrences, carry source locations and excerpts, and distinguish exact
from renamed function matches. Output ordering is deterministic.

Names assigned repeatedly within a Python scope are not treated as constant
definitions. The shell subset conservatively counts writes by name within a file.
Repeated long string values still participate even when their variable is reused
for another model. Booleans and null values are not policy constants. Equal numeric
values with unrelated names do not produce a finding.

Python renaming preserves which references use the same binding, external name
spellings, operators, literal values, keyword names, decorators, annotations and
defaults in the function signature. Canonical local names cannot collide with real
external identifiers. Imports, global/nonlocal declarations, pattern matching,
comprehensions and nested scopes make a function eligible for exact matching only;
the report names each such limitation. Nested functions are also inspected on their
own. External names are not resolved across modules, and recursive references are
not normalized to a common function name.

## Coverage and deliberate limits

Supported files are Python and scalar-definition text in `.sh`, `.bash`, and `.env`.
The shell extractor accepts a complete line of literal assignments separated by
spaces or semicolons, optionally prefixed by `export`, `local`, or `readonly`.
It does not parse shell control flow, expansions, arrays, embedded languages, or
function clones. Assignment-looking lines in multiline strings or heredocs can
therefore produce candidates; shell findings require source review. Unsupported
file formats, such as TOML, requirements files and documentation, are not scanned.
Source symlinks are rejected rather than followed or interpreted as source.

The Python pass currently handles scalar assignments to names and literal keyword
arguments. It does not match positional arguments, dictionary values, function
parameter defaults, computed constants, units, aliases, or data flow. These limits
mean that even a report with no findings is not proof that a repository is DRY.

Function matching is whole-function matching. It does not yet generalize differing
literals into parameters, search consecutive statement blocks, compare reordered
operations, or establish equivalence. Reflection and differently resolved imports
can make even identical function text behave differently.

The scanner does not approve exceptions or infer that an equality test justifies a
copy. For example, this repository intentionally keeps some standalone shell values
in sync through `tests/test_shared_constants.py`; those copies still appear as
candidates. It also groups repeated keyword names without inspecting other arguments:
different subprocess commands can legitimately have different `timeout` policies.

## Historical check and next step

The first historical check compares `5683bf3^` with `5683bf3`. Before the change,
the scanner finds repeated model/revision strings for both gpt-oss models in pod
scripts (four groups). Those groups disappear after the consolidation into
`ops/pod_pins.env`. Remaining findings include copies deliberately protected by
equality tests. Total finding counts alone are not a quality metric: consolidation
can move a value into a source this limited scanner can recognize for the first time.

The helper extraction in `6ae7e54` is a known miss: it concerns shell workflows,
which this version does not structurally analyze. Synthetic positive and negative
Python cases exercise exact/renamed functions, binding identity, external calls,
scope limits, independent numeric values, mutable state, and complete-snapshot edits.

The next detector should anti-unify candidate ASTs: produce the shared template and
explicit differing subtrees, then assess how much shared logic remains relative to
the needed parameters. Shell workflow matching needs a real shell parser first.
Historical refactors supply positive examples; independent policies and test
expectations supply negative examples. Tune precision before making findings block.

For hook integration, `audit(sources)` accepts the complete proposed path-to-text
snapshot. Apply all edits in memory before calling it, so introducing a helper and
replacing its callers is one operation. A commit adapter must read staged blobs,
not the working tree. Neither adapter is implemented here. A future judged exception
must bind to the specific occurrences and context, invalidate when they change,
and explain why independent definitions are justified. This prototype does not
claim semantic review or enforcement.
