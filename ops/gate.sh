#!/usr/bin/env bash
# Run entrypoints/gate.py from the checkout this script is in, with that checkout's library (src/) and the project's
# Python. A linked worktree has no .tools of its own, so the Python is found through the repository's common git
# directory: <main checkout>/.tools/venv. Git sets GIT_DIR and GIT_INDEX_FILE for its hooks, naming the repository
# being committed to (perhaps another one): they are cleared only for finding the Python and kept for gate.py.
# Usage: ops/gate.sh <gate.py subcommand and arguments>
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
COMMON=$(env -u GIT_DIR -u GIT_WORK_TREE -u GIT_INDEX_FILE -u GIT_COMMON_DIR git -C "$HERE" rev-parse --path-format=absolute --git-common-dir)
PYTHON=$(dirname "$COMMON")/.tools/venv/bin/python
[[ -x $PYTHON ]] || { echo "gate: no project Python at $PYTHON, so nothing can be checked and everything is refused. Create the virtual environment there." >&2; exit 2; }
ENTRYPOINT=gate.py
if [[ ${1:-} == runpod ]]; then ENTRYPOINT=runpod_ops.py; shift; fi
PYTHONPATH="$HERE/src" exec "$PYTHON" "$HERE/entrypoints/$ENTRYPOINT" "$@"
