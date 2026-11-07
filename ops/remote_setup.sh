#!/usr/bin/env bash
# Runs on the instance. Installs Lean and the Python stack. Lean always runs with an empty environment
# and no network, because proofs are model-written; the Hugging Face token never reaches that process.
set -euo pipefail
cd ~/pilot-repo
curl -fsSL https://github.com/leanprover/lean4/releases/download/v4.19.0/lean-4.19.0-linux.tar.zst | sudo tar --zstd -xf - -C /opt
LEAN_BIN=/opt/lean-4.19.0-linux/bin/lean
unshare -rn true || { echo 'unprivileged network namespaces unavailable; refusing to run model-written Lean'; exit 1; }
cat > ~/lean-safe <<EOF
#!/bin/bash
exec env -i PATH=/usr/bin:/bin unshare -rn $LEAN_BIN "\$@"
EOF
chmod +x ~/lean-safe
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv venv ~/venv --python 3.12 -q
uv pip install -q --python ~/venv/bin/python torch -r requirements.txt
~/lean-safe --version
~/venv/bin/python entrypoints/run.py controls --out ~/out/controls --lean ~/lean-safe
~/venv/bin/python tests/test_learning.py
nvidia-smi --query-gpu=name,memory.total --format=csv
