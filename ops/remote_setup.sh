#!/usr/bin/env bash
# Runs on the instance. Installs Lean and the Python stack. Lean always runs with an empty environment
# and no network, because proofs are model-written; the Hugging Face token never reaches that process.
set -euo pipefail
cd ~/pilot-repo
# Never timed on an AWS instance (5 s at most on RunPod pods, unpacking included): estimated 20 s, limit 120 s.
# cap: 3x estimated 20 s
curl -fsSL --max-time 120 https://github.com/leanprover/lean4/releases/download/v4.19.0/lean-4.19.0-linux.tar.zst | sudo tar --zstd -xf - -C /opt
LEAN_BIN=/opt/lean-4.19.0-linux/bin/lean
unshare -rn true || { echo 'unprivileged network namespaces unavailable; refusing to run model-written Lean'; exit 1; }
cat > ~/lean-safe <<EOF
#!/bin/bash
exec env -i PATH=/usr/bin:/bin unshare -rn $LEAN_BIN "\$@"
EOF
chmod +x ~/lean-safe
# The installer script is a few kilobytes; never timed: estimated 10 s, limit 60 s. The limit covers this download only, not what the installer then fetches itself.
# cap: 3x estimated 10 s
curl -LsSf --max-time 60 https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv venv ~/venv --python 3.12 -q
uv pip install -q --python ~/venv/bin/python torch -r requirements.txt
~/lean-safe --version
~/venv/bin/python entrypoints/run.py controls --out ~/out/controls --lean ~/lean-safe
~/venv/bin/python tests/test_learning.py
nvidia-smi --query-gpu=name,memory.total --format=csv
