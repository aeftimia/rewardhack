#!/usr/bin/env bash
# On a pod: download one archived trainer checkpoint from the lines `ops/run_archive.py resume` printed on the laptop
# ("sha256 path url" each, on stdin), into <destination>/<path>, and check every file's SHA-256 before it counts
# (notes section 78). No AWS credentials: the URLs are presigned. Usage: fetch_checkpoint.sh <destination> < lines
set -euo pipefail
[[ $# == 1 ]] || { echo 'need the destination folder; the resume lines on stdin'; exit 2; }
DEST=$1
count=0
while read -r digest name url; do
  # Only a checkpoint's own files, and no way out of the destination.
  [[ $name =~ ^[^/]+/trainer/checkpoint-[0-9]+/ && $name != *..* && -n $url ]] || { echo "not a checkpoint file line: $name"; exit 2; }
  mkdir -p "$DEST/$(dirname "$name")"
  # cap: 1.5x estimated 13 min (a 1.5 GB optimizer state at 3 MB/s, the slowest pod download seen was 43 MB/s)
  curl -fsS --max-time 1200 -o "$DEST/$name.part" "$url" || { echo "download failed: $name"; exit 1; }
  got=$(python3 -c 'import base64, hashlib, sys; print(base64.b64encode(hashlib.file_digest(open(sys.argv[1], "rb"), "sha256").digest()).decode())' "$DEST/$name.part")
  [[ $got == "$digest" ]] || { echo "SHA-256 differs for $name: not used"; exit 1; }
  mv "$DEST/$name.part" "$DEST/$name"
  count=$((count + 1))
done
((count > 0)) || { echo 'no lines on stdin'; exit 2; }
echo "fetched $count files, each with its archived SHA-256"
