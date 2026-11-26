#!/usr/bin/env bash
# On a pod: download one archived trainer checkpoint from the lines `ops/run_archive.py resume` printed on the laptop
# ("sha256 path url" per file, then "END <count>"), into <destination>/<path>, checking every file's SHA-256 before it counts
# (notes section 78). No AWS credentials: the URLs are presigned. Usage: fetch_checkpoint.sh <destination> < lines
set -euo pipefail
[[ $# == 1 ]] || { echo 'need the destination folder; the resume lines on stdin'; exit 2; }
DEST=$1
LINES=$(cat)
# The whole list first: it must end with END and the number of files before it, or it was cut short (review round 42).
count=$(grep -c -v '^END ' <<< "$LINES") || { echo 'no file lines on stdin'; exit 2; }
[[ $(tail -n 1 <<< "$LINES") == "END $count" ]] || { echo "the list does not end with 'END $count': it was cut short"; exit 2; }
# One checkpoint: every file under the first line's checkpoint folder, which must not exist yet (no mixing with an earlier fetch).
first=$(head -n 1 <<< "$LINES" | cut -d' ' -f2)
[[ $first =~ ^([^/]+/trainer/checkpoint-[0-9]+)/ ]] || { echo "not a checkpoint file: $first"; exit 2; }
CHECKPOINT=${BASH_REMATCH[1]}
[[ ! -e $DEST/$CHECKPOINT ]] || { echo "$DEST/$CHECKPOINT already exists: fetch into an empty destination"; exit 2; }
while read -r digest name url; do
  [[ $digest == END ]] && break
  [[ $name == "$CHECKPOINT"/* && $name != *..* && -n $url ]] || { echo "not a file of $CHECKPOINT: $name"; exit 2; }
  mkdir -p "$DEST/$(dirname "$name")"
  # cap: 1.5x estimated 13 min (a 1.5 GB optimizer state at 3 MB/s; the slowest pod download seen was 43 MB/s)
  curl -fsS --max-time 1200 -o "$DEST/$name.part" "$url" || { echo "download failed: $name"; exit 1; }
  got=$(python3 -c 'import base64, hashlib, sys; print(base64.b64encode(hashlib.file_digest(open(sys.argv[1], "rb"), "sha256").digest()).decode())' "$DEST/$name.part")
  [[ $got == "$digest" ]] || { echo "SHA-256 differs for $name: not used"; exit 1; }
  mv "$DEST/$name.part" "$DEST/$name"
done <<< "$LINES"
echo "fetched all $count files of $CHECKPOINT, each with its archived SHA-256"
