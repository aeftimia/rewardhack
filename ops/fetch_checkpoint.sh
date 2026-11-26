#!/usr/bin/env bash
# On a pod: fetch what a resume needs from the lines `ops/run_archive.py resume` printed on the laptop ("sha256 path url" per
# file, then "END <count>"): one trainer checkpoint and the run files beside it, into <destination>/<run>/, checked with
# `sha256sum -c` (notes section 80). The run folder appears only when every file has arrived with its archived hash; any
# failure leaves nothing. No AWS credentials: the URLs are presigned. Usage: fetch_checkpoint.sh <destination> < lines
set -euo pipefail
[[ $# == 1 ]] || { echo 'need the destination folder; the resume lines on stdin'; exit 2; }
DEST=$1
LINES=$(cat)
# One whole list: it ends with the only END line, giving the number of files before it (review rounds 42-43).
count=$(grep -c -v '^END ' <<< "$LINES") || { echo 'no file lines on stdin'; exit 2; }
[[ $(tail -n 1 <<< "$LINES") == "END $count" && $(grep -c '^END ' <<< "$LINES") == 1 ]] || { echo "not one whole resume list ending 'END $count'"; exit 2; }
RUN=$(head -n 1 <<< "$LINES" | cut -d' ' -f2 | cut -d/ -f1)
[[ $RUN =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || { echo "not a run folder name: $RUN"; exit 2; }
[[ ! -e $DEST/$RUN ]] || { echo "$DEST/$RUN already exists: fetch into a destination without it"; exit 2; }
mkdir -p "$DEST"
WORK=$(mktemp -d "$DEST/.fetching.XXXXXX")  # on the destination's disk, so the final move is a rename
trap 'rm -rf "$WORK"' EXIT
: > "$WORK/SHA256SUMS"
while read -r digest name url; do
  [[ $digest == END ]] && break
  [[ $name == "$RUN"/* && $name != *..* && $digest =~ ^[0-9a-f]{64}$ && -n $url ]] || { echo "not a file of $RUN: $name"; exit 2; }
  mkdir -p "$WORK/$(dirname "$name")"
  # cap: 1.5x estimated 13 min (a 1.5 GB optimizer state at 3 MB/s; the slowest pod download seen was 43 MB/s)
  curl -fsS --max-time 1200 -o "$WORK/$name" "$url" || { echo "download failed: $name"; exit 1; }
  printf '%s  %s\n' "$digest" "$name" >> "$WORK/SHA256SUMS"
done <<< "$LINES"
[[ $(wc -l < "$WORK/SHA256SUMS") -eq $count ]] || { echo "not all $count files listed were fetched"; exit 1; }
(cd "$WORK" && sha256sum -c --quiet SHA256SUMS) || { echo 'a file differs from its archived SHA-256: nothing kept'; exit 1; }
mv "$WORK/$RUN" "$DEST/$RUN"
echo "fetched all $count files of $RUN's resume, each with its archived SHA-256"
