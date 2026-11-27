#!/usr/bin/env bash
# deadline: 60
# A gated copy-only successor to a completed job. remote_job.sh preserves the
# previous output as out.prev.<new launch id>; no experiment is repeated here.
# Before starting its driver, verify the predecessor's DONE and EXIT_CODE=0:
# the runner stops its predecessor before this script can inspect the archive.
# Copy archive-source.txt's directory separately; the normal driver copies only
# the new out/. Signal completion only after checking the previous output locally.
set -euo pipefail
OUT="$HOME/out"
LAUNCH=$(cat "$OUT/LAUNCH_ID")
[[ $LAUNCH =~ ^[a-zA-Z0-9_-]+$ ]] || { echo 'invalid launch identity'; exit 2; }
SOURCE="$HOME/out.prev.$LAUNCH"
[[ -f $SOURCE/DONE && -f $SOURCE/EXIT_CODE && $(cat "$SOURCE/EXIT_CODE") == 0 ]] || {
  echo 'archive window requires a successfully completed predecessor'; exit 2;
}
printf '%s\n' "$SOURCE" > "$OUT/archive-source.txt"
echo "copy-only window: $SOURCE; signal ARCHIVE_COPY_COMPLETE in this launch's output after verified local copy"
# cap: 2x estimated 30 min (two optimizer checkpoints and remaining adapters over a slow connection)
COPY_WINDOW_SECONDS=3600
while ((SECONDS < COPY_WINDOW_SECONDS)); do
  if [[ -f $OUT/ARCHIVE_COPY_COMPLETE ]]; then
    echo 'local copy confirmed complete'; exit 0;
  fi
  sleep 5
done
echo 'copy window expired without a completion receipt'
exit 1
