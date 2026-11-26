#!/usr/bin/env bash
# Upload a results folder from this machine to the network volume, in the layout jobs with SMOKE_VOLUME=1 write there
# ($VOLUME_RUNS/<folder name>/, ops/pod_pins.env), and verify every file's SHA-256 on the pod afterwards (notes section 78).
# The pod is any cheap one (a CPU pod will do) deployed in $VOLUME_DATACENTER with the volume attached at $VOLUME_MOUNT.
# Like the job driver, the first act on it is a removal timer sized to this upload, and the last is its removal (KEEP_POD=1
# leaves it, still on its timer). Interrupted, run it again: rsync carries on from what arrived, and the check covers all.
# Usage, in this order (the first runs for minutes on a large folder, so it comes before any pod is deployed):
#   volume_upload.sh manifest <results folder>                # writes <folder>/SHA256SUMS on this machine; no pod
#   volume_upload.sh <ssh host> <ssh port> <results folder>   # needs that manifest; deploy the pod just before
set -euo pipefail
cd "$(dirname "$0")/.."
# Symbolic links would be copied as links and left out of the check, so a folder with any is refused.
no_links() { [[ -z $(find "$1" -type l | head -1) ]] || { echo "$1 holds symbolic links, which the check cannot cover: replace them with the files"; exit 2; }; }
if [[ $# == 2 && $1 == manifest ]]; then
  [[ -d $2 ]] || { echo "no folder $2"; exit 2; }
  no_links "$2"
  python3 -c 'import sys; assert sys.version_info >= (3, 11), "python3 3.11 or later is needed for hashlib.file_digest"'
  # The laptop's SSH trust for the job's pod (known_hosts) is not part of the run and is not uploaded.
  python3 ops/sha256_manifest.py write "$2" "$2/SHA256SUMS.all"
  grep -v -E '  known_hosts$' "$2/SHA256SUMS.all" > "$2/SHA256SUMS" || { echo "nothing to upload in $2"; exit 2; }
  rm "$2/SHA256SUMS.all"
  exit 0
fi
[[ $# == 3 ]] || { echo 'need: manifest <results folder>, or host port <results folder>'; exit 2; }
HOST=$1 PORT=$2 SOURCE=${3%/}
source ops/pod_pins.env
[[ -d $SOURCE ]] || { echo "no folder $SOURCE"; exit 2; }
no_links "$SOURCE"
[[ -s $SOURCE/SHA256SUMS ]] || { echo "no $SOURCE/SHA256SUMS: run 'volume_upload.sh manifest $SOURCE' before deploying the pod"; exit 2; }
# A file added since the manifest would be uploaded unchecked.
[[ -z $(find "$SOURCE" -type f -newer "$SOURCE/SHA256SUMS" ! -name known_hosts | head -1) ]] || { echo "files in $SOURCE are newer than its SHA256SUMS: write the manifest again"; exit 2; }
RUN_ID=${SOURCE##*/}
[[ $RUN_ID =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || { echo "the folder name becomes the run's name on the volume and must be letters, digits, '.', '_' or '-': $RUN_ID"; exit 2; }
if [[ -n ${KEEP_POD+set} ]]; then KEEP=$KEEP_POD; else KEEP=0; fi  # unset: the pod is removed after the check
[[ $KEEP == 0 || $KEEP == 1 ]] || { echo 'KEEP_POD must be 0 or 1'; exit 2; }
DEST=$VOLUME_RUNS/$RUN_ID
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
# The laptop's SSH trust for the job's pod is not part of the run.
EXCLUDE=(--exclude=known_hosts --exclude=SHA256SUMS)
cp "$SOURCE/SHA256SUMS" "$WORK/SHA256SUMS"
BYTES=$(du -sk "$SOURCE" | cut -f1)
BYTES=$((BYTES * 1024))
# The pod's removal timer: the transfer at an estimated 5 MB/s from this laptop (never measured to a pod), plus 10 minutes
# for setup and the check on the pod, at the estimated multiple (gates.MULTIPLES).
# cap: 1.5x estimated 68 min (20 GB at 5 MB/s plus 10 minutes; the timer is computed from the folder's size this way)
UPLOAD_SECONDS=$(( (BYTES / 5000000 + 600) * 3 / 2 ))
echo "uploading $((BYTES / 1000000)) MB to $DEST; the pod removes itself at most $((UPLOAD_SECONDS / 60)) minutes from now"
SSH=(ssh -p "$PORT" -o StrictHostKeyChecking=accept-new -o "UserKnownHostsFile=$WORK/known_hosts" -o ConnectTimeout=10 -o ServerAliveInterval=15 -o ServerAliveCountMax=4)
remove() {
  if [[ $KEEP == 1 ]]; then echo "pod left running (KEEP_POD=1); its removal timer still runs"; return 0; fi
  "${SSH[@]}" "root@$HOST" 'bash -s -- --remove-now' < ops/remote_job.sh || { echo 'REMINDER: could not remove the pod from here. Terminate it in the RunPod console now'; return 97; }
  echo 'pod removal accepted; confirm in the RunPod console'
}
"${SSH[@]}" "root@$HOST" true || { echo 'ssh failed: the pod is not answering, so no removal timer is armed on it. Terminate it in the RunPod console'; exit 1; }
"${SSH[@]}" "root@$HOST" "bash -s -- --arm-only $UPLOAD_SECONDS" < ops/remote_job.sh || { echo 'could not arm a removal timer on the pod: terminate it in the RunPod console'; exit 1; }
"${SSH[@]}" "root@$HOST" "mountpoint -q $VOLUME_MOUNT && python3 -c 'import sys; assert sys.version_info >= (3, 11)' && mkdir -p $DEST" || {
  echo "the pod has no volume at $VOLUME_MOUNT (deploy it in $VOLUME_DATACENTER with the volume attached there) or no python3 of 3.11 or later"; remove; exit 1; }
# --partial: an interrupted file carries on where it stopped. --checksum: on a rerun after a failed check, a damaged copy with
# the right size and time is sent again. No --delete: files on the volume are never removed from here.
rsync -a --checksum --partial --timeout=120 "${EXCLUDE[@]}" -e "${SSH[*]}" "$SOURCE/" "root@$HOST:$DEST/" || { echo 'upload failed; run this again to carry on'; remove; exit 1; }
rsync -a --timeout=120 -e "${SSH[*]}" "$WORK/SHA256SUMS" "root@$HOST:$DEST/SHA256SUMS" || { echo 'manifest upload failed'; remove; exit 1; }
if "${SSH[@]}" "root@$HOST" "python3 - verify $DEST $DEST/SHA256SUMS" < ops/sha256_manifest.py; then
  echo "verified: every file in $SOURCE matches its copy in $DEST"
  remove
else
  echo "CHECK FAILED: the files named above differ on the volume; run this again to send them"
  remove
  exit 1
fi
