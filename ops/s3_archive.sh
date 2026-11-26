#!/usr/bin/env bash
# The run archive on S3 (ops/archive.env; notes section 78). Runs on the laptop with the owner's AWS SSO session: pods hold no
# AWS credentials, and get presigned URLs for what they must read. Usage:
#   s3_archive.sh upload <results folder>     # SHA256SUMS written, every file sent (S3 checks each part's SHA-256 on
#                                             # arrival), then every file listed in the bucket at its local size; writes ARCHIVED
#   s3_archive.sh prune <results folder>      # deletes the local adapter-step*/ and trainer/checkpoint-*/ of an archived run,
#                                             # after listing the bucket again. Never run by anything else: the owner says yes per run
#   s3_archive.sh presign <run name> <path>...  # presigned GET URLs, ARCHIVE_URL_SECONDS long, for a pod's resume
set -euo pipefail
cd "$(dirname "$0")/.."
source ops/archive.env
AWS=(aws --region "$ARCHIVE_REGION")
need_folder() {
  [[ -d $1 ]] || { echo "no folder $1"; exit 2; }
  NAME=${1%/}; NAME=${NAME##*/}
  [[ $NAME =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || { echo "the folder name becomes the run's key and must be letters, digits, '.', '_' or '-': $NAME"; exit 2; }
  URI=s3://$ARCHIVE_BUCKET/$ARCHIVE_PREFIX/$NAME
}
# Every local file of the run present in the bucket at its local size; the manifest must name exactly the folder's files.
check_listed() {
  local local_list remote_list missing
  local_list=$(python3 ops/sha256_manifest.py listed "$1" "$1/SHA256SUMS") || { echo "$1 and its SHA256SUMS disagree (above): nothing checked" >&2; exit 1; }
  local_list+=$'\n'"SHA256SUMS $(wc -c < "$1/SHA256SUMS" | tr -d ' ')"
  remote_list=$("${AWS[@]}" s3api list-objects-v2 --bucket "$ARCHIVE_BUCKET" --prefix "$ARCHIVE_PREFIX/$NAME/" \
                  --query 'Contents[].[Key, Size]' --output text | sed "s|^$ARCHIVE_PREFIX/$NAME/||" | tr '\t' ' ')
  missing=$(comm -23 <(sort <<< "$local_list") <(sort <<< "$remote_list"))
  [[ -z $missing ]] || { echo "not in $URI at the local size:"; echo "$missing" | head -20; exit 1; }
  echo "all $(wc -l <<< "$local_list" | tr -d ' ') files are in $URI at their local sizes"
}
COMMAND=${1:?'upload, prune or presign'}
case $COMMAND in
  upload)
    [[ $# == 2 ]] || { echo 'upload <results folder>'; exit 2; }
    need_folder "$2"
    python3 ops/sha256_manifest.py write "$2" "$2/SHA256SUMS"
    # Only the top-level names the manifest leaves out (sha256_manifest.NOT_THE_RUN) are left here; the manifest goes too.
    "${AWS[@]}" s3 sync "$2" "$URI" --storage-class "$ARCHIVE_STORAGE_CLASS" --checksum-algorithm SHA256 \
      --exclude known_hosts --exclude ARCHIVED --no-progress
    check_listed "$2"
    printf '%s %s\n' "$(date -u +%FT%TZ)" "$URI" > "$2/ARCHIVED"
    ;;
  prune)
    [[ $# == 2 ]] || { echo 'prune <results folder>'; exit 2; }
    need_folder "$2"
    [[ -s $2/ARCHIVED ]] || { echo "$2 has not been archived (no ARCHIVED): upload it first"; exit 2; }
    check_listed "$2"
    # Only what can be fetched back for a resume or an analysis; rows, logs, summaries and the final adapter stay here.
    find "$2" -type d \( -name 'adapter-step*' -o -path '*/trainer/checkpoint-*' \) -prune -print -exec rm -r {} +
    ;;
  presign)
    [[ $# -ge 3 ]] || { echo 'presign <run name> <path in the run>...'; exit 2; }
    run=$2; shift 2
    for path in "$@"; do
      "${AWS[@]}" s3 presign "s3://$ARCHIVE_BUCKET/$ARCHIVE_PREFIX/$run/$path" --expires-in "$ARCHIVE_URL_SECONDS"
    done
    ;;
  *) echo "unknown command $COMMAND: upload, prune or presign"; exit 2 ;;
esac
