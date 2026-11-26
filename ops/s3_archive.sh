#!/usr/bin/env bash
# The run archive on S3 (ops/archive.env; notes section 78). Runs on the laptop with the owner's AWS SSO session: pods hold no
# AWS credentials, and get presigned URLs for what they must read. Usage:
#   s3_archive.sh upload <results folder>     # SHA256SUMS written, every file sent (S3 checks each part's SHA-256 on
#                                             # arrival), then every file listed in the bucket at its local size; writes ARCHIVED
#   s3_archive.sh prune <results folder>      # deletes the local <run>/*/adapter-stepN/ and <run>/*/trainer/checkpoint-N/ of an
#                                             # archived run, after checking it again. Never run by anything else: the owner says
#                                             # yes per run
#   s3_archive.sh presign <run name> <path>...  # presigned GET URLs, ARCHIVE_URL_SECONDS long, for a pod's resume
# A run's key is its folder's name, and the first folder uploaded under a name owns it (ORIGIN in the bucket holds its absolute
# path): another folder of the same name is refused before it sends anything (review round 36).
set -euo pipefail
# A relative folder is read from where the command was run, before the move to the repository (review round 36).
if [[ $# -ge 2 && ${1:-} != presign ]]; then
  [[ -d $2 ]] || { echo "no folder $2"; exit 2; }
  FOLDER=$(cd "$2" && pwd -P)
fi
cd "$(dirname "$0")/.."
source ops/archive.env
AWS=(aws --region "$ARCHIVE_REGION")
# macOS's own /usr/bin/python3 is 3.9; the manifest needs hashlib.file_digest (3.11).
python3 -c 'import sys; sys.exit(sys.version_info < (3, 11))' || { echo "python3 here is $(python3 --version 2>&1); 3.11 or later is needed: put the project's venv first on PATH" >&2; exit 2; }
need_folder() {
  NAME=${FOLDER##*/}
  [[ $NAME =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || { echo "the folder name becomes the run's key and must be letters, digits, '.', '_' or '-': $NAME"; exit 2; }
  KEY=$ARCHIVE_PREFIX/$NAME
  URI=s3://$ARCHIVE_BUCKET/$KEY
}
sha256_of() { python3 -c 'import hashlib, sys; print(hashlib.sha256(sys.stdin.buffer.read()).hexdigest())'; }
manifest_hash() { sha256_of < "$FOLDER/SHA256SUMS"; }
# Whether anything is stored under the run's name yet ("None" when nothing is).
first_key() { "${AWS[@]}" s3api list-objects-v2 --bucket "$ARCHIVE_BUCKET" --prefix "$KEY/" --max-items 1 --query 'Contents[0].Key' --output text; }
# The folder the run's name belongs to must be this one; refused before anything is sent or deleted.
check_origin() {
  local origin
  origin=$("${AWS[@]}" s3 cp "$URI/ORIGIN" -)
  [[ $origin == "$FOLDER" ]] || { echo "$URI belongs to $origin, not $FOLDER: rename this folder before archiving it" >&2; exit 1; }
}
# Every local file of the run present in the bucket at its local size; the manifest must name exactly the folder's files.
check_listed() {
  local local_list remote_list missing
  local_list=$(python3 ops/sha256_manifest.py listed "$FOLDER" "$FOLDER/SHA256SUMS") || { echo "$FOLDER and its SHA256SUMS disagree (above): nothing checked" >&2; exit 1; }
  local_list+=$'\n'"SHA256SUMS $(wc -c < "$FOLDER/SHA256SUMS" | tr -d ' ')"
  remote_list=$("${AWS[@]}" s3api list-objects-v2 --bucket "$ARCHIVE_BUCKET" --prefix "$KEY/" \
                  --query 'Contents[].[Key, Size]' --output text | sed "s|^$KEY/||" | tr '\t' ' ')
  missing=$(comm -23 <(sort <<< "$local_list") <(sort <<< "$remote_list"))
  [[ -z $missing ]] || { echo "not in $URI at the local size:"; echo "$missing" | head -20; exit 1; }
  echo "all $(wc -l <<< "$local_list" | tr -d ' ') files are in $URI at their local sizes"
}
COMMAND=${1:?'upload, prune or presign'}
case $COMMAND in
  upload)
    [[ $# == 2 ]] || { echo 'upload <results folder>'; exit 2; }
    need_folder
    if [[ $(first_key) == None ]]; then
      printf '%s' "$FOLDER" > "$FOLDER/ORIGIN"
      "${AWS[@]}" s3 cp "$FOLDER/ORIGIN" "$URI/ORIGIN" --storage-class "$ARCHIVE_STORAGE_CLASS" --no-progress
      rm "$FOLDER/ORIGIN"
    fi
    check_origin
    # An earlier mark no longer holds once the manifest is rewritten: only a finished upload marks the run again (round 34).
    rm -f "$FOLDER/ARCHIVED"
    python3 ops/sha256_manifest.py write "$FOLDER" "$FOLDER/SHA256SUMS"
    # Every file is sent, every time: `s3 sync` skips a file of the same size with an older modification time, which a final
    # copy from a pod can have after an interim one was archived (review round 33). Only the top-level names the manifest
    # leaves out (sha256_manifest.NOT_THE_RUN) stay here; the manifest goes too. S3 checks each part's SHA-256 on arrival.
    "${AWS[@]}" s3 cp --recursive "$FOLDER" "$URI" --storage-class "$ARCHIVE_STORAGE_CLASS" --checksum-algorithm SHA256 \
      --exclude known_hosts --exclude ARCHIVED --exclude ORIGIN --no-progress
    # The bytes sent are the bytes hashed: nothing changed here between the manifest and the end of the copy.
    python3 ops/sha256_manifest.py verify "$FOLDER" "$FOLDER/SHA256SUMS" > /dev/null || { echo "$FOLDER changed during the upload: run it again" >&2; exit 1; }
    check_listed
    # The mark names the manifest it covers: prune refuses a manifest written since.
    printf '%s %s %s\n' "$(date -u +%FT%TZ)" "$URI" "$(manifest_hash)" > "$FOLDER/ARCHIVED"
    ;;
  prune)
    [[ $# == 2 ]] || { echo 'prune <results folder>'; exit 2; }
    need_folder
    [[ -s $FOLDER/ARCHIVED ]] || { echo "$FOLDER has not been archived (no ARCHIVED): upload it first"; exit 2; }
    read -r _ _ marked < "$FOLDER/ARCHIVED"
    [[ $marked == "$(manifest_hash)" ]] || { echo "$FOLDER/SHA256SUMS is not the manifest its ARCHIVED mark covers: upload it again" >&2; exit 1; }
    check_origin
    # The local copy is still the one archived (a file rewritten at the same size since would not show in the listing).
    python3 ops/sha256_manifest.py verify "$FOLDER" "$FOLDER/SHA256SUMS" > /dev/null || { echo "$FOLDER changed since it was archived: upload it again" >&2; exit 1; }
    check_listed
    held=$("${AWS[@]}" s3 cp "$URI/SHA256SUMS" - | sha256_of)
    [[ $held == "$(manifest_hash)" ]] || { echo "$URI/SHA256SUMS is not this folder's manifest: nothing pruned" >&2; exit 1; }
    # Only what can be fetched back for a resume or an analysis, at the places the trainer writes them (each run's folder,
    # one level down); rows, logs, summaries and the final adapter stay here. Shell patterns never cross a '/' (round 36).
    for stored in "$FOLDER"/*/adapter-step[0-9]* "$FOLDER"/*/trainer/checkpoint-[0-9]*; do
      if [[ -d $stored ]]; then echo "$stored"; rm -rf "$stored"; fi
    done
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
