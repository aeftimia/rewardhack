#!/usr/bin/env bash
# The run archive on S3 (ops/archive.env; notes section 78). Runs on the laptop with the owner's AWS SSO session: pods hold no
# AWS credentials, and get presigned URLs for what they must read. Usage:
#   s3_archive.sh upload <results folder>     # hashes the folder, refuses if the archive holds any of its files with other
#                                             # bytes, sends the archive's manifest first and then every file, checks the
#                                             # listing, and writes ARCHIVED
#   s3_archive.sh prune <results folder>      # deletes the local <run>/*/adapter-stepN/ and <run>/*/trainer/checkpoint-N/ whose
#                                             # files are archived exactly as they are now. Never run by anything else: the
#                                             # owner says yes per run
#   s3_archive.sh presign <run name> <path>...  # presigned GET URLs, ARCHIVE_URL_SECONDS long, for a pod's resume
# The archive's own manifest (SHA256SUMS in the bucket) is the record of what it holds: an upload may add files to a run,
# never replace archived bytes, so a reused or same-named folder cannot overwrite a run (review round 37), and a moved folder
# is the same run if its files match.
set -euo pipefail
# A relative folder is read from where the command was run, before the move to the repository (review round 36).
if [[ $# -ge 2 && ${1:-} != presign ]]; then
  [[ -d $2 ]] || { echo "no folder $2"; exit 2; }
  FOLDER=$(cd "$2" && pwd -P)
fi
cd "$(dirname "$0")/.."
source ops/archive.env
AWS=(aws --region "$ARCHIVE_REGION")
MANIFEST=(python3 ops/sha256_manifest.py)
# macOS's own /usr/bin/python3 is 3.9; the manifest needs hashlib.file_digest (3.11).
python3 -c 'import sys; sys.exit(sys.version_info < (3, 11))' || { echo "python3 here is $(python3 --version 2>&1); 3.11 or later is needed: put the project's venv first on PATH" >&2; exit 2; }
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
need_folder() {
  NAME=${FOLDER##*/}
  [[ $NAME =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || { echo "the folder name becomes the run's key and must be letters, digits, '.', '_' or '-': $NAME"; exit 2; }
  KEY=$ARCHIVE_PREFIX/$NAME
  URI=s3://$ARCHIVE_BUCKET/$KEY
}
manifest_hash() { python3 -c 'import hashlib, sys; print(hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest())' "$FOLDER/SHA256SUMS"; }
# The archive's manifest into $WORK/stored, or an empty one when nothing is stored under the name yet. The manifest is always
# the first object sent, so a name with objects and no manifest is a state this tool does not make: it is refused.
fetch_stored() {
  local first
  first=$("${AWS[@]}" s3api list-objects-v2 --bucket "$ARCHIVE_BUCKET" --prefix "$KEY/" --max-items 1 --query 'Contents[0].Key' --output text)
  if [[ $first == None ]]; then : > "$WORK/stored"; return 0; fi
  "${AWS[@]}" s3 cp "$URI/SHA256SUMS" "$WORK/stored" --no-progress > /dev/null \
    || { echo "$URI holds objects but no SHA256SUMS, which this tool never leaves: look at it before archiving under this name" >&2; exit 1; }
}
# Every local file of the run present in the bucket at its local size; the manifest must name exactly the folder's files.
check_listed() {
  local local_list remote_list missing
  local_list=$("${MANIFEST[@]}" listed "$FOLDER" "$FOLDER/SHA256SUMS") || { echo "$FOLDER and its SHA256SUMS disagree (above): nothing checked" >&2; exit 1; }
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
    # An earlier mark no longer holds once the manifest is rewritten: only a finished upload marks the run again (round 34).
    rm -f "$FOLDER/ARCHIVED"
    "${MANIFEST[@]}" write "$FOLDER" "$FOLDER/SHA256SUMS"
    fetch_stored
    "${MANIFEST[@]}" compatible "$FOLDER/SHA256SUMS" "$WORK/stored" \
      || { echo "$URI already holds the files above with other bytes: this is another run of the same name; rename the folder" >&2; exit 1; }
    # The archive's manifest first, as the union of what it held and this folder: whatever stops the copy below, the archive
    # still names the bytes each of its files must have.
    "${MANIFEST[@]}" merge "$FOLDER/SHA256SUMS" "$WORK/stored" "$WORK/merged"
    "${AWS[@]}" s3 cp "$WORK/merged" "$URI/SHA256SUMS" --storage-class "$ARCHIVE_STORAGE_CLASS" --checksum-algorithm SHA256 --no-progress
    # Every file is sent, every time (`s3 sync` skips same-size files with older times, review round 33). The manifest is
    # left out here: the archive's own was sent above. S3 checks each part's SHA-256 on arrival.
    "${AWS[@]}" s3 cp --recursive "$FOLDER" "$URI" --storage-class "$ARCHIVE_STORAGE_CLASS" --checksum-algorithm SHA256 \
      --exclude known_hosts --exclude ARCHIVED --exclude SHA256SUMS --no-progress
    # The bytes sent are the bytes hashed: nothing changed here between the manifest and the end of the copy.
    "${MANIFEST[@]}" verify "$FOLDER" "$FOLDER/SHA256SUMS" > /dev/null || { echo "$FOLDER changed during the upload: run it again" >&2; exit 1; }
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
    fetch_stored
    "${MANIFEST[@]}" covered "$FOLDER/SHA256SUMS" "$WORK/stored" || { echo "the archive's manifest does not name these files with these bytes: nothing pruned" >&2; exit 1; }
    check_listed
    # Only folders at the trainer's own places whose every file, read now, has the archived hash (review round 37: a
    # folder copied in after the checks above, or a file added to one, keeps it).
    "${MANIFEST[@]}" prunable "$FOLDER" "$FOLDER/SHA256SUMS" > "$WORK/prunable" || { echo 'nothing pruned' >&2; exit 1; }
    while IFS= read -r stored; do echo "$FOLDER/$stored"; rm -rf "${FOLDER:?}/$stored"; done < "$WORK/prunable"
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
