#!/usr/bin/env bash
# Drive an already-running RunPod pod over SSH: sync the repo, start the remote script, pull results
# until DONE or the deadline. The pod has no OS shutdown timer, so termination is done in the console
# by whoever launched it; this script prints a loud reminder on every exit.
# Usage: drive_runpod.sh <ssh host> <ssh port> <remote script> <results dir> <deadline minutes>
set -uo pipefail
cd "$(dirname "$0")/.."
HOST=$1 PORT=$2 REMOTE=$3 RESULTS=$4 DEADLINE=$5
mkdir -p "$RESULTS"
SSH=(ssh -p "$PORT" -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=.tools/known_hosts -o ConnectTimeout=10)
pull() { rsync -az --partial -e "${SSH[*]}" "root@$HOST:out/" "$RESULTS/" 2>/dev/null; }
finish() { echo "copying results"; pull; date -u +%FT%TZ > "$RESULTS/driver-exit.txt"; echo "REMINDER: terminate the pod in the RunPod console now"; }
trap finish EXIT
for _ in $(seq 20); do "${SSH[@]}" "root@$HOST" true 2>/dev/null && break; sleep 15; done
"${SSH[@]}" "root@$HOST" true || { echo 'ssh failed'; exit 1; }
rsync -az --exclude .tools --exclude spending --exclude results -e "${SSH[*]}" ./ "root@$HOST:pilot-repo/"
"${SSH[@]}" "root@$HOST" "nohup bash pilot-repo/$REMOTE > remote.log 2>&1 < /dev/null & disown"
date -u +%FT%TZ > "$RESULTS/started-at.txt"
END=$((SECONDS+DEADLINE*60))
while ((SECONDS < END)); do
  sleep 60
  "${SSH[@]}" "root@$HOST" 'test -f out/DONE' 2>/dev/null && { echo 'remote finished'; exit 0; }
  ((SECONDS % 300 < 60)) && pull
done
echo 'driver deadline reached'
exit 1
