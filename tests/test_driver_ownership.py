"""Execute the driver's remote guards against a local pod whose output changes owners."""
import os
import subprocess

import pytest

from conftest import TIMEOUT_STUB, driver_script


def executable(path, body):
    path.write_text('#!/bin/bash\nset -eu\n' + body)
    path.chmod(0o755)


def drive_with_ownership_change(tmp_path, change):
    pod = tmp_path/'pod'
    (pod/'out').mkdir(parents=True)
    local_bin, remote_bin = tmp_path/'bin', tmp_path/'remote-bin'
    local_bin.mkdir()
    remote_bin.mkdir()
    destination = tmp_path/'results'
    executable(local_bin/'sleep', 'exit 0\n')
    if change != 'slow-copy':
        executable(local_bin/'timeout', TIMEOUT_STUB)
    executable(local_bin/'ssh', '''
command=${!#}
printf '%s\\n' "$command" >> "$POD/ssh-calls"
[[ ! -e $POD/removed ]] || exit 255
if [[ $command == *--arm-only* ]]; then exit 0; fi
if [[ $command == *nohup* ]]; then
  read -r token < "$DESTINATION/launch-id.txt"
  printf '%s\\n' "$token" > "$POD/out/LAUNCH_ID"
  if [[ $CHANGE != slow-copy ]]; then printf '0\\n' > "$POD/out/EXIT_CODE"; fi
  printf 'own rows\\n' > "$POD/out/rows.jsonl"
  exit 0
fi
if [[ $CHANGE == slow-copy && $command == *'cat EXIT_CODE'* && -f $POD/interim-started ]]; then
  printf '0\\n' > "$POD/out/EXIT_CODE"
fi
cd "$POD"
export HOME="$POD" PATH="$REMOTE_BIN:$PATH"
bash -c "$command"
if [[ $command == *'for i in '* && $CHANGE == after-receipt ]]; then
  printf 'other-launch\\n' > out/LAUNCH_ID
  printf 'other rows\\n' > out/rows.jsonl
fi
''')
    executable(local_bin/'rsync', '''
for arg in "$@"; do
  if [[ $arg == --rsync-path=* ]]; then
    command=${arg#--rsync-path=}
    if [[ $CHANGE == slow-copy && ! -f $POD/interim-started ]]; then
      touch "$POD/interim-running" "$POD/interim-started"
      trap 'rm "$POD/interim-running"; touch "$POD/interim-stopped"' EXIT
      # Continuous progress would defeat an inactivity timeout. Completion must
      # still be observed and cancel this transfer before final copying begins.
      for i in {1..200}; do
        echo progress >> "$POD/progress"
        /bin/sleep 0.1
      done
      touch "$POD/interim-finished-naturally"
      exit 0
    fi
    if [[ -f $POD/interim-running ]]; then touch "$POD/concurrent-copy"; fi
    printf '%s\\n' "$command" >> "$POD/copy-commands"
    if [[ $CHANGE == before-copy ]]; then
      printf 'other-launch\\n' > "$POD/out/LAUNCH_ID"
      printf 'other rows\\n' > "$POD/out/rows.jsonl"
    fi
    cd "$POD"
    export PATH="$REMOTE_BIN:$PATH"
    exec bash -c "$command"
  fi
done
exit 0
''')
    executable(remote_bin/'rsync', '''
printf 'transfer\\n' >> "$POD/transfers"
if [[ $CHANGE == during-copy ]]; then
  mv "$POD/out" "$POD/out.previous"
  mkdir "$POD/out"
  printf 'other-launch\\n' > "$POD/out/LAUNCH_ID"
  printf 'other rows\\n' > "$POD/out/rows.jsonl"
fi
cp rows.jsonl "$DESTINATION/rows.jsonl"
if [[ $CHANGE == after-copy && $(wc -l < "$POD/transfers") -eq 3 ]]; then
  printf 'other-launch\\n' > "$POD/out/LAUNCH_ID"
  printf 'other rows\\n' > "$POD/out/rows.jsonl"
fi
''')
    # This suite exercises the real ownership guard. Timer lifecycle has its own
    # process tests; acknowledge only its exact guarded removal command here.
    executable(remote_bin/'bash', '''
if [[ $* == '-s -- --remove-now' ]]; then
  cat > /dev/null
  touch "$POD/removed"
  exit 0
fi
exec /bin/bash "$@"
''')
    executable(remote_bin/'runpodctl', '''
[[ $1 == remove && $2 == pod ]] || exit 2
touch "$POD/removed"
''')
    env = dict(os.environ, PATH=f"{local_bin}:{os.environ['PATH']}", POD=str(pod), REMOTE_BIN=str(remote_bin),
               DESTINATION=str(destination), CHANGE=change, JUDGE_KIND='pod', FORWARD_JUDGE_KEY='0',
               LEAN_ISOLATION='none', KEEP_POD='0', RUNPOD_POD_ID='fake-pod', RUNPOD_API_KEY='fake-key')
    result = subprocess.run(['bash', driver_script(tmp_path), 'example.invalid', '22', 'ops/remote_runpod_smoke.sh',
                             str(destination), '240'], env=env, capture_output=True, text=True, timeout=30)
    return result, pod, destination


@pytest.mark.parametrize('change', ['after-receipt', 'before-copy', 'during-copy', 'after-copy'])
def test_losing_output_ownership_never_transfers_or_removes_the_other_run(tmp_path, change):
    result, pod, destination = drive_with_ownership_change(tmp_path, change)
    assert result.returncode != 0, result.stdout + result.stderr
    assert not (pod/'removed').exists(), result.stdout + result.stderr
    assert (pod/'out'/'rows.jsonl').read_text() == 'other rows\n'
    if change in ('during-copy', 'after-copy'):
        # A copy already in progress stays in the old directory even if the shared path is replaced.
        copies = 1 if change == 'during-copy' else 3
        assert (pod/'transfers').read_text().splitlines() == ['transfer'] * copies
        assert (destination/'rows.jsonl').read_text() == 'own rows\n'
    else:
        assert not (pod/'transfers').exists()
        assert not (destination/'rows.jsonl').exists()
    if change == 'before-copy':
        # This exercised rsync's real remote command, not a stub choosing a failure status from the launch id.
        assert (pod/'copy-commands').read_text()


def test_matching_output_ownership_allows_copy_and_removal(tmp_path):
    result, pod, destination = drive_with_ownership_change(tmp_path, 'never')
    assert result.returncode == 0, result.stdout + result.stderr
    assert (destination/'rows.jsonl').read_text() == 'own rows\n'
    assert (pod/'transfers').read_text().splitlines() == ['transfer'] * 3
    assert (pod/'removed').exists()


def test_progressing_interim_copy_does_not_block_completion_or_race_final_copy(tmp_path):
    result, pod, destination = drive_with_ownership_change(tmp_path, 'slow-copy')
    assert result.returncode == 0, result.stdout + result.stderr
    assert (pod/'interim-started').exists()
    assert (pod/'interim-stopped').exists()
    assert not (pod/'interim-finished-naturally').exists()
    assert not (pod/'concurrent-copy').exists()
    assert (destination/'rows.jsonl').read_text() == 'own rows\n'
    assert (pod/'transfers').read_text().splitlines() == ['transfer'] * 3
    assert (pod/'removed').exists()
