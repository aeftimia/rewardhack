"""Watch an rsync checkpoint upload against its bounded startup window."""
import argparse
import os
import re
import selectors
import signal
import subprocess
import sys
import time
from pathlib import Path

from rewardhack.time_projection import ProgressProjection


PROGRESS = re.compile(rb'^\s*([\d,]+)\s+\d+%')
TRANSFER = re.compile(rb'\(xfer#(\d+),')
SAMPLE_SECONDS = 20
STALL_SECONDS = 180
FINISH_MARGIN_SECONDS = 60


def source_bytes(source):
    path = Path(source)
    if path.is_file():
        return path.stat().st_size
    return sum(member.stat().st_size for member in path.rglob('*') if member.is_file())


def watch(source, seconds, command):
    total = source_bytes(source)
    if total <= 0:
        raise ValueError(f'checkpoint source has no files: {source}')
    # dry-ok: entrypoints/qwen_length_measure.py::<module> | This baseline times checkpoint transfer; the measurement script's elapsed-time read uses its own baseline and measures a different operation. Both use a standard clock call, not a shared project rule.
    started = time.monotonic()
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               start_new_session=True)
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ)
    completed = current = last_transfer = last_work = 0
    last_progress_at = started
    projection = ProgressProjection(total)
    projection.update(0.0, 0.0)
    next_sample = started + SAMPLE_SECONDS
    buffer = b''
    try:
        while process.poll() is None:
            now = time.monotonic()
            if now - started >= seconds:
                raise TimeoutError(f'checkpoint upload exceeded {seconds}s')
            for key, _ in selector.select(timeout=min(1, seconds - (now - started), max(0, next_sample-now))):
                buffer += os.read(key.fd, 65536)
                parts = re.split(rb'[\r\n]', buffer)
                buffer = parts.pop()
                for part in parts:
                    progress = PROGRESS.search(part)
                    if progress is None:
                        continue
                    amount = int(progress.group(1).replace(b',', b''))
                    transfer = TRANSFER.search(part)
                    if transfer and int(transfer.group(1)) > last_transfer:
                        completed += amount
                        last_transfer = int(transfer.group(1))
                        current = 0
                    elif transfer is None:
                        current = amount
            now = time.monotonic()
            if now >= next_sample:
                work = max(last_work, min(total, completed + current))
                if work > last_work:
                    last_progress_at = now
                if now - last_progress_at >= STALL_SECONDS:
                    raise TimeoutError('checkpoint upload made no progress for three minutes')
                upper = projection.update(float(work), now-started)
                last_work = work
                next_sample += SAMPLE_SECONDS
                if upper is not None and work < total:
                    remaining = seconds - (now-started)
                    print(f'checkpoint upload {work}/{total} bytes; 95% projected finish {upper:.0f}s; '
                          f'allowance {seconds}s', flush=True)
                    if upper + FINISH_MARGIN_SECONDS > seconds:
                        raise TimeoutError(f'checkpoint upload projected past its allowance ({remaining:.0f}s remain)')
        return process.returncode
    finally:
        selector.close()
        process.stdout.close()
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--seconds', type=int, required=True)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.seconds <= 0 or not args.command or args.command[0] != '--' or len(args.command) < 2:
        parser.error('a positive timeout and -- COMMAND are required')
    try:
        return watch(args.source, args.seconds, args.command[1:])
    except TimeoutError as error:
        print(error, file=sys.stderr)
        return 124


if __name__ == '__main__':
    raise SystemExit(main())
