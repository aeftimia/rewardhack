"""SHA-256 manifest of a folder, and its check: run on the laptop to write one and on a pod to verify it (ops/volume_upload.sh).

Standard library only (Python 3.11 or later, for hashlib.file_digest), so it runs on any pod with a python3 and is sent
there over SSH on stdin. Usage:
  python3 sha256_manifest.py write <folder> <manifest file>   # every file under the folder, in `sha256sum` format
  python3 sha256_manifest.py verify <folder> <manifest file>  # exit 1 naming each missing or differing file
Files are hashed in a thread pool (hashlib releases the GIL on large reads); results keep the sorted path order.
"""
import hashlib
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

WORKERS = 8  # a network volume serves several streams faster than one; a laptop disk is not slowed by eight


def digest(path):
    with open(path, 'rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def write(folder, manifest):
    """Every regular file under folder (not the manifest itself), hashed; 0 once written."""
    root = Path(folder)
    skip = Path(manifest).resolve()
    paths = sorted(p for p in root.rglob('*') if p.is_file() and not p.is_symlink() and p.resolve() != skip)
    with ThreadPoolExecutor(WORKERS) as pool:
        digests = list(pool.map(digest, paths))
    Path(manifest).write_text(''.join(f'{d}  {p.relative_to(root).as_posix()}\n' for d, p in zip(digests, paths, strict=True)))
    print(f'{len(paths)} files, {sum(p.stat().st_size for p in paths) / 1e9:.2f} GB')
    return 0


def verify(folder, manifest):
    """1 if any listed file is missing or differs, or the manifest lists nothing; 0 when every one matches."""
    entries = [line.split('  ', 1) for line in Path(manifest).read_text().splitlines()]
    root = Path(folder)

    def check(entry):
        expected, name = entry
        path = root / name
        if not path.is_file():
            return f'MISSING {name}'
        return None if digest(path) == expected else f'DIFFERS {name}'

    with ThreadPoolExecutor(WORKERS) as pool:
        problems = [problem for problem in pool.map(check, entries) if problem]
    for problem in problems:
        print(problem)
    print(f'{len(entries) - len(problems)} of {len(entries)} files match')
    return 1 if problems or not entries else 0


COMMANDS = {'write': write, 'verify': verify}

if __name__ == '__main__':
    if len(sys.argv) != 4 or sys.argv[1] not in COMMANDS:
        sys.exit(__doc__)
    sys.exit(COMMANDS[sys.argv[1]](sys.argv[2], sys.argv[3]))
