"""SHA-256 manifest of a results folder, its check, and its file list (ops/s3_archive.sh).

Standard library only (Python 3.11 or later, for hashlib.file_digest), so a copy fetched back from the archive can be
checked anywhere, a pod included. Usage:
  python3 sha256_manifest.py write <folder> <manifest file>   # every file of the run, in `sha256sum` format
  python3 sha256_manifest.py verify <folder> <manifest file>  # exit 1 naming each missing or differing file
  python3 sha256_manifest.py listed <folder> <manifest file>  # 'path size' per file; exit 1 if the folder and manifest differ
  python3 sha256_manifest.py compatible <manifest> <archived manifest>  # exit 1 if a file both name has another hash
  python3 sha256_manifest.py prunable <folder> <manifest file>  # snapshot and checkpoint folders archived exactly as they are
Files are hashed in a thread pool (hashlib releases the GIL on large reads); results keep the sorted path order.
"""
import hashlib
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

WORKERS = 8  # a network volume serves several streams faster than one; a laptop disk is not slowed by eight


def digest(path):
    with open(path, 'rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


# Files at the top of a results folder that are not part of the run: the manifest itself, the archive's marker, and the
# laptop's SSH trust for the job's pod. Only these names, only at the top (ops/s3_archive.sh excludes the same).
NOT_THE_RUN = ('SHA256SUMS', 'ARCHIVED', 'known_hosts')


def run_files(root):
    """The run's regular files under root, sorted. A symbolic link is refused: it would be archived as its target, unchecked."""
    links = [p for p in root.rglob('*') if p.is_symlink()]
    if links:
        raise SystemExit(f'{root} holds symbolic links, which the archive cannot check: {links[:3]}')
    return sorted(p for p in root.rglob('*') if p.is_file() and p.relative_to(root).as_posix() not in NOT_THE_RUN)


def write(folder, manifest):
    """Every file of the run under folder, hashed; 0 once written."""
    root = Path(folder)
    paths = run_files(root)
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


def listed(folder, manifest):
    """Each file of the run with its size, as 'path size' lines, after checking that the manifest names exactly these files
    (a file added since it was written, or one gone, is refused: timestamps do not show it). 1 on a mismatch."""
    root = Path(folder)
    files = {p.relative_to(root).as_posix(): p.stat().st_size for p in run_files(root)}
    names = {line.split('  ', 1)[1] for line in Path(manifest).read_text().splitlines()}
    if names != set(files):
        print(f'the manifest and the folder differ: only in the folder {sorted(set(files) - names)[:5]}, '
              f'only in the manifest {sorted(names - set(files))[:5]}; write the manifest again')
        return 1
    print(''.join(f'{name} {size}\n' for name, size in sorted(files.items())), end='')
    return 0


def entries(manifest):
    return dict(reversed(line.split('  ', 1)) for line in Path(manifest).read_text().splitlines())


def compatible(manifest, stored):
    """0 if every file named in both manifests has the same hash: an upload may add files to an archived run, never replace
    archived bytes (review round 37). 1, naming the files, otherwise."""
    ours, theirs = entries(manifest), entries(stored)
    clashes = sorted(name for name in ours.keys() & theirs.keys() if ours[name] != theirs[name])
    for name in clashes[:20]:
        print(f'ARCHIVED DIFFERENTLY {name}')
    return 1 if clashes else 0


def merge(manifest, stored, out):
    """The archive's manifest after an upload: what it held and this folder's files (compatible() checked first)."""
    union = {**entries(stored), **entries(manifest)}
    Path(out).write_text(''.join(f'{union[name]}  {name}\n' for name in sorted(union)))
    return 0


def covered(manifest, stored):
    """0 if the archive's manifest names every file of this one with the same hash; 1, naming the rest, otherwise."""
    ours, theirs = entries(manifest), entries(stored)
    missing = sorted(name for name in ours if theirs.get(name) != ours[name])
    for name in missing[:20]:
        print(f'NOT IN THE ARCHIVE AS IT IS {name}')
    return 1 if missing or not ours else 0


# Where the trainer writes what prune may delete (hf_grpo: adapter-step{global_step}; the trainer's checkpoint-{step}), one
# run folder down, digits only.
PRUNABLE = re.compile(r'[^/]+/(?:adapter-step|trainer/checkpoint-)\d+')


def prunable(folder, manifest):
    """The snapshot and checkpoint folders whose every file is in the manifest with its hash, read now, as paths under
    folder; exit 1 naming any that hold a file the manifest lacks or one that differs (review round 37)."""
    root, named = Path(folder), entries(manifest)
    found, problems = [], []
    for candidate in sorted({'/'.join(name.split('/')[:3 if '/trainer/' in name else 2]) for name in named}):
        if PRUNABLE.fullmatch(candidate) and (root/candidate).is_dir():
            for path in sorted(p for p in (root/candidate).rglob('*') if not p.is_dir()):
                name = path.relative_to(root).as_posix()
                if path.is_symlink() or named.get(name) != digest(path):
                    problems.append(name)
            found.append(candidate)
    for name in problems[:20]:
        print(f'NOT ARCHIVED AS IT IS {name}', file=sys.stderr)
    if problems:
        return 1
    print(''.join(f'{candidate}\n' for candidate in found), end='')
    return 0


# Each command with the number of paths it takes.
COMMANDS = {'write': (write, 2), 'verify': (verify, 2), 'listed': (listed, 2), 'compatible': (compatible, 2),
            'merge': (merge, 3), 'covered': (covered, 2), 'prunable': (prunable, 2)}

if __name__ == '__main__':
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS or len(sys.argv) != 2 + COMMANDS[sys.argv[1]][1]:
        sys.exit(__doc__)
    sys.exit(COMMANDS[sys.argv[1]][0](*sys.argv[2:]))
