"""Digests of files and installed package sources used in run records."""
import hashlib
from pathlib import Path


def file_sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def source_sha256(package=Path(__file__).parent):
    return {path.name: file_sha256(path) for path in sorted(package.glob('*.py'))}
