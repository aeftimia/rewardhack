"""JSON Lines, read and written one way: one JSON value per line.

Reading. A line holding only whitespace is skipped, not an error: this is what nearly every reader here did
before they shared this module, a trailing blank line is what an editor leaves in a hand-written labels file,
and positional ids (`<run>#<row index>`) have always counted the rows, not the lines. A line that is not JSON
raises `MalformedLine` naming the source and the 1-based line number. Lines end at a newline only (`\\n`,
`\\r\\n` or `\\r`), never at the other separators `str.splitlines` knows: a row written with
`ensure_ascii=False` may carry U+2028 or U+0085 inside a string.

Writing. One row per line, `ensure_ascii=False`, the standard `json.dumps` separators, flushed. Files written
before this module with the `json.dumps` default say so with `ensure_ascii=True`, so their bytes do not change.
"""
import io
import json


class MalformedLine(ValueError):
    """A line of a JSON Lines source is not JSON."""


def parse(lines, source, object_pairs_hook=None):
    """Yield the row of each non-blank line of `lines`; `source` names them in an error. An error raised by
    `object_pairs_hook` is the hook's own finding and passes through unchanged."""
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            yield json.loads(line, object_pairs_hook=object_pairs_hook)
        except json.JSONDecodeError as error:
            raise MalformedLine(f'line {number} of {source} is not JSON: {error.msg} at column {error.colno}') from error


def read(path, object_pairs_hook=None):
    """Yield the rows of the file at `path`."""
    with open(path) as handle:
        yield from parse(handle, path, object_pairs_hook)


def read_finished(data, source):
    """The rows of `data`, the bytes of a file a process may still be writing or that a killed process left: every line
    but an unterminated last one, and whether there was one. Each row is written with its newline, so a last line
    without one is a write cut short, possibly inside a multi-byte character: the bytes are split at the last newline
    before anything is decoded. Every other line is parsed as strictly as `read` does."""
    complete, _, tail = data.rpartition(b'\n')
    return list(loads(complete.decode(), source)), bool(tail.strip())


def loads(text, source, object_pairs_hook=None):
    """Yield the rows of `text` already in memory, for content that is also hashed or came from a process."""
    return parse(io.StringIO(text, newline=None), source, object_pairs_hook)


def dumps(rows, ensure_ascii=False):
    """The text of `rows`, one line each."""
    return ''.join(json.dumps(row, ensure_ascii=ensure_ascii)+'\n' for row in rows)


def write(sink, row, ensure_ascii=False):
    """Append one row to the open text file `sink` and flush, so a stopped run keeps every row written."""
    sink.write(dumps([row], ensure_ascii))
    sink.flush()


def write_rows(sink, rows, ensure_ascii=False):
    """Append `rows` in one write, then flush: a batch is on disk whole or, short of a crash inside the write, not at all."""
    sink.write(dumps(rows, ensure_ascii))
    sink.flush()
