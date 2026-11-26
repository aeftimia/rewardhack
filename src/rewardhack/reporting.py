"""Versioned report envelopes; prose remains the historical default."""
from pydantic import BaseModel

from .json_objects import STRICT, Utf8Text, strict_object

MODES = ('prose', 'typed')
# What a run that predates report modes used: every such row is prose (README; the loader in provenance.py relies on it).
HISTORICAL_MODE = 'prose'


def validate_mode(mode):
    if mode not in MODES:
        raise ValueError(f'unknown report mode {mode!r}; choose from {MODES}')
    return mode


def add_arguments(parser):
    parser.add_argument('--report-mode', choices=MODES, default='prose',
                        help='prose judge reading (default), or a strict Boolean error flag')


class TypedReport(BaseModel):
    """The typed envelope: exactly these four keys, a JSON Boolean flag, and text that UTF-8 can hold (an escaped
    lone surrogate cannot be written to Lean or to a JSONL sink)."""
    model_config = STRICT
    established: Utf8Text
    proof_body: Utf8Text
    error: bool
    justification: Utf8Text


def parse_typed(raw):
    """Read one final envelope, never salvage a draft or nested object.

    Reasoning may precede it but cannot contain braces. Duplicate keys, trailing
    text, extra fields and coerced booleans are invalid, even if Lean would pass.
    """
    start = raw.find('{')
    if start < 0 or '}' in raw[:start]:
        return None
    envelope, unreadable = strict_object(raw[start:], TypedReport)
    if unreadable is not None:
        return None  # an invalid envelope is scored as no answer (`outcomes.score`)
    return dict(envelope.model_dump(), report=envelope.justification)
