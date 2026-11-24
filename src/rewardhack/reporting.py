"""Versioned report envelopes; prose remains the historical default."""
import json

from .json_objects import unique_object

MODES = ('prose', 'typed')


def validate_mode(mode):
    if mode not in MODES:
        raise ValueError(f'unknown report mode {mode!r}; choose from {MODES}')
    return mode


def add_arguments(parser):
    parser.add_argument('--report-mode', choices=MODES, default='prose',
                        help='prose judge reading (default), or a strict Boolean error flag')


def parse_typed(raw):
    """Read one final envelope, never salvage a draft or nested object.

    Reasoning may precede it but cannot contain braces. Duplicate keys, trailing
    text, extra fields and coerced booleans are invalid, even if Lean would pass.
    """
    start = raw.find('{')
    if start < 0 or '}' in raw[:start]:
        return None
    try:
        value = json.loads(raw[start:], object_pairs_hook=unique_object)
    except (ValueError, RecursionError):
        return None
    fields = ('established', 'proof_body', 'justification')
    if (not isinstance(value, dict) or set(value) != {*fields, 'error'}
            or type(value['error']) is not bool
            or not all(isinstance(value[k], str) for k in fields)):
        return None
    try:
        for key in fields:
            value[key].encode('utf-8')
    except UnicodeEncodeError:
        return None  # escaped lone surrogates cannot be written to Lean or JSONL sinks
    return dict(value, report=value['justification'])
