"""Strict JSON object decoding shared by model-output readers."""

class DuplicateReplyKey(ValueError):
    """A decoded object repeated a key; it must never be skipped as a prose aside."""


def unique_object(pairs):
    """Reject duplicate keys instead of silently replacing an earlier field."""
    result = {}
    for key, value in pairs:
        if key in result:
            raise DuplicateReplyKey(f'duplicate JSON key: {key}')
        result[key] = value
    return result

