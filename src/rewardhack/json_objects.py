"""Strict JSON object decoding shared by model-output readers."""
import json
from typing import Annotated

from pydantic import AfterValidator, ConfigDict, ValidationError

# Every field must be present, no other field may be, and no value is coerced (a "true" string is not a bool).
STRICT = ConfigDict(strict=True, extra='forbid')


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


def _utf8(text):
    text.encode('utf-8')  # an escaped lone surrogate decodes to a str no UTF-8 file can hold; UnicodeEncodeError is a ValueError
    return text


Utf8Text = Annotated[str, AfterValidator(_utf8)]
assert issubclass(ValidationError, ValueError)  # what `strict_object` relies on to catch a refused shape


def strict_object(text, model):
    """`(instance, None)` when `text` is exactly one `model` object, else `(None, why)`.

    The whole text must be one JSON object: no prose around it, no trailing text, no repeated key, nothing the
    model does not accept exactly. `why` is the caller's to record as its "unreadable" verdict.
    """
    if not isinstance(text, str):
        return None, f'reply is {type(text).__name__}, not text'
    try:
        return model.model_validate(json.loads(text, object_pairs_hook=unique_object)), None
    except (ValueError, RecursionError) as error:  # decode errors, DuplicateReplyKey and ValidationError
        return None, f'{type(error).__name__}: {error}'[:300]
