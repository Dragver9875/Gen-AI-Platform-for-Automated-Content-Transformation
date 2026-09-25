from __future__ import annotations

import base64
import binascii


def encode_query_b64(value: str) -> str:
    return base64.b64encode(value.encode("utf-8")).decode("ascii")


def decode_query_b64(value: str) -> str:
    try:
        return base64.b64decode(value, validate=True).decode("utf-8")
    except (binascii.Error, UnicodeDecodeError) as exc:
        raise ValueError(f"Invalid UTF-8 Base64 query: {exc}") from exc
