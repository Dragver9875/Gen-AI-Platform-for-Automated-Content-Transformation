from __future__ import annotations

from scripts.query_transport import decode_query_b64, encode_query_b64


def test_query_base64_round_trip_preserves_quotes_spaces_unicode_and_newlines():
    query = 'Make it formal "I finished a project today" into LinkedIn post.\nHindi: परियोजना पूरी हुई'
    assert decode_query_b64(encode_query_b64(query)) == query
