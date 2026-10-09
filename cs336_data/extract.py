"""HTML -> plain text extraction (problem ``extract_text``)."""

from __future__ import annotations

from resiliparse.extract.html2text import extract_plain_text
from resiliparse.parse.encoding import detect_encoding


def decode_html_bytes(html_bytes: bytes) -> str:
    """Decode raw HTML bytes to a Unicode string.

    UTF-8 is tried first (the overwhelmingly common case). If that fails, the
    encoding is sniffed with Resiliparse's ``detect_encoding`` and the bytes are
    decoded leniently so that a few bad bytes never drop a whole page.
    """
    try:
        return html_bytes.decode("utf-8")
    except UnicodeDecodeError:
        pass

    encoding = detect_encoding(html_bytes)
    if encoding:
        try:
            return html_bytes.decode(encoding, errors="replace")
        except LookupError:
            pass
    return html_bytes.decode("utf-8", errors="replace")


def extract_text_from_html_bytes(html_bytes: bytes) -> str:
    """Extract the visible plain text from a byte string containing HTML."""
    html = decode_html_bytes(html_bytes)
    return extract_plain_text(html)
