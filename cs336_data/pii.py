"""Masking of personally identifiable information (problem ``mask_pii``)."""

from __future__ import annotations

import re

EMAIL_MASK = "|||EMAIL_ADDRESS|||"
PHONE_MASK = "|||PHONE_NUMBER|||"
IP_MASK = "|||IP_ADDRESS|||"

# Local part, "@", then a dotted domain ending in a 2+ letter TLD.
EMAIL_RE = re.compile(
    r"""
    (?<![\w.+-])                      # don't start mid-token
    [A-Za-z0-9!#$%&'*+/=?^_`{|}~-]+   # local part
    (?:\.[A-Za-z0-9!#$%&'*+/=?^_`{|}~-]+)*
    @
    (?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\.)+  # sub-domains
    [A-Za-z]{2,63}                    # TLD
    (?![\w-])
    """,
    re.VERBOSE,
)

# US-style phone numbers: optional +1 / 1 country code, 3-digit area code with
# optional parentheses, then 3 + 4 digits. Separators may be spaces, dots or
# dashes (also tolerated right after a parenthesised area code, e.g. "(283)-182-3829").
PHONE_RE = re.compile(
    r"""
    (?<![\w.+-])                      # not glued to a preceding word / number
    (?:\+1[ \t.-]?|1[ \t.-])?         # optional country code ("+1" or "1-"/"1 ")
    (?:\(\d{3}\)|\d{3})               # area code
    [ \t.-]*                          # separator (possibly none)
    \d{3}                             # exchange
    [ \t.-]?                          # separator (possibly none)
    \d{4}                             # subscriber number
    (?![\w-])                         # not followed by more digits/letters
    """,
    re.VERBOSE,
)

_OCTET = r"(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)"
IPV4_RE = re.compile(rf"(?<![\w.]){_OCTET}(?:\.{_OCTET}){{3}}(?![\w.]\w)")


def _mask(pattern: re.Pattern[str], text: str, mask: str) -> tuple[str, int]:
    return pattern.subn(mask, text)


def mask_emails(text: str) -> tuple[str, int]:
    """Replace every email address with ``|||EMAIL_ADDRESS|||``."""
    return _mask(EMAIL_RE, text, EMAIL_MASK)


def mask_phone_numbers(text: str) -> tuple[str, int]:
    """Replace every (US-formatted) phone number with ``|||PHONE_NUMBER|||``."""
    return _mask(PHONE_RE, text, PHONE_MASK)


def mask_ips(text: str) -> tuple[str, int]:
    """Replace every IPv4 address with ``|||IP_ADDRESS|||``."""
    return _mask(IPV4_RE, text, IP_MASK)


def mask_pii(text: str) -> tuple[str, dict[str, int]]:
    """Apply all PII maskers, returning the new text and per-type counts."""
    text, n_emails = mask_emails(text)
    text, n_ips = mask_ips(text)
    text, n_phones = mask_phone_numbers(text)
    return text, {"emails": n_emails, "phone_numbers": n_phones, "ips": n_ips}
