"""C4-style line and document heuristics (Raffel et al., 2020, section 2.2).

The evaluation target is the C4 100-domains subset of Paloma, so mirroring C4's cleaning
rules on the WET text is a cheap way to make our training distribution closer to it:

* keep only lines that end in terminal punctuation and have at least ``min_words_per_line`` words,
* drop lines containing boilerplate markers (e.g. "javascript", "cookies"),
* drop documents containing "lorem ipsum" or a curly brace (code),
* drop documents that end up with fewer than ``min_sentences`` sentences.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

TERMINAL_PUNCTUATION = (".", "!", "?", '"', "\u201d", "\u2019", "'")
_SENTENCE_END_RE = re.compile(r"[.!?][\"\u201d\u2019')\]]*(?:\s|$)")

DEFAULT_LINE_BLOCKLIST = (
    "javascript",
    "cookie",
    "cookies",
    "privacy policy",
    "terms of use",
    "terms of service",
    "all rights reserved",
    "click here",
    "sign in",
    "log in",
    "sign up",
    "subscribe",
    "newsletter",
    "read more",
    "share this",
    "powered by",
)
_LINE_BLOCKLIST_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(w) for w in DEFAULT_LINE_BLOCKLIST) + r")\b", re.IGNORECASE
)
_LOREM_IPSUM_RE = re.compile(r"lorem ipsum", re.IGNORECASE)


@dataclass(frozen=True)
class C4Result:
    text: str | None
    rejected_by: str | None
    lines_in: int
    lines_kept: int


def count_sentences(text: str) -> int:
    return len(_SENTENCE_END_RE.findall(text))


def c4_clean(
    text: str,
    *,
    min_words_per_line: int = 5,
    min_sentences: int = 3,
    drop_code: bool = True,
    filter_lines: bool = True,
) -> C4Result:
    """Apply C4's line-level and document-level heuristics to ``text``."""
    if _LOREM_IPSUM_RE.search(text):
        return C4Result(None, "c4_lorem_ipsum", 0, 0)
    if drop_code and "{" in text:
        return C4Result(None, "c4_curly_brace", 0, 0)

    raw_lines = text.split("\n")
    kept_lines: list[str] = []
    paragraph_break_pending = False
    for raw_line in raw_lines:
        line = raw_line.strip()
        if not line:
            paragraph_break_pending = bool(kept_lines)
            continue
        if filter_lines:
            if not line.endswith(TERMINAL_PUNCTUATION):
                continue
            if len(line.split()) < min_words_per_line:
                continue
            if _LINE_BLOCKLIST_RE.search(line):
                continue
        if paragraph_break_pending:
            kept_lines.append("")
            paragraph_break_pending = False
        kept_lines.append(line)

    cleaned = "\n".join(kept_lines).strip()
    if not cleaned:
        return C4Result(None, "c4_no_lines", len(raw_lines), 0)
    if count_sentences(cleaned) < min_sentences:
        return C4Result(None, "c4_too_few_sentences", len(raw_lines), len(kept_lines))
    return C4Result(cleaned, None, len(raw_lines), len(kept_lines))
