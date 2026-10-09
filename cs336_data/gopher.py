"""Rule-based quality filters from the Gopher paper (Rae et al., 2021, Appendix A)."""

from __future__ import annotations

from dataclasses import dataclass

MIN_WORDS = 50
MAX_WORDS = 100_000
MIN_MEAN_WORD_LENGTH = 3.0
MAX_MEAN_WORD_LENGTH = 10.0
MAX_ELLIPSIS_LINE_FRACTION = 0.3
MIN_ALPHA_WORD_FRACTION = 0.8

_ELLIPSES = ("...", "\u2026")


def tokenize_words(text: str) -> list[str]:
    """Whitespace tokenisation; fast and dependency-free (no NLTK data download needed)."""
    return text.split()


@dataclass(frozen=True)
class GopherStats:
    num_words: int
    mean_word_length: float
    ellipsis_line_fraction: float
    alpha_word_fraction: float

    def failed_rules(self) -> list[str]:
        failed = []
        if self.num_words < MIN_WORDS:
            failed.append("too_few_words")
        if self.num_words > MAX_WORDS:
            failed.append("too_many_words")
        if not MIN_MEAN_WORD_LENGTH <= self.mean_word_length <= MAX_MEAN_WORD_LENGTH:
            failed.append("mean_word_length")
        if self.ellipsis_line_fraction > MAX_ELLIPSIS_LINE_FRACTION:
            failed.append("ellipsis_lines")
        if self.alpha_word_fraction < MIN_ALPHA_WORD_FRACTION:
            failed.append("non_alpha_words")
        return failed

    @property
    def passes(self) -> bool:
        return not self.failed_rules()


def compute_gopher_stats(text: str) -> GopherStats:
    words = tokenize_words(text)
    num_words = len(words)
    if num_words == 0:
        return GopherStats(0, 0.0, 0.0, 0.0)

    mean_word_length = sum(len(w) for w in words) / num_words
    alpha_word_fraction = sum(1 for w in words if any(c.isalpha() for c in w)) / num_words

    lines = [line.rstrip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    if lines:
        ellipsis_line_fraction = sum(1 for line in lines if line.endswith(_ELLIPSES)) / len(lines)
    else:
        ellipsis_line_fraction = 0.0

    return GopherStats(num_words, mean_word_length, ellipsis_line_fraction, alpha_word_fraction)


def gopher_quality_filter(text: str) -> bool:
    """Return ``True`` if ``text`` passes all Gopher quality rules implemented here.

    A document is rejected if it
      * has fewer than 50 or more than 100,000 words,
      * has a mean word length outside [3, 10] characters,
      * has more than 30% of its lines ending in an ellipsis, or
      * has fewer than 80% of words containing at least one alphabetic character.
    """
    return compute_gopher_stats(text).passes
