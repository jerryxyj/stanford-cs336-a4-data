"""Language identification with the fastText ``lid.176`` model (problem ``language_identification``)."""

from __future__ import annotations

from cs336_data.classifiers import LID_MODEL, classifier_path, load_fasttext_model, predict_label

# lid.176 uses ISO 639 codes; a few are mapped to the identifiers expected downstream.
_LANGUAGE_ALIASES = {
    "zh-cn": "zh",
    "zh-tw": "zh",
    "zh-hans": "zh",
    "zh-hant": "zh",
}


def identify_language(text: str) -> tuple[str, float]:
    """Return ``(language_code, confidence)`` for the main language of ``text``.

    The language code is a (mostly ISO 639-1) identifier such as ``"en"`` or ``"zh"``
    and the confidence is the classifier probability, clipped to ``[0, 1]``.
    """
    model = load_fasttext_model(classifier_path(LID_MODEL))
    label, score = predict_label(model, text)
    return _LANGUAGE_ALIASES.get(label, label), score


def english_probability(text: str) -> float:
    """Probability that ``text`` is English (0 if English is not the top prediction)."""
    language, score = identify_language(text)
    return score if language == "en" else 0.0


def is_english(text: str, threshold: float = 0.7) -> bool:
    return english_probability(text) >= threshold
