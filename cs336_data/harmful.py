"""NSFW and toxic-speech detection with Dolma's Jigsaw fastText models (problem ``harmful_content``)."""

from __future__ import annotations

from cs336_data.classifiers import (
    HATESPEECH_MODEL,
    NSFW_MODEL,
    classifier_path,
    load_fasttext_model,
    predict_label,
    predict_proba,
)

NSFW_LABEL = "nsfw"
NON_NSFW_LABEL = "non-nsfw"
TOXIC_LABEL = "toxic"
NON_TOXIC_LABEL = "non-toxic"


def classify_nsfw(text: str) -> tuple[str, float]:
    """Label ``text`` as ``"nsfw"`` or ``"non-nsfw"`` with the classifier's confidence."""
    return predict_label(load_fasttext_model(classifier_path(NSFW_MODEL)), text)


def classify_toxic_speech(text: str) -> tuple[str, float]:
    """Label ``text`` as ``"toxic"`` or ``"non-toxic"`` with the classifier's confidence."""
    return predict_label(load_fasttext_model(classifier_path(HATESPEECH_MODEL)), text)


def nsfw_probability(text: str) -> float:
    return predict_proba(load_fasttext_model(classifier_path(NSFW_MODEL)), text).get(NSFW_LABEL, 0.0)


def toxic_probability(text: str) -> float:
    return predict_proba(load_fasttext_model(classifier_path(HATESPEECH_MODEL)), text).get(TOXIC_LABEL, 0.0)
