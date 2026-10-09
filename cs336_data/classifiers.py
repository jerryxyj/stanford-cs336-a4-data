"""Shared helpers for loading fastText classifiers."""

from __future__ import annotations

import functools
import os
from pathlib import Path

import fasttext

from cs336_data.common import get_shared_assets_path

# Silence fastText's spurious "load_model does not return WordVectorModel ..." warning.
fasttext.FastText.eprint = lambda *args, **kwargs: None

LID_MODEL = "lid.176.bin"
NSFW_MODEL = "dolma_fasttext_nsfw_jigsaw_model.bin"
HATESPEECH_MODEL = "dolma_fasttext_hatespeech_jigsaw_model.bin"
QUALITY_MODEL = "quality_classifier.bin"


def classifiers_dir() -> Path:
    """Directory holding the fastText models (override with ``CS336_CLASSIFIERS_DIR``)."""
    override = os.environ.get("CS336_CLASSIFIERS_DIR")
    if override:
        return Path(override)
    return get_shared_assets_path() / "classifiers"


def classifier_path(name: str) -> Path:
    return classifiers_dir() / name


@functools.cache
def load_fasttext_model(path: str | os.PathLike) -> fasttext.FastText._FastText:
    """Load (and cache per process) a fastText model from disk."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"fastText model not found at {path}. Run `uv run scripts/download_data.py --offline-only` "
            "(or `uv run scripts/train_quality_classifier.py` for the quality model)."
        )
    return fasttext.load_model(str(path))


def prepare_text(text: str) -> str:
    """fastText predicts on a single line, so newlines must be collapsed first."""
    return " ".join(text.split())


def predict_label(model: fasttext.FastText._FastText, text: str) -> tuple[str, float]:
    """Top-1 prediction as ``(label without the __label__ prefix, probability in [0, 1])``."""
    labels, probs = model.predict(prepare_text(text), k=1)
    if not labels:
        return "", 0.0
    label = labels[0].removeprefix("__label__")
    return label, float(min(max(probs[0], 0.0), 1.0))


def predict_proba(model: fasttext.FastText._FastText, text: str) -> dict[str, float]:
    """Probabilities for all labels of the model."""
    labels, probs = model.predict(prepare_text(text), k=-1)
    return {label.removeprefix("__label__"): float(min(max(p, 0.0), 1.0)) for label, p in zip(labels, probs)}
