"""fastText quality classifier: Wikipedia-referenced pages ("wiki") vs. random Common Crawl ("cc").

Problem ``quality_classifier``. Training data is produced by
``scripts/build_quality_classifier_data.py`` and the model by
``scripts/train_quality_classifier.py``.
"""

from __future__ import annotations

import os
from pathlib import Path

import fasttext

from cs336_data.classifiers import (
    QUALITY_MODEL,
    classifier_path,
    load_fasttext_model,
    predict_label,
    predict_proba,
    prepare_text,
)

HIGH_QUALITY_LABEL = "wiki"
LOW_QUALITY_LABEL = "cc"
LABEL_PREFIX = "__label__"


def quality_model_path(model_path: os.PathLike | str | None = None) -> Path:
    """Resolve the quality model: explicit path > ``CS336_QUALITY_MODEL`` env var > default location."""
    if model_path:
        return Path(model_path)
    override = os.environ.get("CS336_QUALITY_MODEL")
    return Path(override) if override else classifier_path(QUALITY_MODEL)


def classify_quality(text: str, model_path: os.PathLike | str | None = None) -> tuple[str, float]:
    """Label ``text`` as ``"wiki"`` (high quality) or ``"cc"`` (low quality) with a confidence."""
    return predict_label(load_fasttext_model(quality_model_path(model_path)), text)


def quality_score(text: str, model_path: os.PathLike | str | None = None) -> float:
    """Probability that ``text`` is high quality, i.e. ``1 - P(cc)``.

    Defined through the negative class so that it works for any positive label
    (``wiki`` for the Wikipedia-reference classifier, ``c4`` for the Paloma-based one).
    """
    probs = predict_proba(load_fasttext_model(quality_model_path(model_path)), text)
    return 1.0 - probs.get(LOW_QUALITY_LABEL, 0.0)


def format_training_example(label: str, text: str, max_chars: int | None = 20_000) -> str:
    """One fastText training line: ``__label__<label> <single-line text>``."""
    text = prepare_text(text)
    if max_chars is not None:
        text = text[:max_chars]
    return f"{LABEL_PREFIX}{label} {text}"


def train_quality_classifier(
    train_path: os.PathLike | str,
    output_path: os.PathLike | str,
    *,
    valid_path: os.PathLike | str | None = None,
    epoch: int = 10,
    lr: float = 0.5,
    dim: int = 100,
    word_ngrams: int = 2,
    min_count: int = 3,
    bucket: int = 500_000,
    quantize: bool = False,
    threads: int | None = None,
    seed: int = 0,
) -> fasttext.FastText._FastText:
    """Train a supervised fastText model on ``__label__wiki`` / ``__label__cc`` lines and save it."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    kwargs = {
        "input": str(train_path),
        "epoch": epoch,
        "lr": lr,
        "dim": dim,
        "wordNgrams": word_ngrams,
        "minCount": min_count,
        "bucket": bucket,
        "loss": "softmax",
        "seed": seed,
        "verbose": 2,
    }
    if threads is not None:
        kwargs["thread"] = threads
    model = fasttext.train_supervised(**kwargs)
    if quantize:
        model.quantize(input=str(train_path), retrain=True, cutoff=200_000, qnorm=True)
    model.save_model(str(output_path))
    if valid_path is not None:
        n, precision, recall = model.test(str(valid_path))
        print(f"validation: n={n} precision@1={precision:.4f} recall@1={recall:.4f}")
    return model
