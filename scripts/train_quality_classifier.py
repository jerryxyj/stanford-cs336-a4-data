"""Train the fastText quality classifier (problem ``quality_classifier``).

Reads ``train.txt`` / ``valid.txt`` produced by ``scripts/build_quality_classifier_data.py`` and
writes the model to ``<shared>/classifiers/quality_classifier.bin`` (the path used by
``cs336_data.quality.classify_quality``).

Example:
    uv run scripts/train_quality_classifier.py --epoch 10 --lr 0.5 --word-ngrams 2
"""

from __future__ import annotations

import argparse
from pathlib import Path

from cs336_data.classifiers import QUALITY_MODEL, classifier_path
from cs336_data.common import get_shared_assets_path
from cs336_data.quality import train_quality_classifier


def main() -> None:
    shared = get_shared_assets_path()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=shared / "quality_classifier")
    parser.add_argument("--output", type=Path, default=classifier_path(QUALITY_MODEL))
    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.5)
    parser.add_argument("--dim", type=int, default=100)
    parser.add_argument("--word-ngrams", type=int, default=2)
    parser.add_argument("--min-count", type=int, default=3)
    parser.add_argument("--bucket", type=int, default=500_000)
    parser.add_argument(
        "--quantize", action="store_true", help="Quantize the model to shrink it (slightly less accurate)"
    )
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    train_path = args.data_dir / "train.txt"
    valid_path = args.data_dir / "valid.txt"
    if not train_path.exists():
        raise SystemExit(f"{train_path} not found; run scripts/build_quality_classifier_data.py first")

    model = train_quality_classifier(
        train_path,
        args.output,
        valid_path=valid_path if valid_path.exists() else None,
        epoch=args.epoch,
        lr=args.lr,
        dim=args.dim,
        word_ngrams=args.word_ngrams,
        min_count=args.min_count,
        bucket=args.bucket,
        quantize=args.quantize,
        threads=args.threads,
        seed=args.seed,
    )
    print(f"saved {args.output} ({args.output.stat().st_size / 1e6:.1f} MB, labels={model.get_labels()})")


if __name__ == "__main__":
    main()
