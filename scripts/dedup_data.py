"""Corpus-level deduplication of filtered documents (section 3, applied to the pipeline output).

Input is a directory of ``*.jsonl.gz`` files written by ``scripts/filter_data.py`` (one JSON
document per line with a ``text`` field). Two steps are applied (see
``cs336_data.corpus_dedup``):

1. **Exact line deduplication** across the whole corpus: lines whose hash occurs more than
   once are removed from every document (boilerplate such as menus, footers and cookie
   banners). Documents left with fewer than ``--min-words`` words are dropped.
2. **MinHash + LSH fuzzy document deduplication** on the line-deduplicated documents: documents
   bucketed together by LSH and whose true word-``n``-gram Jaccard similarity exceeds the
   threshold are clustered, keeping one random document per cluster.

Example:
    uv run scripts/dedup_data.py --input-dir data/filtered --output-dir data/deduped
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from cs336_data.corpus_dedup import run_corpus_dedup


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--glob", default="*.jsonl.gz")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--workers",
        type=int,
        default=len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count(),
    )
    parser.add_argument("--skip-line-dedup", action="store_true")
    parser.add_argument("--skip-minhash", action="store_true")
    parser.add_argument("--min-words", type=int, default=50, help="Drop documents shorter than this after line dedup")
    parser.add_argument("--num-hashes", type=int, default=128)
    parser.add_argument("--num-bands", type=int, default=16)
    parser.add_argument("--ngrams", type=int, default=5)
    parser.add_argument("--jaccard-threshold", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    input_paths = sorted(p for p in args.input_dir.glob(args.glob) if p.is_file())
    if not input_paths:
        parser.error(f"no files matching {args.glob} in {args.input_dir}")

    totals = run_corpus_dedup(
        input_paths,
        args.output_dir,
        workers=args.workers,
        line_dedup=not args.skip_line_dedup,
        minhash=not args.skip_minhash,
        min_words=args.min_words,
        num_hashes=args.num_hashes,
        num_bands=args.num_bands,
        ngrams=args.ngrams,
        jaccard_threshold=args.jaccard_threshold,
        seed=args.seed,
    )
    print(f"\nfinished in {totals['elapsed_seconds'] / 60:.1f} min")
    print(json.dumps(totals, indent=2))
    with open(args.output_dir / "dedup_stats.json", "w") as f:
        json.dump({"args": {k: str(v) for k, v in vars(args).items()}, "totals": totals}, f, indent=2)


if __name__ == "__main__":
    main()
