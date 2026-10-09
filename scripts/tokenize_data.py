"""Tokenize filtered documents with the GPT-2 tokenizer into a ``np.uint16`` binary (problem ``tokenize_data``).

Every document is encoded with the GPT-2 tokenizer (via ``transformers``) and followed by the
``<|endoftext|>`` token; the ids are concatenated and written with ``ndarray.tofile`` so the file
can be memory-mapped by ``scripts/train.py``.

Accepted inputs: ``*.jsonl(.gz)`` files with a ``text`` field (output of the filtering / dedup
scripts) or plain ``*.txt(.gz)`` files with one document per line.

Example:
    uv run scripts/tokenize_data.py --input-dir data/deduped --output data/tokenized/filtered_cc.bin
"""

from __future__ import annotations

import argparse
import multiprocessing
from pathlib import Path

from cs336_data.tokenization import tokenize_files

TRAINING_TOKENS = 8 * 128 * 512 * 16_384  # 8 GPUs x batch 128 x context 512 x 16,384 steps (~8.6B)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, nargs="*", default=[])
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--glob", default="*.jsonl.gz")
    parser.add_argument("--output", type=Path, required=True, help="Output .bin path (np.uint16)")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--workers", type=int, default=multiprocessing.cpu_count())
    parser.add_argument("--batch-size", type=int, default=64, help="Documents per worker task")
    args = parser.parse_args()

    paths = list(args.input)
    if args.input_dir:
        paths += sorted(p for p in args.input_dir.glob(args.glob) if p.is_file())
    if not paths:
        parser.error("no input files")

    info = tokenize_files(
        paths, args.output, workers=args.workers, batch_size=args.batch_size, text_field=args.text_field
    )
    print(
        f"Tokenized {info['documents']:,d} documents from {info['files']} files into {info['tokens']:,d} tokens -> {args.output}"
    )
    print(
        f"({args.output.stat().st_size / 1e9:.2f} GB; the final training run samples ~{TRAINING_TOKENS / 1e9:.1f}B tokens, "
        f"i.e. ~{TRAINING_TOKENS / max(info['tokens'], 1):.1f} epochs over this data)"
    )


if __name__ == "__main__":
    main()
