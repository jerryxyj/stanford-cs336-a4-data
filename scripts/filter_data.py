"""Filter Common Crawl WET files into language-modeling data (problem ``filter_data``).

Each WET file is processed by a separate worker process. For every ``conversion`` record
the text goes through :func:`cs336_data.pipeline.filter_document`; kept documents are
written to ``<output-dir>/<wet name>.jsonl.gz`` (one JSON object per line with ``url``,
``text`` and the filter scores). A random sample of discarded documents can be saved to
``<output-dir>/discarded/`` with the rejecting stage for inspection.

At the end the script prints how many documents each filter removed and writes
``<output-dir>/stats.json``.

Examples:
    # local smoke test on the example WET file
    uv run scripts/filter_data.py --input local-shared-data/CC/example.warc.wet.gz --output-dir data/filtered

    # the English-filtered WET files
    uv run scripts/filter_data.py --input-dir /shared-data/english-wet-data --output-dir /root/data/filtered
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import sys
import time
from collections import Counter
from dataclasses import fields
from pathlib import Path

from tqdm import tqdm

from cs336_data.pipeline import FilterConfig, output_name, process_single_wet_file, summarize_stats


def add_config_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("filter thresholds (see cs336_data.pipeline.FilterConfig)")
    for f in fields(FilterConfig):
        flag = "--" + f.name.replace("_", "-")
        if f.type == "bool" or f.type is bool:
            group.add_argument(
                flag, type=lambda s: s.lower() in ("1", "true", "yes", "y"), default=f.default, metavar="BOOL"
            )
        else:
            group.add_argument(flag, type=type(f.default), default=f.default)


def config_from_args(args: argparse.Namespace) -> FilterConfig:
    return FilterConfig(**{f.name: getattr(args, f.name) for f in fields(FilterConfig)})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, nargs="*", default=[], help="WET file(s) to process")
    parser.add_argument("--input-dir", type=Path, help="Directory of WET files")
    parser.add_argument("--glob", default="*.warc.wet.gz")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--workers",
        type=int,
        default=len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count(),
    )
    parser.add_argument("--limit-files", type=int, default=None, help="Only process the first N WET files")
    parser.add_argument(
        "--max-records-per-file", type=int, default=None, help="Debugging: only read N records per file"
    )
    parser.add_argument(
        "--save-discarded-rate", type=float, default=0.0, help="Fraction of discarded docs to save for inspection"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true", help="Re-process files whose output already exists")
    add_config_args(parser)
    args = parser.parse_args()

    wet_paths = list(args.input)
    if args.input_dir:
        wet_paths += sorted(args.input_dir.glob(args.glob))
    wet_paths = [p for p in dict.fromkeys(Path(p) for p in wet_paths)]
    if args.limit_files:
        wet_paths = wet_paths[: args.limit_files]
    if not wet_paths:
        parser.error("no input WET files found")

    cfg = config_from_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    discarded_dir = args.output_dir / "discarded"
    if args.save_discarded_rate > 0:
        discarded_dir.mkdir(exist_ok=True)
    print(
        f"filtering {len(wet_paths)} WET files with {args.workers} workers\nconfig: {json.dumps(cfg.to_dict())}",
        flush=True,
    )

    started = time.perf_counter()
    totals: Counter[str] = Counter()
    per_file: list[dict] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = []
        for i, wet_path in enumerate(wet_paths):
            output_path = args.output_dir / output_name(wet_path)
            if output_path.exists() and not args.overwrite:
                print(f"skipping {wet_path.name}: {output_path} exists", file=sys.stderr)
                continue
            discarded_path = str(discarded_dir / output_name(wet_path)) if args.save_discarded_rate > 0 else None
            futures.append(
                executor.submit(
                    process_single_wet_file,
                    str(wet_path),
                    str(output_path),
                    cfg.to_dict(),
                    discarded_path,
                    args.save_discarded_rate,
                    args.seed + i,
                    args.max_records_per_file,
                )
            )
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="WET files"):
            info = future.result()
            per_file.append(info)
            totals.update(info["stats"])

    elapsed = time.perf_counter() - started
    print(
        f"\nfinished in {elapsed / 60:.1f} min ({elapsed / max(len(per_file), 1):.1f} s per file, {args.workers} workers)"
    )
    print(summarize_stats(totals))
    with open(args.output_dir / "stats.json", "w") as f:
        json.dump(
            {
                "config": cfg.to_dict(),
                "elapsed_seconds": elapsed,
                "workers": args.workers,
                "totals": dict(totals),
                "files": per_file,
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
