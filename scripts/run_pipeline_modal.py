"""Run the data pipeline on Modal, where the English-filtered WET files live (``/shared-data/english-wet-data``).

The heavy lifting is the same code used locally (``cs336_data.pipeline``, ``cs336_data.corpus_dedup``,
``cs336_data.tokenization``); this file only fans the work out over Modal containers.

0. Upload your trained quality classifier to your data volume (the shared volume is read-only):

       uv run modal volume put data-<SUNET_ID> local-shared-data/classifiers/quality_classifier.bin \\
           classifiers/quality_classifier.bin

1. Filter every WET file in parallel (one container per file via ``.map``):

       uv run modal run scripts/run_pipeline_modal.py::filter --output-dir /root/data/filtered

2. Deduplicate (single large container) and 3. tokenize:

       uv run modal run scripts/run_pipeline_modal.py::dedup --input-dir /root/data/filtered --output-dir /root/data/deduped
       uv run modal run scripts/run_pipeline_modal.py::tokenize --input-dir /root/data/deduped --output /root/data/filtered_cc.bin

4. Train: ``uv run modal run scripts/train.py --train-bin /root/data/filtered_cc.bin``
"""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

from cs336_data.modal_utils import VOLUME_MOUNTS, app, build_image, data_volume
from cs336_data.pipeline import FilterConfig, output_name, process_single_wet_file, summarize_stats

DEFAULT_WET_DIR = "/shared-data/english-wet-data"
DEFAULT_QUALITY_MODEL = "/root/data/classifiers/quality_classifier.bin"
image = build_image()


@app.function(image=image, volumes=VOLUME_MOUNTS, timeout=60 * 60)
def list_files(directory: str, glob: str) -> list[str]:
    return sorted(str(p) for p in Path(directory).glob(glob) if p.is_file() or p.is_symlink())


@app.function(image=image, volumes=VOLUME_MOUNTS, timeout=60 * 60 * 2, max_containers=128)
def filter_wet_file(job: dict) -> dict:
    result = process_single_wet_file(**job)
    data_volume.commit()
    return result


@app.function(image=image, volumes=VOLUME_MOUNTS, timeout=60 * 60 * 12, cpu=32, memory=128 * 1024)
def dedup_remote(input_dir: str, output_dir: str, glob: str, workers: int, options: dict) -> dict:
    from cs336_data.corpus_dedup import run_corpus_dedup

    input_paths = sorted(p for p in Path(input_dir).glob(glob) if p.is_file())
    totals = run_corpus_dedup(input_paths, Path(output_dir), workers=workers, **options)
    with open(Path(output_dir) / "dedup_stats.json", "w") as f:
        json.dump(totals, f, indent=2)
    data_volume.commit()
    return totals


@app.function(image=image, volumes=VOLUME_MOUNTS, timeout=60 * 60 * 12, cpu=32, memory=64 * 1024)
def tokenize_remote(input_dir: str, output: str, glob: str, workers: int) -> dict:
    from cs336_data.tokenization import tokenize_files

    paths = sorted(p for p in Path(input_dir).glob(glob) if p.is_file())
    info = tokenize_files(paths, Path(output), workers=workers)
    data_volume.commit()
    return info


@app.local_entrypoint()
def filter(
    input_dir: str = DEFAULT_WET_DIR,
    glob: str = "*.warc.wet.gz",
    output_dir: str = "/root/data/filtered",
    limit_files: int = 0,
    save_discarded_rate: float = 0.01,
    quality_model: str = DEFAULT_QUALITY_MODEL,
    quality_threshold: float = FilterConfig.quality_threshold,
    english_threshold: float = FilterConfig.english_threshold,
    nsfw_threshold: float = FilterConfig.nsfw_threshold,
    toxic_threshold: float = FilterConfig.toxic_threshold,
    apply_c4: bool = FilterConfig.apply_c4,
) -> None:
    cfg = FilterConfig(
        quality_model=quality_model,
        quality_threshold=quality_threshold,
        english_threshold=english_threshold,
        nsfw_threshold=nsfw_threshold,
        toxic_threshold=toxic_threshold,
        apply_c4=apply_c4,
    )
    wet_paths = list_files.remote(input_dir, glob)
    if limit_files:
        wet_paths = wet_paths[:limit_files]
    print(f"filtering {len(wet_paths)} WET files from {input_dir}\nconfig: {json.dumps(cfg.to_dict())}", flush=True)

    jobs = [
        {
            "input_path": wet_path,
            "output_path": f"{output_dir}/{output_name(wet_path)}",
            "cfg_dict": cfg.to_dict(),
            "discarded_path": f"{output_dir}/discarded/{output_name(wet_path)}" if save_discarded_rate > 0 else None,
            "discarded_rate": save_discarded_rate,
            "seed": i,
        }
        for i, wet_path in enumerate(wet_paths)
    ]
    started = time.perf_counter()
    totals: Counter[str] = Counter()
    per_file = []
    for i, info in enumerate(filter_wet_file.map(jobs, order_outputs=False), start=1):
        per_file.append(info)
        totals.update(info["stats"])
        if i % 100 == 0:
            print(f"{i}/{len(jobs)} files done; kept so far {totals['kept']:,d} / {totals['input']:,d}", flush=True)
    elapsed = time.perf_counter() - started
    print(f"\nfinished in {elapsed / 60:.1f} min wall-clock ({totals['seconds'] / 3600:.1f} CPU-hours)")
    print(summarize_stats(totals))
    Path("data").mkdir(exist_ok=True)
    with open("data/modal_filter_stats.json", "w") as f:
        json.dump(
            {"config": cfg.to_dict(), "elapsed_seconds": elapsed, "totals": dict(totals), "files": per_file},
            f,
            indent=2,
        )


@app.local_entrypoint()
def dedup(
    input_dir: str = "/root/data/filtered",
    output_dir: str = "/root/data/deduped",
    glob: str = "*.jsonl.gz",
    workers: int = 32,
    min_words: int = 50,
    num_hashes: int = 128,
    num_bands: int = 16,
    ngrams: int = 5,
    jaccard_threshold: float = 0.8,
) -> None:
    options = {
        "min_words": min_words,
        "num_hashes": num_hashes,
        "num_bands": num_bands,
        "ngrams": ngrams,
        "jaccard_threshold": jaccard_threshold,
    }
    totals = dedup_remote.remote(input_dir, output_dir, glob, workers, options)
    print(json.dumps(totals, indent=2))


@app.local_entrypoint()
def tokenize(
    input_dir: str = "/root/data/deduped",
    output: str = "/root/data/filtered_cc.bin",
    glob: str = "*.jsonl.gz",
    workers: int = 32,
) -> None:
    info = tokenize_remote.remote(input_dir, output, glob, workers)
    print(f"Tokenized {info['documents']:,d} documents into {info['tokens']:,d} tokens -> {info['output']}")
