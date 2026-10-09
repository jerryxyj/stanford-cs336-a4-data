"""Corpus-level deduplication over JSONL(.gz) document files (used by ``scripts/dedup_data.py``).

Two steps, parallelised per input file with a process pool:

1. **Exact line deduplication** across the whole corpus (lines whose hash occurs more than once
   are removed from every document; documents left too short are dropped).
2. **MinHash + LSH fuzzy document deduplication** on the result, keeping one random document
   per near-duplicate cluster. Only candidate documents are re-read for Jaccard verification.
"""

from __future__ import annotations

import concurrent.futures
import json
import os
import random
import tempfile
import time
from collections import Counter, defaultdict
from collections.abc import Iterator
from pathlib import Path

import numpy as np
from tqdm import tqdm
from xopen import xopen

from cs336_data.dedup import (
    MinHasher,
    find_duplicate_line_hashes,
    hash_line,
    lsh_candidate_pairs,
    resolve_duplicate_clusters,
    word_ngrams,
)
from cs336_data.pipeline import temporary_path


def iter_docs(path: os.PathLike | str) -> Iterator[dict]:
    with xopen(path, "rt") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def doc_line_hashes(text: str) -> tuple[list[str], np.ndarray]:
    lines = text.split("\n")
    hashes = np.fromiter((hash_line(line) for line in lines if line.strip()), dtype=np.uint64)
    return lines, hashes


# ---------------------------------------------------------------------------------------- stage A


def collect_line_hashes(path: str) -> np.ndarray:
    arrays = [doc_line_hashes(doc["text"])[1] for doc in iter_docs(path)]
    return np.concatenate(arrays) if arrays else np.empty(0, dtype=np.uint64)


# ---------------------------------------------------------------------------------------- stage B


def remove_duplicate_lines(text: str, duplicated: np.ndarray) -> tuple[str, int]:
    """Drop lines whose hash is in ``duplicated``; blank lines (paragraph breaks) are kept but collapsed."""
    lines, hashes = doc_line_hashes(text)
    if hashes.size == 0:
        return text, 0
    is_dup = np.isin(hashes, duplicated)
    kept: list[str] = []
    removed = 0
    j = 0
    for line in lines:
        if not line.strip():
            if kept and kept[-1] != "":
                kept.append("")
            continue
        if is_dup[j]:
            removed += 1
        else:
            kept.append(line)
        j += 1
    return "\n".join(kept).strip(), removed


def line_dedup_and_sign(
    path: str,
    tmp_path: str,
    duplicated_path: str | None,
    min_words: int,
    num_hashes: int,
    ngrams: int,
    seed: int,
) -> dict:
    duplicated = np.load(duplicated_path) if duplicated_path else None
    hasher = MinHasher(num_hashes, seed=seed)
    stats: Counter[str] = Counter()
    signatures: list[np.ndarray] = []
    with xopen(tmp_path, "wt") as out:
        for doc in iter_docs(path):
            stats["input"] += 1
            text = doc["text"]
            if duplicated is not None:
                text, removed = remove_duplicate_lines(text, duplicated)
                stats["lines_removed"] += removed
                if len(text.split()) < min_words:
                    stats["dropped_short_after_line_dedup"] += 1
                    continue
                doc["text"] = text
            signatures.append(hasher.signature(word_ngrams(text, ngrams)))
            out.write(json.dumps(doc, ensure_ascii=False) + "\n")
    stats["after_line_dedup"] = len(signatures)
    sig_array = np.stack(signatures) if signatures else np.empty((0, num_hashes), dtype=np.uint32)
    return {"path": path, "tmp_path": tmp_path, "stats": dict(stats), "signatures": sig_array}


# ---------------------------------------------------------------------------------------- stage C


def read_texts(tmp_path: str, local_indices: list[int]) -> dict[int, str]:
    wanted = set(local_indices)
    texts: dict[int, str] = {}
    for i, doc in enumerate(iter_docs(tmp_path)):
        if i in wanted:
            texts[i] = doc["text"]
            if len(texts) == len(wanted):
                break
    return texts


# ---------------------------------------------------------------------------------------- stage D


def write_final(tmp_path: str, output_path: str, removed_local: list[int]) -> int:
    removed = set(removed_local)
    kept = 0
    staging_path = temporary_path(output_path)
    with xopen(staging_path, "wt") as out:
        for i, doc in enumerate(iter_docs(tmp_path)):
            if i in removed:
                continue
            out.write(json.dumps(doc, ensure_ascii=False) + "\n")
            kept += 1
    os.replace(staging_path, output_path)
    return kept


def run_corpus_dedup(
    input_paths: list[Path],
    output_dir: Path,
    *,
    workers: int,
    line_dedup: bool = True,
    minhash: bool = True,
    min_words: int = 50,
    num_hashes: int = 128,
    num_bands: int = 16,
    ngrams: int = 5,
    jaccard_threshold: float = 0.8,
    seed: int = 0,
) -> dict:
    """Deduplicate ``input_paths`` into ``output_dir`` (same file names); returns statistics."""
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    totals: Counter[str] = Counter()

    with (
        tempfile.TemporaryDirectory(prefix="dedup_", dir=output_dir) as tmp_dir,
        concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor,
    ):
        # Stage A: count line hashes over the whole corpus.
        duplicated_path = None
        if line_dedup:
            hash_arrays = list(
                tqdm(
                    executor.map(collect_line_hashes, map(str, input_paths)),
                    total=len(input_paths),
                    desc="counting lines",
                )
            )
            total_lines = int(sum(a.size for a in hash_arrays))
            duplicated = find_duplicate_line_hashes(hash_arrays)
            del hash_arrays
            duplicated_path = os.path.join(tmp_dir, "duplicated_line_hashes.npy")
            np.save(duplicated_path, duplicated)
            totals["lines_total"] = total_lines
            totals["unique_lines_with_duplicates"] = int(duplicated.size)
            print(f"{total_lines:,d} lines, {duplicated.size:,d} distinct lines occur more than once", flush=True)

        # Stage B: rewrite documents without duplicated lines and compute MinHash signatures.
        futures = [
            executor.submit(
                line_dedup_and_sign,
                str(path),
                os.path.join(tmp_dir, f"{i:06d}.jsonl.gz"),
                duplicated_path,
                min_words,
                num_hashes,
                ngrams,
                seed,
            )
            for i, path in enumerate(input_paths)
        ]
        results: list[dict] = [None] * len(futures)  # type: ignore[list-item]
        index_of = {f: i for i, f in enumerate(futures)}
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="line dedup + minhash"):
            results[index_of[future]] = future.result()
        for r in results:
            totals.update(r["stats"])

        # Stage C: LSH candidates -> Jaccard verification -> clusters.
        removed_per_file: dict[int, list[int]] = defaultdict(list)
        if minhash:
            sizes = [r["signatures"].shape[0] for r in results]
            offsets = np.cumsum([0] + sizes)
            all_sigs = (
                np.concatenate([r["signatures"] for r in results])
                if sum(sizes)
                else np.empty((0, num_hashes), dtype=np.uint32)
            )
            for r in results:
                r["signatures"] = None
            pairs = lsh_candidate_pairs(all_sigs, num_bands)
            totals["candidate_pairs"] = len(pairs)
            print(f"{all_sigs.shape[0]:,d} documents, {len(pairs):,d} candidate pairs", flush=True)
            del all_sigs

            def locate(global_idx: int) -> tuple[int, int]:
                file_idx = int(np.searchsorted(offsets, global_idx, side="right") - 1)
                return file_idx, int(global_idx - offsets[file_idx])

            by_file: dict[int, list[int]] = defaultdict(list)
            for g in {doc for pair in pairs for doc in pair}:
                file_idx, local = locate(g)
                by_file[file_idx].append(local)
            text_futures = {
                file_idx: executor.submit(read_texts, results[file_idx]["tmp_path"], locals_)
                for file_idx, locals_ in by_file.items()
            }
            shingle_sets: dict[int, set[str]] = {}
            for file_idx, future in tqdm(text_futures.items(), desc="reading candidates"):
                for local, text in future.result().items():
                    shingle_sets[int(offsets[file_idx] + local)] = word_ngrams(text, ngrams)
            removed = resolve_duplicate_clusters(pairs, shingle_sets, jaccard_threshold, random.Random(seed))
            totals["removed_near_duplicates"] = len(removed)
            for g in removed:
                file_idx, local = locate(g)
                removed_per_file[file_idx].append(local)

        # Stage D: write the final files.
        write_futures = [
            executor.submit(
                write_final, r["tmp_path"], str(output_dir / Path(r["path"]).name), removed_per_file.get(i, [])
            )
            for i, r in enumerate(results)
        ]
        totals["kept"] = sum(f.result() for f in tqdm(write_futures, desc="writing"))

    totals["elapsed_seconds"] = int(time.perf_counter() - started)
    return dict(totals)
