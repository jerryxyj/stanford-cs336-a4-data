"""GPT-2 tokenization of filtered documents into a ``np.uint16`` binary (used by ``scripts/tokenize_data.py``)."""

from __future__ import annotations

import json
import multiprocessing
import os
from collections.abc import Iterator
from pathlib import Path

import numpy as np
from tqdm import tqdm
from xopen import xopen

GPT2_EOS_TOKEN_ID = 50256

_tokenizer = None


def get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        import transformers

        transformers.logging.set_verbosity_error()  # silence the >1024-token sequence warnings
        _tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")
    return _tokenizer


def tokenize_batch(texts: list[str]) -> np.ndarray:
    """Encode documents with GPT-2 BPE, appending ``<|endoftext|>`` after each one."""
    tokenizer = get_tokenizer()
    eos = tokenizer.eos_token_id
    encoded = tokenizer(texts, add_special_tokens=False)["input_ids"]
    ids: list[int] = []
    for doc_ids in encoded:
        ids.extend(doc_ids)
        ids.append(eos)
    return np.asarray(ids, dtype=np.uint16)


def iter_texts(paths: list[Path], text_field: str = "text") -> Iterator[str]:
    """Documents from ``*.jsonl(.gz)`` files (``text_field``) or ``*.txt(.gz)`` files (one per line)."""
    for path in paths:
        is_jsonl = ".jsonl" in path.name
        with xopen(path, "rt") as f:
            for line in f:
                if not line.strip():
                    continue
                text = json.loads(line).get(text_field, "") if is_jsonl else line.rstrip("\n")
                if text:
                    yield text


def batched(iterator: Iterator[str], batch_size: int) -> Iterator[list[str]]:
    batch: list[str] = []
    for item in iterator:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def tokenize_files(
    paths: list[Path],
    output: Path,
    *,
    workers: int | None = None,
    batch_size: int = 64,
    text_field: str = "text",
) -> dict:
    """Tokenize all documents in ``paths`` into ``output`` (written with ``ndarray.tofile``)."""
    output.parent.mkdir(parents=True, exist_ok=True)
    total_tokens = 0
    total_docs = 0
    tmp_output = output.with_suffix(output.suffix + ".tmp")
    with multiprocessing.Pool(workers or multiprocessing.cpu_count()) as pool, open(tmp_output, "wb") as out:
        batches = batched(iter_texts(paths, text_field), batch_size)
        for ids in tqdm(pool.imap(tokenize_batch, batches, chunksize=1), desc="tokenizing batches", unit="batch"):
            ids.tofile(out)
            total_tokens += int(ids.size)
            total_docs += int((ids == GPT2_EOS_TOKEN_ID).sum())
    os.replace(tmp_output, output)
    return {"documents": total_docs, "tokens": total_tokens, "files": len(paths), "output": str(output)}
