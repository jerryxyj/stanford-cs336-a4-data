"""Exact line deduplication and MinHash + LSH fuzzy document deduplication (section 3)."""

from __future__ import annotations

import os
import random
import re
import unicodedata
from collections import defaultdict
from collections.abc import Callable, Hashable, Iterable, Iterator
from pathlib import Path

import mmh3
import numpy as np
from xopen import xopen

# --------------------------------------------------------------------------------------
# Exact line deduplication
# --------------------------------------------------------------------------------------


def hash_line(line: bytes | str) -> int:
    """64-bit MurmurHash3 of a line (without its line terminator)."""
    if isinstance(line, str):
        line = line.encode("utf-8", errors="replace")
    return mmh3.hash64(line.rstrip(b"\r\n"), signed=False)[0]


def _iter_lines(path: os.PathLike | str) -> Iterator[bytes]:
    with xopen(path, "rb") as f:
        yield from f


def line_hashes(lines: Iterable[bytes | str]) -> np.ndarray:
    return np.fromiter((hash_line(line) for line in lines), dtype=np.uint64)


def find_duplicate_line_hashes(hash_arrays: Iterable[np.ndarray]) -> np.ndarray:
    """Return the sorted array of line hashes that occur more than once overall.

    Storing fixed-size hashes (instead of the lines themselves) bounds memory by the
    number of lines rather than by their total length.
    """
    arrays = [arr for arr in hash_arrays if arr.size]
    if not arrays:
        return np.empty(0, dtype=np.uint64)
    all_hashes = np.concatenate(arrays)
    unique, counts = np.unique(all_hashes, return_counts=True)
    return unique[counts > 1]


def exact_line_deduplication(input_files: list[os.PathLike | str], output_directory: os.PathLike | str) -> None:
    """Rewrite each input file into ``output_directory`` keeping only corpus-unique lines.

    Pass 1 counts (hashed) lines across all files, pass 2 rewrites each file keeping the
    lines whose hash was seen exactly once. Output files keep their input file names.
    """
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    duplicated = find_duplicate_line_hashes(line_hashes(_iter_lines(path)) for path in input_files)

    for path in input_files:
        path = Path(path)
        lines = list(_iter_lines(path))
        if lines:
            is_dup = np.isin(line_hashes(lines), duplicated)
        else:
            is_dup = np.empty(0, dtype=bool)
        with xopen(output_directory / path.name, "wb") as f:
            for line, dup in zip(lines, is_dup):
                if not dup:
                    f.write(line)


# --------------------------------------------------------------------------------------
# MinHash + LSH
# --------------------------------------------------------------------------------------

_MERSENNE_PRIME = np.uint64((1 << 61) - 1)
_MAX_HASH = np.uint64((1 << 32) - 1)
_PUNCTUATION_RE = re.compile(r"[^\w\s]|_")


def normalize_text(text: str) -> str:
    """NFD-normalise, strip accents, lowercase, drop punctuation and collapse whitespace.

    Following RefinedWeb, normalising before shingling improves duplicate recall.
    """
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower()
    text = _PUNCTUATION_RE.sub(" ", text)
    return " ".join(text.split())


def word_ngrams(text: str, n: int) -> set[str]:
    """Set of word ``n``-grams of the normalised text.

    Documents shorter than ``n`` words get a single shingle made of all their words, so
    that identical short documents still collapse while different ones do not collide.
    """
    words = normalize_text(text).split()
    if not words:
        return set()
    if len(words) < n:
        return {" ".join(words)}
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}


def jaccard_similarity(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


class MinHasher:
    """Computes MinHash signatures with ``num_hashes`` universal hash functions.

    Each shingle is hashed once with MurmurHash3, and the ``k`` hash functions are
    ``h_i(x) = ((a_i * x + b_i) mod p) mod 2^32`` with ``p = 2^61 - 1`` and random
    ``a_i, b_i``; the ``a_i * x + b_i`` products fit in 64 bits so the arithmetic is exact.
    """

    def __init__(self, num_hashes: int, seed: int = 0, chunk_size: int = 4096):
        rng = np.random.default_rng(seed)
        self.num_hashes = num_hashes
        self.a = rng.integers(1, 1 << 32, size=num_hashes, dtype=np.uint64)[:, None]
        self.b = rng.integers(0, 1 << 32, size=num_hashes, dtype=np.uint64)[:, None]
        self.chunk_size = chunk_size

    def signature(self, shingles: Iterable[str]) -> np.ndarray:
        shingles = list(shingles)
        sig = np.full(self.num_hashes, _MAX_HASH, dtype=np.uint64)
        if not shingles:
            return sig.astype(np.uint32)
        base = np.fromiter((mmh3.hash(s, signed=False) for s in shingles), dtype=np.uint64, count=len(shingles))
        for start in range(0, base.size, self.chunk_size):
            chunk = base[None, start : start + self.chunk_size]
            hashes = ((self.a * chunk + self.b) % _MERSENNE_PRIME) & _MAX_HASH
            np.minimum(sig, hashes.min(axis=1), out=sig)
        return sig.astype(np.uint32)


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[Hashable, Hashable] = {}

    def find(self, x: Hashable) -> Hashable:
        self.parent.setdefault(x, x)
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:  # path compression
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, x: Hashable, y: Hashable) -> None:
        rx, ry = self.find(x), self.find(y)
        if rx != ry:
            self.parent[ry] = rx

    def clusters(self) -> list[list[Hashable]]:
        groups: dict[Hashable, list[Hashable]] = defaultdict(list)
        for x in list(self.parent):
            groups[self.find(x)].append(x)
        return [members for members in groups.values() if len(members) > 1]


def lsh_candidate_pairs(
    signatures: np.ndarray,
    num_bands: int,
    max_pairwise_bucket_size: int = 64,
) -> set[tuple[int, int]]:
    """Index pairs of rows of ``signatures`` (``n_docs x num_hashes``) that share an LSH band.

    Banding is vectorised with ``np.unique`` over the raw bytes of each band. Very large
    buckets are paired against their first member only, keeping the number of candidate
    pairs (and later Jaccard computations) linear in the bucket size.
    """
    n_docs, num_hashes = signatures.shape
    if num_hashes % num_bands != 0:
        raise ValueError(f"{num_hashes=} must be divisible by {num_bands=}")
    rows_per_band = num_hashes // num_bands
    pairs: set[tuple[int, int]] = set()
    if n_docs < 2:
        return pairs
    for band in range(num_bands):
        band_slice = np.ascontiguousarray(signatures[:, band * rows_per_band : (band + 1) * rows_per_band])
        keys = band_slice.view(np.dtype((np.void, band_slice.dtype.itemsize * rows_per_band))).ravel()
        _, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)
        inverse = inverse.ravel()
        if counts.max() < 2:
            continue
        order = np.argsort(inverse, kind="stable")
        boundaries = np.cumsum(counts)
        start = 0
        for end in boundaries:
            if end - start >= 2:
                members = order[start:end].tolist()
                if len(members) <= max_pairwise_bucket_size:
                    for i in range(len(members)):
                        for j in range(i + 1, len(members)):
                            pairs.add((members[i], members[j]))
                else:
                    first = members[0]
                    pairs.update((first, other) for other in members[1:])
            start = end
    return pairs


def resolve_duplicate_clusters(
    pairs: Iterable[tuple[Hashable, Hashable]],
    shingle_sets: dict[Hashable, set[str]],
    jaccard_threshold: float,
    rng: random.Random,
) -> set[Hashable]:
    """Verify candidate pairs with true Jaccard, cluster them and pick ids to drop."""
    uf = UnionFind()
    for a, b in pairs:
        if jaccard_similarity(shingle_sets[a], shingle_sets[b]) >= jaccard_threshold:
            uf.union(a, b)
    to_remove: set[Hashable] = set()
    for cluster in uf.clusters():
        cluster = sorted(cluster, key=repr)
        keep = rng.choice(cluster)
        to_remove.update(doc for doc in cluster if doc != keep)
    return to_remove


class MinHashLSH:
    """Collects MinHash signatures and resolves fuzzy duplicates with LSH + Jaccard verification.

    Usage: call :meth:`add` for every document, then :meth:`find_duplicates` with a
    callback that returns the text of the requested documents (so only candidate
    documents need to be re-read when the corpus does not fit in memory).
    """

    def __init__(
        self,
        num_hashes: int,
        num_bands: int,
        ngrams: int,
        jaccard_threshold: float,
        seed: int = 0,
        max_pairwise_bucket_size: int = 64,
    ):
        if num_hashes % num_bands != 0:
            raise ValueError(f"{num_hashes=} must be divisible by {num_bands=}")
        self.num_hashes = num_hashes
        self.num_bands = num_bands
        self.ngrams = ngrams
        self.jaccard_threshold = jaccard_threshold
        self.seed = seed
        self.max_pairwise_bucket_size = max_pairwise_bucket_size
        self.hasher = MinHasher(num_hashes, seed=seed)
        self.doc_ids: list[Hashable] = []
        self.signatures: list[np.ndarray] = []

    def shingles(self, text: str) -> set[str]:
        return word_ngrams(text, self.ngrams)

    def add(self, doc_id: Hashable, text: str) -> None:
        shingles = self.shingles(text)
        if not shingles:
            return  # empty documents are never considered duplicates
        self.doc_ids.append(doc_id)
        self.signatures.append(self.hasher.signature(shingles))

    def candidate_pairs(self) -> set[tuple[Hashable, Hashable]]:
        """Pairs of documents sharing at least one LSH bucket."""
        if len(self.signatures) < 2:
            return set()
        index_pairs = lsh_candidate_pairs(np.stack(self.signatures), self.num_bands, self.max_pairwise_bucket_size)
        return {(self.doc_ids[i], self.doc_ids[j]) for i, j in index_pairs}

    def find_duplicates(
        self,
        get_texts: Callable[[list[Hashable]], dict[Hashable, str]],
        rng: random.Random | None = None,
    ) -> set[Hashable]:
        """Return the ids of documents to drop (all but one random member of each cluster)."""
        rng = rng or random.Random(self.seed)
        pairs = self.candidate_pairs()
        if not pairs:
            return set()
        candidate_ids = sorted({doc for pair in pairs for doc in pair}, key=repr)
        texts = get_texts(candidate_ids)
        shingle_sets = {doc_id: self.shingles(texts[doc_id]) for doc_id in candidate_ids}
        return resolve_duplicate_clusters(pairs, shingle_sets, self.jaccard_threshold, rng)


def _read_text(path: os.PathLike | str) -> str:
    with xopen(path, "rb") as f:
        return f.read().decode("utf-8", errors="replace")


def minhash_deduplication(
    input_files: list[os.PathLike | str],
    num_hashes: int,
    num_bands: int,
    ngrams: int,
    jaccard_threshold: float,
    output_directory: os.PathLike | str,
    seed: int = 0,
) -> set[Path]:
    """Fuzzy-deduplicate a set of files (one document per file) with MinHash + LSH.

    Each surviving file is copied byte-for-byte to ``output_directory`` under its own
    name. Returns the set of input paths that were dropped as duplicates.
    """
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    paths = [Path(p) for p in input_files]

    lsh = MinHashLSH(num_hashes, num_bands, ngrams, jaccard_threshold, seed=seed)
    for path in paths:
        lsh.add(path, _read_text(path))

    removed = lsh.find_duplicates(lambda ids: {p: _read_text(p) for p in ids})

    for path in paths:
        if path in removed:
            continue
        with xopen(path, "rb") as src, xopen(output_directory / path.name, "wb") as dst:
            dst.write(src.read())
    return removed
