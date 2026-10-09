"""Per-document filtering logic shared by the data pipeline scripts (problem ``filter_data``).

``filter_document`` applies the primitives implemented in this package in a fixed order,
cheapest first, and reports which stage (if any) rejected the document so that the
driver script can account for how much each filter removes:

    language -> gopher -> c4 -> harmful (nsfw / toxic) -> quality classifier -> PII masking
"""

from __future__ import annotations

import json
import os
import random
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

from xopen import xopen

from cs336_data.c4 import c4_clean
from cs336_data.gopher import compute_gopher_stats
from cs336_data.harmful import nsfw_probability, toxic_probability
from cs336_data.langid import identify_language
from cs336_data.pii import mask_pii
from cs336_data.quality import quality_score
from cs336_data.warc import iter_wet_records

STAGES = ("language", "gopher", "c4", "nsfw", "toxic", "quality")


@dataclass
class FilterConfig:
    english_threshold: float = 0.85
    apply_gopher: bool = True
    apply_c4: bool = True
    c4_min_words_per_line: int = 5
    c4_min_sentences: int = 3
    c4_drop_code: bool = True
    min_words_after_cleaning: int = 50
    nsfw_threshold: float = 0.5
    toxic_threshold: float = 0.5
    quality_threshold: float = 0.05
    apply_quality_classifier: bool = True
    quality_model: str = ""  # path override; empty = default wiki-vs-cc model (or $CS336_QUALITY_MODEL)
    mask_pii: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FilterResult:
    text: str | None
    rejected_by: str | None = None
    language: str = ""
    language_score: float = 0.0
    nsfw_score: float | None = None
    toxic_score: float | None = None
    quality_score: float | None = None
    pii_counts: dict[str, int] = field(default_factory=dict)

    @property
    def kept(self) -> bool:
        return self.text is not None

    def metadata(self) -> dict:
        return {
            "language": self.language,
            "language_score": round(self.language_score, 4),
            "nsfw_score": None if self.nsfw_score is None else round(self.nsfw_score, 4),
            "toxic_score": None if self.toxic_score is None else round(self.toxic_score, 4),
            "quality_score": None if self.quality_score is None else round(self.quality_score, 4),
            "pii_counts": self.pii_counts,
        }


def filter_document(text: str, cfg: FilterConfig) -> FilterResult:
    """Run the full filter cascade over one document."""
    result = FilterResult(text=None)

    language, language_score = identify_language(text)
    result.language, result.language_score = language, language_score
    if language != "en" or language_score < cfg.english_threshold:
        result.rejected_by = "language"
        return result

    if cfg.apply_gopher and not compute_gopher_stats(text).passes:
        result.rejected_by = "gopher"
        return result

    if cfg.apply_c4:
        c4 = c4_clean(
            text,
            min_words_per_line=cfg.c4_min_words_per_line,
            min_sentences=cfg.c4_min_sentences,
            drop_code=cfg.c4_drop_code,
        )
        if c4.text is None:
            result.rejected_by = c4.rejected_by
            return result
        text = c4.text
        if compute_gopher_stats(text).num_words < cfg.min_words_after_cleaning:
            result.rejected_by = "c4_too_short"
            return result

    result.nsfw_score = nsfw_probability(text)
    if result.nsfw_score >= cfg.nsfw_threshold:
        result.rejected_by = "nsfw"
        return result
    result.toxic_score = toxic_probability(text)
    if result.toxic_score >= cfg.toxic_threshold:
        result.rejected_by = "toxic"
        return result

    if cfg.apply_quality_classifier:
        result.quality_score = quality_score(text, cfg.quality_model or None)
        if result.quality_score < cfg.quality_threshold:
            result.rejected_by = "quality"
            return result

    if cfg.mask_pii:
        text, result.pii_counts = mask_pii(text)

    result.text = text
    return result


def summarize_stats(stats: Counter) -> str:
    """Render the per-stage removal table reported by the filtering script."""
    total = stats.get("input", 0)
    kept = stats.get("kept", 0)
    discarded = total - kept
    rows = []
    for key, count in sorted(stats.items(), key=lambda kv: -kv[1]):
        if not key.startswith("rejected:"):
            continue
        stage = key.removeprefix("rejected:")
        pct_input = 100 * count / total if total else 0.0
        pct_discarded = 100 * count / discarded if discarded else 0.0
        rows.append(f"  {stage:<24} {count:>12,d} {pct_input:>10.2f}% {pct_discarded:>14.2f}%")
    header = f"  {'stage':<24} {'removed':>12} {'% of input':>11} {'% of discarded':>15}"
    lines = [
        f"documents in: {total:,d}   kept: {kept:,d} ({100 * kept / total if total else 0:.2f}%)   discarded: {discarded:,d}",
        header,
        *rows,
    ]
    for key in ("pii:emails", "pii:phone_numbers", "pii:ips"):
        if key in stats:
            lines.append(f"  masked {key.removeprefix('pii:'):<17} {stats[key]:>12,d}")
    return "\n".join(lines)


# ------------------------------------------------------------------------------------------
# Per-file processing (used by scripts/filter_data.py locally and by the Modal runner)
# ------------------------------------------------------------------------------------------


def output_name(wet_path: os.PathLike | str) -> str:
    """``CC-MAIN-...-00000.warc.wet.gz`` -> ``CC-MAIN-...-00000.jsonl.gz``."""
    name = Path(wet_path).name
    for suffix in (".warc.wet.gz", ".warc.wet", ".wet.gz", ".wet", ".gz"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return name + ".jsonl.gz"


def temporary_path(path: os.PathLike | str) -> str:
    """``x.jsonl.gz`` -> ``x.tmp.jsonl.gz`` so that xopen still picks the right compression."""
    p = Path(path)
    suffixes = "".join(p.suffixes)
    stem = p.name[: -len(suffixes)] if suffixes else p.name
    return str(p.with_name(f"{stem}.tmp{suffixes}"))


def process_single_wet_file(
    input_path: str,
    output_path: str,
    cfg_dict: dict,
    discarded_path: str | None = None,
    discarded_rate: float = 0.0,
    seed: int = 0,
    max_records: int | None = None,
) -> dict:
    """Filter one WET file into ``output_path`` (JSONL.gz) and return per-stage statistics.

    Kept documents are written as ``{"url", "text", ...scores}``. A ``discarded_rate`` fraction
    of rejected documents is written to ``discarded_path`` together with the rejecting stage.
    """
    cfg = FilterConfig(**cfg_dict)
    rng = random.Random(seed)
    stats: Counter[str] = Counter()
    started = time.perf_counter()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    tmp_output = temporary_path(output_path)
    discarded_file = None
    if discarded_path:
        Path(discarded_path).parent.mkdir(parents=True, exist_ok=True)
        discarded_file = xopen(discarded_path, "wt")
    try:
        with xopen(tmp_output, "wt") as out:
            for record in iter_wet_records(input_path):
                if max_records is not None and stats["input"] >= max_records:
                    break
                stats["input"] += 1
                result = filter_document(record.text, cfg)
                if result.kept:
                    stats["kept"] += 1
                    for key, count in result.pii_counts.items():
                        stats[f"pii:{key}"] += count
                    out.write(
                        json.dumps({"url": record.url, "text": result.text, **result.metadata()}, ensure_ascii=False)
                        + "\n"
                    )
                else:
                    stats[f"rejected:{result.rejected_by}"] += 1
                    if discarded_file is not None and rng.random() < discarded_rate:
                        row = {
                            "url": record.url,
                            "rejected_by": result.rejected_by,
                            "text": record.text,
                            **result.metadata(),
                        }
                        discarded_file.write(json.dumps(row, ensure_ascii=False) + "\n")
    finally:
        if discarded_file is not None:
            discarded_file.close()
    os.replace(tmp_output, output_path)
    stats["seconds"] = int(time.perf_counter() - started)
    return {"input": input_path, "output": output_path, "stats": dict(stats)}
