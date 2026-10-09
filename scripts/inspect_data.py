"""Inspection helpers for the written parts of the assignment.

Sub-commands:
  wet        Show WET records (look_at_cc): URL, language guess and an excerpt.
  warc       Random WARC pages run through text extraction + every classifier/filter
             (language_identification c, harmful_content d, gopher_quality_filters b).
  compare    Our extraction vs. the WET text for the same page (extract_text b).
  pii        Random WARC pages where PII masking fired, with the masked snippets (mask_pii e).
  filtered   Random documents from the filtered / deduplicated output (inspect_filtered_data a).
  discarded  Random discarded documents and the filter that removed them (inspect_filtered_data b).
  paloma     Decode the beginning of the Paloma C4-100-domains validation set.

Examples:
    uv run scripts/inspect_data.py wet --n 26
    uv run scripts/inspect_data.py warc --n 20 --seed 1
    uv run scripts/inspect_data.py compare --index 0
    uv run scripts/inspect_data.py discarded --dir data/filtered/discarded --n 5
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import re
import textwrap
from collections import Counter
from pathlib import Path

from xopen import xopen

from cs336_data import pii
from cs336_data.common import get_shared_assets_path
from cs336_data.extract import extract_text_from_html_bytes
from cs336_data.gopher import compute_gopher_stats
from cs336_data.harmful import classify_nsfw, classify_toxic_speech
from cs336_data.langid import identify_language
from cs336_data.pii import EMAIL_MASK, IP_MASK, PHONE_MASK, mask_pii
from cs336_data.warc import iter_warc_responses, iter_wet_records

SHARED = get_shared_assets_path()
DEFAULT_WARC = SHARED / "CC/example.warc.gz"
DEFAULT_WET = SHARED / "CC/example.warc.wet.gz"
DEFAULT_PALOMA = SHARED / "tokenized_paloma_c4_100_domains_validation.bin"


def excerpt(text: str, chars: int) -> str:
    text = text.strip()
    if len(text) > chars:
        text = text[:chars] + " [...]"
    return textwrap.indent(text, "    ")


def reservoir(iterable, n: int, rng: random.Random) -> list:
    sample: list = []
    for i, item in enumerate(iterable):
        if len(sample) < n:
            sample.append(item)
        else:
            j = rng.randrange(i + 1)
            if j < n:
                sample[j] = item
    return sample


def try_quality(text: str) -> str:
    try:
        from cs336_data.quality import classify_quality

        label, score = classify_quality(text)
        return f"{label} ({score:.2f})"
    except FileNotFoundError:
        return "n/a (quality model not trained)"


def describe(text: str) -> str:
    lang, lang_score = identify_language(text)
    nsfw_label, nsfw_score = classify_nsfw(text)
    toxic_label, toxic_score = classify_toxic_speech(text)
    gopher = compute_gopher_stats(text)
    failed = ", ".join(gopher.failed_rules()) or "pass"
    return (
        f"  language: {lang} ({lang_score:.2f}) | nsfw: {nsfw_label} ({nsfw_score:.2f}) | "
        f"toxic: {toxic_label} ({toxic_score:.2f}) | quality: {try_quality(text)}\n"
        f"  gopher: {failed} [words={gopher.num_words}, mean_len={gopher.mean_word_length:.1f}, "
        f"ellipsis={gopher.ellipsis_line_fraction:.2f}, alpha={gopher.alpha_word_fraction:.2f}]"
    )


def cmd_wet(args: argparse.Namespace) -> None:
    records = itertools.islice(iter_wet_records(args.path), args.offset, args.offset + args.n)
    for i, rec in enumerate(records, start=args.offset):
        lang, score = identify_language(rec.text)
        print(f"\n[{i}] {rec.url}\n  language: {lang} ({score:.2f}), {len(rec.text.split())} words")
        print(excerpt(rec.text, args.chars))


def cmd_warc(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    responses = itertools.islice(iter_warc_responses(args.path), args.scan)
    sample = reservoir(responses, args.n, rng)
    for i, resp in enumerate(sample):
        text = extract_text_from_html_bytes(resp.payload)
        print(f"\n[{i}] {resp.url}")
        print(describe(text))
        print(excerpt(text, args.chars))


def cmd_compare(args: argparse.Namespace) -> None:
    resp = next(itertools.islice(iter_warc_responses(args.warc), args.index, args.index + 1))
    ours = extract_plain = extract_text_from_html_bytes(resp.payload)
    wet_text = next((rec.text for rec in iter_wet_records(args.wet) if rec.url == resp.url), None)
    print(f"URL: {resp.url}\n\n===== our extraction ({len(ours.split())} words) =====")
    print(excerpt(extract_plain, args.chars))
    print(f"\n===== WET text ({len(wet_text.split()) if wet_text else 0} words) =====")
    print(excerpt(wet_text, args.chars) if wet_text is not None else "    (URL not found in WET file)")


_MASK_RE = re.compile("|".join(map(re.escape, (EMAIL_MASK, PHONE_MASK, IP_MASK))))


def cmd_pii(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    hits = []
    for resp in itertools.islice(iter_warc_responses(args.path), args.scan):
        text = extract_text_from_html_bytes(resp.payload)
        masked, counts = mask_pii(text)
        if sum(counts.values()):
            hits.append((resp.url, text, masked, counts))
    sample = rng.sample(hits, min(args.n, len(hits)))
    print(f"{len(hits)} of the first {args.scan} pages contained a masked span")
    for i, (url, original, masked, counts) in enumerate(sample):
        print(f"\n[{i}] {url}  {counts}")
        for m in itertools.islice(_MASK_RE.finditer(masked), 5):
            start, end = max(0, m.start() - args.context), min(len(masked), m.end() + args.context)
            print("    ..." + masked[start:end].replace("\n", " ") + "...")
        # show what was actually replaced
        originals = []
        for pattern_name, pattern in (("email", pii.EMAIL_RE), ("phone", pii.PHONE_RE), ("ip", pii.IPV4_RE)):
            originals += [f"{pattern_name}={m.group(0)!r}" for m in pattern.finditer(original)]
        print("    matched: " + ", ".join(originals[:8]))


def _sample_jsonl(directory: Path, pattern: str, n: int, rng: random.Random, predicate=lambda d: True) -> list[dict]:
    def docs():
        for path in sorted(directory.glob(pattern)):
            with xopen(path, "rt") as f:
                for line in f:
                    doc = json.loads(line)
                    if predicate(doc):
                        yield doc

    return reservoir(docs(), n, rng)


def cmd_filtered(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    for i, doc in enumerate(_sample_jsonl(args.dir, args.glob, args.n, rng)):
        meta = {k: v for k, v in doc.items() if k not in ("text", "url")}
        print(f"\n[{i}] {doc.get('url')}\n  {meta}")
        print(excerpt(doc["text"], args.chars))


def cmd_discarded(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    predicate = (lambda d: d.get("rejected_by") == args.reason) if args.reason else (lambda d: True)
    docs = _sample_jsonl(args.dir, args.glob, args.n, rng, predicate)
    reasons = Counter()
    for path in sorted(args.dir.glob(args.glob)):
        with xopen(path, "rt") as f:
            for line in f:
                reasons[json.loads(line).get("rejected_by")] += 1
    print(f"saved discarded documents by stage: {dict(reasons.most_common())}")
    for i, doc in enumerate(docs):
        meta = {k: v for k, v in doc.items() if k not in ("text", "url")}
        print(f"\n[{i}] {doc.get('url')}\n  {meta}")
        print(excerpt(doc["text"], args.chars))


def cmd_paloma(args: argparse.Namespace) -> None:
    import numpy as np
    from transformers import AutoTokenizer

    data = np.fromfile(args.path, dtype=np.uint16)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    print(f"{data.size:,d} tokens, {(data == tokenizer.eos_token_id).sum():,d} documents")
    print(tokenizer.decode(data[args.offset : args.offset + args.tokens]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("wet")
    p.add_argument("--path", type=Path, default=DEFAULT_WET)
    p.add_argument("--n", type=int, default=26)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--chars", type=int, default=600)
    p.set_defaults(func=cmd_wet)

    p = sub.add_parser("warc")
    p.add_argument("--path", type=Path, default=DEFAULT_WARC)
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--scan", type=int, default=5000, help="Sample from the first N responses")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--chars", type=int, default=600)
    p.set_defaults(func=cmd_warc)

    p = sub.add_parser("compare")
    p.add_argument("--warc", type=Path, default=DEFAULT_WARC)
    p.add_argument("--wet", type=Path, default=DEFAULT_WET)
    p.add_argument("--index", type=int, default=0, help="Index of the WARC response record")
    p.add_argument("--chars", type=int, default=3000)
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser("pii")
    p.add_argument("--path", type=Path, default=DEFAULT_WARC)
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--scan", type=int, default=3000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--context", type=int, default=60)
    p.set_defaults(func=cmd_pii)

    p = sub.add_parser("filtered")
    p.add_argument("--dir", type=Path, required=True)
    p.add_argument("--glob", default="*.jsonl.gz")
    p.add_argument("--n", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--chars", type=int, default=1200)
    p.set_defaults(func=cmd_filtered)

    p = sub.add_parser("discarded")
    p.add_argument("--dir", type=Path, required=True)
    p.add_argument("--glob", default="*.jsonl.gz")
    p.add_argument("--n", type=int, default=5)
    p.add_argument("--reason", default=None, help="Only show documents rejected by this stage")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--chars", type=int, default=1200)
    p.set_defaults(func=cmd_discarded)

    p = sub.add_parser("paloma")
    p.add_argument("--path", type=Path, default=DEFAULT_PALOMA)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--tokens", type=int, default=2000)
    p.set_defaults(func=cmd_paloma)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
