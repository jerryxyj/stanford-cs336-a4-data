"""Build training data for the "wiki vs. cc" quality classifier (problem ``quality_classifier``).

Stages (each is skipped when its output already exists, so the script can be re-run):

1. ``positive_urls.txt``  Subsample URLs referenced from English Wikipedia. If the staff file
                           ``<shared>/wiki/enwiki-20260501-extracted_urls.txt.gz`` exists it is sampled;
                           otherwise external links of random articles are pulled from the Wikipedia API.
2. ``positives.warc.gz``  Scrape the sampled URLs into a WARC file (``wget --warc-file`` equivalent).
3. ``positives.jsonl.gz`` Extract text from the WARC and keep documents that are English, pass the
                           Gopher rules and are not flagged as NSFW/toxic.
4. ``negatives.jsonl.gz`` Random English documents from Common Crawl WET files.
5. ``train.txt`` / ``valid.txt``  Balanced, shuffled fastText files (``__label__wiki`` / ``__label__cc``).

Both classes are written twice when ``--augment-c4-cleaned`` is on: the raw extracted text and
its C4-cleaned version (``cs336_data.c4.c4_clean``), so the classifier learns to judge content
rather than the presence of navigation boilerplate (the pipeline scores C4-cleaned text).

``--positive-source paloma`` instead uses the documents of the Paloma C4-100-domains validation
set as positives (label ``c4``). The handout allows using the validation data to *build* filters;
these documents only ever train the fastText classifier and are never written to the LM data.

Examples:
    uv run scripts/build_quality_classifier_data.py --num-urls 6000 --num-negatives 4000
    uv run scripts/build_quality_classifier_data.py --positive-source paloma \
        --output-dir local-shared-data/quality_classifier_c4 --num-negatives 14000
"""

from __future__ import annotations

import argparse
import concurrent.futures
import gzip
import http.client
import json
import random
import ssl
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from io import BytesIO
from pathlib import Path

from tldextract import TLDExtract
from tqdm import tqdm
from warcio.statusandheaders import StatusAndHeaders
from warcio.warcwriter import WARCWriter
from xopen import xopen

from cs336_data.c4 import c4_clean
from cs336_data.common import get_shared_assets_path
from cs336_data.extract import extract_text_from_html_bytes
from cs336_data.gopher import gopher_quality_filter
from cs336_data.harmful import nsfw_probability, toxic_probability
from cs336_data.langid import english_probability
from cs336_data.quality import HIGH_QUALITY_LABEL, LOW_QUALITY_LABEL, format_training_example
from cs336_data.warc import iter_warc_responses, iter_wet_records

USER_AGENT = "cs336-a4-quality-classifier/0.1 (self-study; +https://github.com/stanford-cs336/assignment4-data)"
WIKI_API = "https://en.wikipedia.org/w/api.php"

# Link targets that are not useful "web page" positives: archives, bibliographic authority
# records, social media, file hosts, etc.
BLOCKED_DOMAINS = {
    "archive.org",
    "archive.today",
    "archive.ph",
    "archive.is",
    "webcitation.org",
    "wikipedia.org",
    "wikimedia.org",
    "wikidata.org",
    "wiktionary.org",
    "wikisource.org",
    "wikiquote.org",
    "doi.org",
    "jstor.org",
    "worldcat.org",
    "viaf.org",
    "isni.org",
    "d-nb.info",
    "loc.gov",
    "bnf.fr",
    "nla.gov.au",
    "ndl.go.jp",
    "nkp.cz",
    "nii.ac.jp",
    "bibliotheken.nl",
    "deutsche-digitale-bibliothek.de",
    "idref.fr",
    "getty.edu",
    "sudoc.fr",
    "openlibrary.org",
    "google.com",
    "youtube.com",
    "youtu.be",
    "twitter.com",
    "x.com",
    "facebook.com",
    "instagram.com",
    "tiktok.com",
    "reddit.com",
    "linkedin.com",
    "amazon.com",
    "goodreads.com",
    "imdb.com",
    "discogs.com",
    "musicbrainz.org",
    "allmusic.com",
    "spotify.com",
    "apple.com",
    "flickr.com",
    "pinterest.com",
    "vimeo.com",
    "soundcloud.com",
    "handle.net",
    "semanticscholar.org",
    "orcid.org",
    "ncbi.nlm.nih.gov",
    "arxiv.org",
    "zenodo.org",
    "researchgate.net",
    "academia.edu",
    "scholar.google.com",
    "books.google.com",
}
BLOCKED_EXTENSIONS = (".pdf", ".jpg", ".jpeg", ".png", ".gif", ".zip", ".mp3", ".mp4", ".doc", ".docx", ".xls", ".ppt")

tld_extract = TLDExtract(suffix_list_urls=())  # offline suffix list bundled with the package


def registered_domain(url: str) -> str:
    parts = tld_extract(url)
    return parts.top_domain_under_public_suffix or parts.domain


def is_candidate_url(url: str) -> bool:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        return False
    if parsed.path.lower().endswith(BLOCKED_EXTENSIONS):
        return False
    domain = registered_domain(url)
    return bool(domain) and domain not in BLOCKED_DOMAINS


def ssl_context() -> ssl.SSLContext:
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:  # pragma: no cover
        return ssl.create_default_context()


def _ascii(value: str) -> str:
    """WARC/HTTP header lines must be ASCII; escape anything else instead of crashing."""
    return value.encode("ascii", "backslashreplace").decode("ascii")


def http_get(
    url: str, timeout: float, max_bytes: int, context: ssl.SSLContext
) -> tuple[int, str, list[tuple[str, str]], bytes]:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "text/html,*/*;q=0.5"})
    with urllib.request.urlopen(request, timeout=timeout, context=context) as response:
        headers = [(_ascii(k), _ascii(v)) for k, v in response.getheaders()]
        body = response.read(max_bytes + 1)
        if len(body) > max_bytes:
            raise ValueError("response too large")
        return response.status, _ascii(response.reason or ""), headers, body


# ------------------------------------------------------------------------------------------
# Stage 1: positive URLs
# ------------------------------------------------------------------------------------------


def sample_urls_from_file(path: Path, num_urls: int, rng: random.Random, max_per_domain: int) -> list[str]:
    """Reservoir-sample candidate URLs from the (large) extracted-URLs file."""
    print(f"[urls] sampling from {path}", flush=True)
    reservoir: list[str] = []
    seen = 0
    with gzip.open(path, "rt", errors="ignore") as f:
        for line in f:
            url = line.strip()
            if not is_candidate_url(url):
                continue
            seen += 1
            if len(reservoir) < num_urls * 4:
                reservoir.append(url)
            else:
                j = rng.randrange(seen)
                if j < len(reservoir):
                    reservoir[j] = url
    rng.shuffle(reservoir)
    return cap_per_domain(reservoir, num_urls, max_per_domain)


def sample_urls_from_wikipedia_api(
    num_urls: int, rng: random.Random, max_per_domain: int, context: ssl.SSLContext
) -> list[str]:
    """Collect external links from random English Wikipedia articles."""
    print("[urls] sampling external links of random Wikipedia articles via the API", flush=True)
    urls: list[str] = []
    params = {
        "action": "query",
        "generator": "random",
        "grnnamespace": "0",
        "grnlimit": "50",
        "prop": "extlinks",
        "ellimit": "max",
        "format": "json",
        "formatversion": "2",
    }
    target_pool = num_urls * 3  # oversample so that the per-domain cap still leaves enough URLs
    with tqdm(total=target_pool, desc="wiki external links") as pbar:
        while len(urls) < target_pool:
            query = WIKI_API + "?" + urllib.parse.urlencode(params)
            request = urllib.request.Request(query, headers={"User-Agent": USER_AGENT})
            try:
                with urllib.request.urlopen(request, timeout=30, context=context) as response:
                    data = json.load(response)
            except (urllib.error.URLError, http.client.HTTPException, TimeoutError) as e:
                print(f"[urls] API error {e!r}; retrying", file=sys.stderr, flush=True)
                time.sleep(2)
                continue
            before = len(urls)
            for page in data.get("query", {}).get("pages", []):
                for link in page.get("extlinks", []):
                    url = link.get("url", "")
                    if is_candidate_url(url):
                        urls.append(url)
            pbar.update(len(urls) - before)
            # Do not follow extlinks continuation: a fresh random batch gives more diverse pages.
            params.pop("elcontinue", None)
            time.sleep(0.2)
    urls = list(dict.fromkeys(urls))
    rng.shuffle(urls)
    return cap_per_domain(urls, num_urls, max_per_domain)


def cap_per_domain(urls: list[str], num_urls: int, max_per_domain: int) -> list[str]:
    per_domain: Counter[str] = Counter()
    kept: list[str] = []
    for url in urls:
        domain = registered_domain(url)
        if per_domain[domain] >= max_per_domain:
            continue
        per_domain[domain] += 1
        kept.append(url)
        if len(kept) >= num_urls:
            break
    return kept


# ------------------------------------------------------------------------------------------
# Stage 2: scrape to WARC
# ------------------------------------------------------------------------------------------


def scrape_to_warc(urls: list[str], output_path: Path, *, workers: int, timeout: float, max_bytes: int) -> None:
    context = ssl_context()
    ok = failed = 0

    def fetch(url: str):
        try:
            return url, http_get(url, timeout, max_bytes, context), None
        except Exception as e:  # noqa: BLE001 - any network failure just drops the URL
            return url, None, repr(e)

    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with open(tmp_path, "wb") as f, concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        writer = WARCWriter(f, gzip=True)
        futures = [executor.submit(fetch, url) for url in urls]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="scraping"):
            url, result, _error = future.result()
            if result is None:
                failed += 1
                continue
            status, reason, headers, body = result
            http_headers = StatusAndHeaders(f"{status} {reason}", headers, protocol="HTTP/1.1")
            warc_url = urllib.parse.quote(url, safe=":/?#[]@!$&'()*+,;=%")  # WARC headers must be ASCII
            try:
                record = writer.create_warc_record(
                    warc_url, "response", payload=BytesIO(body), http_headers=http_headers
                )
                writer.write_record(record)
            except Exception as e:  # noqa: BLE001
                failed += 1
                print(f"[scrape] could not write record for {url}: {e!r}", file=sys.stderr, flush=True)
                continue
            ok += 1
    tmp_path.rename(output_path)
    print(f"[scrape] wrote {ok} responses to {output_path} ({failed} URLs failed)", flush=True)


# ------------------------------------------------------------------------------------------
# Stage 3 / 4: documents
# ------------------------------------------------------------------------------------------


def write_jsonl(path: Path, docs: list[dict]) -> None:
    with xopen(path, "wt") as f:
        for doc in docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    with xopen(path, "rt") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_positives(
    warc_path: Path, *, english_threshold: float, harmful_threshold: float
) -> tuple[list[dict], Counter]:
    docs: list[dict] = []
    stats: Counter[str] = Counter()
    for response in tqdm(iter_warc_responses(warc_path), desc="extracting positives"):
        stats["responses"] += 1
        text = extract_text_from_html_bytes(response.payload)
        if english_probability(text) < english_threshold:
            stats["dropped_language"] += 1
            continue
        if not gopher_quality_filter(text):
            stats["dropped_gopher"] += 1
            continue
        if nsfw_probability(text) >= harmful_threshold or toxic_probability(text) >= harmful_threshold:
            stats["dropped_harmful"] += 1
            continue
        stats["kept"] += 1
        docs.append({"url": response.url, "text": text})
    return docs, stats


def build_negatives(
    wet_paths: list[Path], num_negatives: int, rng: random.Random, *, english_threshold: float
) -> list[dict]:
    """Reservoir-sample English WET documents as low-quality ("cc") examples."""
    reservoir: list[dict] = []
    seen = 0
    for wet_path in wet_paths:
        for record in tqdm(iter_wet_records(wet_path), desc=f"negatives from {wet_path.name}"):
            if english_probability(record.text) < english_threshold:
                continue
            seen += 1
            doc = {"url": record.url, "text": record.text}
            if len(reservoir) < num_negatives:
                reservoir.append(doc)
            else:
                j = rng.randrange(seen)
                if j < num_negatives:
                    reservoir[j] = doc
    return reservoir


def build_paloma_positives(paloma_bin: Path) -> list[dict]:
    """Decode the GPT-2-tokenized Paloma validation documents (split on <|endoftext|>)."""
    import numpy as np
    import transformers

    transformers.logging.set_verbosity_error()
    tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")
    data = np.fromfile(paloma_bin, dtype=np.uint16)
    ends = np.flatnonzero(data == tokenizer.eos_token_id)
    starts = np.concatenate([[0], ends[:-1] + 1])
    docs = []
    for i, (start, end) in enumerate(tqdm(zip(starts, ends), total=ends.size, desc="decoding paloma")):
        text = tokenizer.decode(data[start:end]).strip()
        if text:
            docs.append({"url": f"paloma:{i}", "text": text})
    return docs


# ------------------------------------------------------------------------------------------
# Stage 5: fastText files
# ------------------------------------------------------------------------------------------


def examples_for(label: str, docs: list[dict], augment: bool) -> list[str]:
    examples = []
    for doc in docs:
        text = doc["text"]
        examples.append(format_training_example(label, text))
        if augment:
            cleaned = c4_clean(text).text
            if cleaned and cleaned != text.strip():
                examples.append(format_training_example(label, cleaned))
    return examples


def write_fasttext_files(
    positives: list[dict],
    negatives: list[dict],
    output_dir: Path,
    rng: random.Random,
    valid_fraction: float,
    *,
    positive_label: str = HIGH_QUALITY_LABEL,
    augment: bool = True,
) -> None:
    rng.shuffle(positives)
    rng.shuffle(negatives)
    pos_examples = examples_for(positive_label, positives, augment)
    neg_examples = examples_for(LOW_QUALITY_LABEL, negatives, augment)
    rng.shuffle(pos_examples)
    rng.shuffle(neg_examples)
    n = min(len(pos_examples), len(neg_examples))
    examples = pos_examples[:n] + neg_examples[:n]
    rng.shuffle(examples)
    n_valid = int(len(examples) * valid_fraction)
    with open(output_dir / "valid.txt", "w") as f:
        f.write("\n".join(examples[:n_valid]) + "\n")
    with open(output_dir / "train.txt", "w") as f:
        f.write("\n".join(examples[n_valid:]) + "\n")
    print(f"[fasttext] {len(examples) - n_valid} train / {n_valid} valid examples ({n} per class)", flush=True)


def main() -> None:
    shared = get_shared_assets_path()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=shared / "quality_classifier")
    parser.add_argument("--wiki-urls-file", type=Path, default=shared / "wiki/enwiki-20260501-extracted_urls.txt.gz")
    parser.add_argument(
        "--wet", type=Path, nargs="*", default=[shared / "CC/example.warc.wet.gz"], help="WET files for negatives"
    )
    parser.add_argument("--num-urls", type=int, default=6000)
    parser.add_argument("--max-per-domain", type=int, default=3)
    parser.add_argument("--num-negatives", type=int, default=4000)
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--max-bytes", type=int, default=2_000_000)
    parser.add_argument("--english-threshold", type=float, default=0.7)
    parser.add_argument("--harmful-threshold", type=float, default=0.5)
    parser.add_argument("--valid-fraction", type=float, default=0.1)
    parser.add_argument(
        "--extra-positives",
        type=Path,
        nargs="*",
        default=[],
        help="Additional positives.jsonl.gz files (e.g. from another run with a different --seed) to merge in",
    )
    parser.add_argument(
        "--extra-negatives", type=Path, nargs="*", default=[], help="Additional negatives.jsonl.gz files"
    )
    parser.add_argument("--positive-source", choices=("wiki", "paloma"), default="wiki")
    parser.add_argument("--paloma-bin", type=Path, default=shared / "tokenized_paloma_c4_100_domains_validation.bin")
    parser.add_argument(
        "--positive-label", default=None, help="fastText label for positives (default: wiki, or c4 for paloma)"
    )
    parser.add_argument(
        "--augment-c4-cleaned", type=lambda v: v.lower() in ("1", "true", "yes"), default=True, metavar="BOOL"
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    positive_label = args.positive_label or (HIGH_QUALITY_LABEL if args.positive_source == "wiki" else "c4")

    positives_path = out / "positives.jsonl.gz"
    if args.positive_source == "paloma":
        if not positives_path.exists():
            write_jsonl(positives_path, build_paloma_positives(args.paloma_bin))
    elif not positives_path.exists():
        collect_wiki_positives(args, out, rng, positives_path)
    positives = read_jsonl(positives_path)

    negatives_path = out / "negatives.jsonl.gz"
    if not negatives_path.exists():
        negatives = build_negatives(args.wet, args.num_negatives, rng, english_threshold=args.english_threshold)
        write_jsonl(negatives_path, negatives)
    negatives = read_jsonl(negatives_path)

    for extra in args.extra_positives:
        positives += read_jsonl(extra)
    for extra in args.extra_negatives:
        negatives += read_jsonl(extra)
    positives = list({d["url"]: d for d in positives}.values())
    negatives = list({d["url"]: d for d in negatives}.values())
    print(f"[docs] {len(positives)} positives ({positive_label}), {len(negatives)} negatives", flush=True)
    write_fasttext_files(
        positives,
        negatives,
        out,
        rng,
        args.valid_fraction,
        positive_label=positive_label,
        augment=args.augment_c4_cleaned,
    )


def collect_wiki_positives(args: argparse.Namespace, out: Path, rng: random.Random, positives_path: Path) -> None:
    """Stages 1-3 of the wiki positive pipeline: URLs -> WARC -> filtered documents."""
    urls_path = out / "positive_urls.txt"
    if not urls_path.exists():
        if args.wiki_urls_file.exists():
            urls = sample_urls_from_file(args.wiki_urls_file, args.num_urls, rng, args.max_per_domain)
        else:
            urls = sample_urls_from_wikipedia_api(args.num_urls, rng, args.max_per_domain, ssl_context())
        urls_path.write_text("\n".join(urls) + "\n")
        print(f"[urls] wrote {len(urls)} URLs to {urls_path}", flush=True)
    urls = [line.strip() for line in urls_path.read_text().splitlines() if line.strip()]

    warc_path = out / "positives.warc.gz"
    if not warc_path.exists():
        scrape_to_warc(urls, warc_path, workers=args.workers, timeout=args.timeout, max_bytes=args.max_bytes)

    positives, stats = build_positives(
        warc_path, english_threshold=args.english_threshold, harmful_threshold=args.harmful_threshold
    )
    write_jsonl(positives_path, positives)
    print(f"[positives] {dict(stats)}", flush=True)


if __name__ == "__main__":
    main()
