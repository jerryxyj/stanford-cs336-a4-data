# CS336 Spring 2026 Assignment 4: Data

For a full description of the assignment, see the assignment handout at
[cs336_assignment4_data.pdf](./cs336_assignment4_data.pdf)

If you see any issues with the assignment handout or code, please feel free to
raise a GitHub issue or open a pull request with a fix.

## Setup

This directory is organized as follows:

- [`./cs336_basics`](./cs336_basics): This module contains the staff 
  implementation of the language model from assignment 1. You will use this training code
  to train an LM on your filtered data. You should not modify the training logic, since
  your leaderboard submission must use it exactly.
- [`./cs336_data`](./cs336_data): This folder is basically empty! This is the
  module where you will implement code to filter and process the data.

Visually, it should look something like:

``` sh
.
├── cs336_basics  # A python module named cs336_basics
│   └── ... an optimized training implementation ...
├── cs336_data  # TODO(you): code that you'll write for assignment 4
│   ├── __init__.py
│   └── ... TODO(you): any other files or folders you need for assignment 4 ...
├── README.md
├── pyproject.toml
└── ... TODO(you): other files or folders you need for assignment 4 ...
```

As in previous assignments, we use `uv` to manage dependencies.

## Downloading data

### For students

Data is available at `/shared-data`, such as `uv run modal shell ./scripts/download_data.py::main --cmd "ls -l /shared-data"`

To only download the files needed for running offline, run `uv run scripts/download_data.py --offline-only`.
To download all data for the students, run `uv run modal run scripts/download_data.py`

### For non-students

To only download the files needed for running offline, run `uv run scripts/download_data.py --offline-only`.

Implement the method `is_english` in [`./cs336_data/wet_files.py`](./cs336_data/wet_files.py) before downloading the full non-offline data.

#### Modal

Remove `environment_name` from `shared_data_volume` in [`./cs336_data/modal_utils.py`](./cs336_data/modal_utils.py)

```python
uv run modal run scripts/download_data.py
```

#### Non-modal

Change the path in [`./cs336_data/common.py`](./cs336_data/common.py) and run `uv run scripts/download_data.py`.

Consider changing `n_files` for `EnglishWetFiles` in [`./cs336_data/wet_files.py`](./cs336_data/wet_files.py) if you want to download less than 2.5k WET files.

## Training on Modal

The Modal entrypoint in [`./scripts/train.py`](./scripts/train.py) contains the full training config; pass the path to your GPT-2-tokenized training data with `--train-bin`.

The final training run uses 8 B200 GPUs:

```sh
uv run modal run scripts/train.py --train-bin /root/data/your_data.bin
```

## Submitting

To submit, run `./test_and_make_submission.sh` . This script will install your
code's dependencies, run tests, and create a zip with the output. We
should be able to unzip your submitted tarball and run
`./test_and_make_submission.sh` to verify your test results.

## Implementation notes (this fork)

Everything below was added on top of the staff scaffold. All 21 provided tests pass
(`uv run pytest`).

### Local setup gotchas

- `uv sync --frozen` installs the staff lockfile exactly. If your uv is configured with a private
  index that lacks `furu` (e.g. a corporate proxy), plain `uv run ...` fails while re-resolving the
  lockfile; `export UV_FROZEN=1` (or `uv run --frozen ...`) makes every command below — and
  `test_and_make_submission.sh` — use the hash-pinned lockfile as-is.
- `cs336_data/modal_utils.py` needs a `SUNET_ID`; any identifier works for non-students (it only
  names the Modal app/volume). `build_image` now skips the optional `AGENTS.md` / `CLAUDE.md`.
- Local assets live in `local-shared-data/` next to the repo (override with `CS336_SHARED_DATA=...`).
  On macOS framework Pythons `urllib` may lack CA certificates; run the downloader as
  `SSL_CERT_FILE=$(uv run python -c 'import certifi; print(certifi.where())') uv run scripts/download_data.py --offline-only`.

### Code map

| Module | Problem | What it does |
| --- | --- | --- |
| `cs336_data/extract.py` | `extract_text` | UTF-8 first, `detect_encoding` fallback, `extract_plain_text` |
| `cs336_data/langid.py` | `language_identification` | fastText `lid.176.bin`; returns `(code, prob)`; `is_english` (p ≥ 0.7) is used by `wet_files.py` |
| `cs336_data/pii.py` | `mask_pii` | regexes for emails, US phone formats and IPv4 |
| `cs336_data/harmful.py` | `harmful_content` | Dolma Jigsaw fastText NSFW / hate-speech models |
| `cs336_data/gopher.py` | `gopher_quality_filters` | word count, mean word length, ellipsis lines, alphabetic-word ratio |
| `cs336_data/quality.py` | `quality_classifier` | fastText `wiki` vs `cc` classifier (inference + training helpers) |
| `cs336_data/dedup.py` | `exact_deduplication`, `minhash_deduplication` | hashed line counting; MinHash signatures, LSH banding, Jaccard verification, union-find clusters |
| `cs336_data/c4.py` | `filter_data` | C4-style line/document heuristics (terminal punctuation, ≥5 words/line, no `{`, no lorem ipsum) |
| `cs336_data/pipeline.py` | `filter_data` | per-document cascade `language → gopher → c4 → nsfw/toxic → quality → PII masking` and per-file worker |
| `cs336_data/corpus_dedup.py` | `filter_data` | corpus-level exact line dedup + MinHash/LSH over JSONL shards |
| `cs336_data/tokenization.py` | `tokenize_data` | GPT-2 tokenization with `<|endoftext|>` per document into `np.uint16` |
| `cs336_data/warc.py` | – | FastWARC iterators for WARC responses and WET records |

### Workflow

```sh
# 0. assets (Paloma validation set, example WARC/WET, lid.176 and the Dolma classifiers)
uv run scripts/download_data.py --offline-only

# 1. quality classifier: Wikipedia-referenced pages (positives) vs. random English CC pages (negatives).
#    Without the staff URL dump the script samples external links of random Wikipedia articles via the API,
#    scrapes them into a WARC, extracts text, and keeps English pages that pass the Gopher/harmful filters.
uv run scripts/build_quality_classifier_data.py --num-urls 6000 --num-negatives 4000
uv run scripts/train_quality_classifier.py            # -> local-shared-data/classifiers/quality_classifier.bin

# 2. filter WET files in parallel (one process per file); prints per-stage removal counts + stats.json
uv run scripts/filter_data.py --input-dir /path/to/english-wet-data --output-dir data/filtered --save-discarded-rate 0.01

# 3. corpus-level dedup (exact lines, then MinHash+LSH documents) and 4. tokenization
uv run scripts/dedup_data.py --input-dir data/filtered --output-dir data/deduped
uv run scripts/tokenize_data.py --input-dir data/deduped --output data/tokenized/filtered_cc.bin

# 5. train (unchanged staff script)
uv run modal run scripts/train.py --train-bin /root/data/filtered_cc.bin
```

`scripts/run_pipeline_modal.py` runs steps 2–4 on Modal against `/shared-data/english-wet-data`
(`::filter` maps one container per WET file; `::dedup` / `::tokenize` use one large container).
Upload your quality model to your data volume first (see the script docstring).

`scripts/inspect_data.py` produces the material for the written questions: `wet` (browse WET
records), `warc` (random pages through extraction + every classifier), `compare` (our extraction vs.
WET text), `pii` (masked spans with the original matches), `filtered` / `discarded` (samples from the
pipeline output, the latter grouped by the rejecting stage), `paloma` (decode the validation set).

### Pipeline defaults and local results

Defaults (`cs336_data/pipeline.py::FilterConfig`, all overridable as `--flag` on `filter_data.py`):
English p ≥ 0.85, Gopher rules, C4 line cleaning, NSFW/toxic p < 0.5, quality `1 − P(cc)` ≥ 0.05,
PII masking; dedup with 128 hashes / 16 bands / 5-grams / Jaccard ≥ 0.8.

Running the pipeline on three raw 2026 WET files (58,871 records) on a laptop takes ~5 s per file
per core and gives:

```
documents in: 58,871   kept: 2,518 (4.28%)   discarded: 56,353
  stage                         removed  % of input  % of discarded
  language                       50,109      85.12%          88.92%   (raw WETs; english-wet-data is pre-filtered)
  quality                         4,090       6.95%           7.26%
  gopher                          1,355       2.30%           2.40%
  c4_*                              787       1.34%           1.40%
  toxic / nsfw                       12       0.02%           0.02%
dedup: 7,281 duplicated lines removed, 46 documents dropped as too short, 1 near-duplicate removed
tokens: 3,351,727 for 2,471 documents  (~1.1M tokens per raw WET file -> ~2.8B tokens for 2,500 files)
```

The quality threshold is the main knob: the final run samples 8.6B tokens, so with ~2,500 WET files
a keep rate that yields ≥2B tokens (≤4 epochs) is preferable to a stricter filter. Two classifiers
are available: the required `wiki` vs `cc` model, and (allowed by the handout) a `c4` vs `cc` model
trained with the Paloma documents as positives — build it with
`scripts/build_quality_classifier_data.py --positive-source paloma --output-dir local-shared-data/quality_classifier_c4`,
train it with `--data-dir ... --output local-shared-data/classifiers/quality_classifier_c4.bin`, and pass
`--quality-model` to `filter_data.py`. Both score the C4-cleaned text, so training data for both classes
includes raw and C4-cleaned versions of each page.
