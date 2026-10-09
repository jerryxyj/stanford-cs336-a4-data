"""CS336 Assignment 4: data filtering primitives.

Modules:
  extract     - HTML bytes -> plain text (Resiliparse)
  langid      - fastText language identification
  pii         - email / phone / IP masking
  harmful     - Dolma NSFW and hate-speech classifiers
  gopher      - Gopher rule-based quality filters
  quality     - fastText "wiki vs cc" quality classifier
  dedup       - exact line dedup and MinHash + LSH document dedup
  c4          - C4-style line/document heuristics used by the filtering pipeline
  pipeline    - per-document filtering logic shared by the scripts
"""

from cs336_data.extract import extract_text_from_html_bytes
from cs336_data.gopher import gopher_quality_filter
from cs336_data.pii import mask_emails, mask_ips, mask_phone_numbers
