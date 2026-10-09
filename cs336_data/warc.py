"""Helpers for iterating over WARC / WET archives with FastWARC."""

from __future__ import annotations

import os
from collections.abc import Iterator
from dataclasses import dataclass

from fastwarc.warc import ArchiveIterator, WarcRecordType


@dataclass(frozen=True)
class WarcResponse:
    url: str
    record_id: str
    content_type: str
    payload: bytes


@dataclass(frozen=True)
class WetRecord:
    url: str
    record_id: str
    text: str


def iter_warc_responses(path: os.PathLike | str, *, html_only: bool = True) -> Iterator[WarcResponse]:
    """Yield HTTP response records (URL + raw HTTP payload bytes) from a WARC file."""
    # FastWARC detects gzip itself and needs a seekable stream, so use a plain file object.
    with open(path, "rb") as stream:
        for record in ArchiveIterator(stream, record_types=WarcRecordType.response, parse_http=True):
            http_headers = record.http_headers
            content_type = (http_headers.get("Content-Type") if http_headers is not None else None) or ""
            if html_only and content_type and "html" not in content_type.lower():
                continue
            yield WarcResponse(
                url=record.headers.get("WARC-Target-URI", ""),
                record_id=record.headers.get("WARC-Record-ID", ""),
                content_type=content_type,
                payload=record.reader.read(),
            )


def iter_wet_records(path: os.PathLike | str) -> Iterator[WetRecord]:
    """Yield the extracted-text ("conversion") records of a WET file."""
    with open(path, "rb") as stream:
        for record in ArchiveIterator(stream, record_types=WarcRecordType.conversion, parse_http=False):
            yield WetRecord(
                url=record.headers.get("WARC-Target-URI", ""),
                record_id=record.headers.get("WARC-Record-ID", ""),
                text=record.reader.read().decode("utf-8", errors="replace"),
            )
