"""Shared collector base: ingest bytes → R2 + manifest row, with resumability."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from ..common import get_logger, load_config
from . import manifest as mf
from . import r2_client
from .image_utils import (
    BadImage,
    content_id,
    encode_webp,
    open_validated,
    perceptual_hash,
    resize_for_storage,
)

log = get_logger(__name__)


@dataclass
class IngestResult:
    image_id: str
    skipped: bool
    reason: Optional[str] = None


@retry(
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=1, max=20),
    retry=retry_if_exception_type((requests.RequestException, OSError)),
    reraise=True,
)
def fetch_url(url: str, timeout: int = 30) -> bytes:
    headers = {"User-Agent": "TzniutClassifier/0.1 (image-collection)"}
    r = requests.get(url, timeout=timeout, headers=headers)
    r.raise_for_status()
    return r.content


def ingest(
    source: str,
    source_id: str,
    raw: bytes,
    url_original: str,
    license: str,
) -> IngestResult:
    """Process a single image: validate, resize, encode WebP, upload to R2, append manifest.

    Returns IngestResult. Safe to call repeatedly with the same image — checks R2
    presence before uploading and manifest knowledge before appending.
    """
    cfg = load_config()
    try:
        img = open_validated(raw)
    except BadImage as e:
        return IngestResult(image_id="", skipped=True, reason=f"bad_image:{e}")

    iid = content_id(raw)

    known = _cached_known_ids(source)
    if iid in known:
        return IngestResult(image_id=iid, skipped=True, reason="already_in_manifest")

    resized = resize_for_storage(img)
    body = encode_webp(resized)
    # Shard into 256 sub-directories by the first 2 hex chars of the content
    # hash. HF/git rejects directories with more than 10k files; sharding
    # gives us up to 2.56M files per source.
    shard = iid[4:6] if len(iid) >= 6 else "00"
    r2_key = f"{cfg['storage']['prefix_resized']}{source}/{shard}/{iid}.webp"

    if not r2_client.exists(r2_key):
        r2_client.upload_bytes(r2_key, body, content_type="image/webp")

    row = mf.ManifestRow(
        image_id=iid,
        source=source,
        source_id=source_id,
        r2_key=r2_key,
        url_original=url_original,
        width=resized.size[0],
        height=resized.size[1],
        format="webp",
        file_size=len(body),
        phash=perceptual_hash(img),
        license=license,
        collected_at=mf.utcnow_iso(),
    )
    mf.append(source, row)
    known.add(iid)
    return IngestResult(image_id=iid, skipped=False)


_KNOWN: dict[str, set[str]] = {}


def _cached_known_ids(source: str) -> set[str]:
    if source not in _KNOWN:
        _KNOWN[source] = mf.known_ids(source)
    return _KNOWN[source]
