"""Blob storage backed by HuggingFace Datasets, with BATCHED commits.

Original approach (one upload_file per image) is silently dropped by HF Hub
when called rapidly — every image becomes its own commit, which HF rate-limits.

This module stages files on local disk, then commits them in batches via
upload_folder (one HF commit per ~100 files). Much more efficient and
actually persists.

Public interface (same as before — callers don't change):
  - upload_bytes(key, data)            — stage one file; auto-flushes at BATCH_SIZE
  - flush_pending(min_count=1)         — commit staged files now
  - exists(key)                        — check membership against repo + staging
  - download_bytes(key)                — read a single object
  - list_keys(prefix, limit=None)      — enumerate objects under a prefix
  - object_url(key)                    — URL for server-side fetching

Collectors should call flush_pending() once at the end of each source
to commit anything below BATCH_SIZE.
"""
from __future__ import annotations

import io
import re
import shutil
import time
from functools import lru_cache
from pathlib import Path
from typing import Iterator, Optional

from huggingface_hub import HfApi, hf_hub_download, hf_hub_url

from ..common import REPO_ROOT, get_logger, require_env

log = get_logger(__name__)

_STAGING = REPO_ROOT / ".hf_staging"
_BATCH_SIZE = 2000  # files staged before auto-flush; large to minimize commits
_COMMIT_LIMIT_PER_HOUR = 128  # HF hard limit on free tier
_MIN_COMMIT_INTERVAL_S = 3600 / _COMMIT_LIMIT_PER_HOUR  # ≈28s
_last_commit_ts: float = 0.0


@lru_cache(maxsize=1)
def _api() -> HfApi:
    return HfApi(token=require_env("HF_TOKEN"))


@lru_cache(maxsize=1)
def _repo() -> str:
    return require_env("HF_DATASET_REPO")


def bucket() -> str:
    return _repo()


@lru_cache(maxsize=1)
def _known_files_cache() -> set[str]:
    try:
        return set(_api().list_repo_files(_repo(), repo_type="dataset"))
    except Exception as e:
        log.warning("list_repo_files failed (will retry on first miss): %s", e)
        return set()


def _staged_paths() -> list[Path]:
    if not _STAGING.exists():
        return []
    return [p for p in _STAGING.rglob("*") if p.is_file()]


def _staged_count() -> int:
    return len(_staged_paths())


def upload_bytes(key: str, data: bytes, content_type: str = "image/webp") -> None:
    """Stage one file for the next batch commit; auto-flush at BATCH_SIZE."""
    p = _STAGING / key
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    if _staged_count() >= _BATCH_SIZE:
        flush_pending()


def _parse_retry_after(msg: str) -> int:
    """Extract sleep duration from HF 429 error message."""
    m = re.search(r"Retry after (\d+) seconds?", msg)
    if m:
        return int(m.group(1)) + 5
    if "commits" in msg and "per hour" in msg:
        return 3600
    return 60


def flush_pending(min_count: int = 1) -> int:
    """Commit all staged files in a single HF commit. Returns count committed.

    Self-throttles to stay under HF's 128-commits-per-hour limit. On 429
    rate-limit errors, sleeps for the duration HF specifies and retries.
    """
    global _last_commit_ts
    staged = _staged_paths()
    n = len(staged)
    if n < min_count:
        return 0

    elapsed = time.time() - _last_commit_ts
    if elapsed < _MIN_COMMIT_INTERVAL_S:
        wait = _MIN_COMMIT_INTERVAL_S - elapsed
        log.info("pre-flush throttle: sleeping %.1fs to stay under 128/hr", wait)
        time.sleep(wait)

    log.info("flushing %d staged files to HF (one commit)", n)
    for attempt in range(6):
        try:
            _api().upload_folder(
                folder_path=str(_STAGING),
                path_in_repo="",
                repo_id=_repo(),
                repo_type="dataset",
                commit_message=f"batch upload: {n} files",
            )
            _last_commit_ts = time.time()
            break
        except Exception as e:
            msg = str(e)
            if "429" in msg or "Too Many Requests" in msg or "rate limit" in msg.lower():
                wait_s = _parse_retry_after(msg)
                log.warning(
                    "HF rate-limit hit (attempt %d/6): sleeping %ds before retry",
                    attempt + 1,
                    wait_s,
                )
                time.sleep(wait_s)
                continue
            log.exception("flush_pending failed: %s — keeping staged files for next attempt", e)
            raise
    else:
        raise RuntimeError("flush_pending: 6 retries exhausted; staged files preserved on disk")

    cache = _known_files_cache()
    for f in staged:
        rel = f.relative_to(_STAGING)
        cache.add(str(rel))
    shutil.rmtree(_STAGING)
    _STAGING.mkdir(parents=True, exist_ok=True)
    log.info("flush OK: %d files now in HF dataset", n)
    return n


def exists(key: str) -> bool:
    if key in _known_files_cache():
        return True
    staged = _STAGING / key
    return staged.exists()


def download_bytes(key: str) -> bytes:
    path = hf_hub_download(
        repo_id=_repo(),
        filename=key,
        repo_type="dataset",
        token=require_env("HF_TOKEN"),
    )
    with open(path, "rb") as f:
        return f.read()


def object_url(key: str) -> str:
    return hf_hub_url(repo_id=_repo(), filename=key, repo_type="dataset")


def presigned_get_url(key: str, expires_in: int = 600) -> str:
    return object_url(key)


def list_keys(prefix: str = "", limit: Optional[int] = None) -> Iterator[str]:
    n = 0
    for f in sorted(_known_files_cache()):
        if f.startswith(prefix):
            yield f
            n += 1
            if limit and n >= limit:
                return
