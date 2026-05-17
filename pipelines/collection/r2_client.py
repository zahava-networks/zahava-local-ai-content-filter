"""Blob storage backed by HuggingFace Datasets.

Despite the file name (kept for import stability), this stores images inside
the configured HF Dataset repo. The `key` is the relative path inside the repo.

The pipeline only needs:
  - upload_bytes(key, data)         — write a single object
  - exists(key)                     — check membership (cached)
  - download_bytes(key)             — read a single object
  - list_keys(prefix, limit=None)   — enumerate objects under a prefix
  - object_url(key)                 — a URL the *local server* can fetch with auth.
                                      NOT safe to put in <img> tags for private repos.

For browser display, the review UI proxies the bytes via /api/image/{image_id}.
"""
from __future__ import annotations

import io
from functools import lru_cache
from typing import Iterator, Optional

from huggingface_hub import HfApi, hf_hub_download, hf_hub_url

from ..common import load_config, require_env


@lru_cache(maxsize=1)
def _api() -> HfApi:
    return HfApi(token=require_env("HF_TOKEN"))


@lru_cache(maxsize=1)
def _repo() -> str:
    return require_env("HF_DATASET_REPO")


def bucket() -> str:
    return _repo()


@lru_cache(maxsize=1)
def _known_files() -> set[str]:
    try:
        files = _api().list_repo_files(_repo(), repo_type="dataset")
        return set(files)
    except Exception:
        return set()


def _invalidate_cache() -> None:
    _known_files.cache_clear()


def upload_bytes(key: str, data: bytes, content_type: str = "image/webp") -> None:
    _api().upload_file(
        path_or_fileobj=io.BytesIO(data),
        path_in_repo=key,
        repo_id=_repo(),
        repo_type="dataset",
    )
    cached = _known_files()
    cached.add(key)


def exists(key: str) -> bool:
    if key in _known_files():
        return True
    try:
        _api().get_paths_info(_repo(), [key], repo_type="dataset")
        cached = _known_files()
        cached.add(key)
        return True
    except Exception:
        return False


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
    """URL for server-side fetching (NOT browser-safe for private repos)."""
    return hf_hub_url(repo_id=_repo(), filename=key, repo_type="dataset")


def presigned_get_url(key: str, expires_in: int = 600) -> str:
    """Backward-compatible alias.

    For the local review UI: prefer the /api/image/{image_id} proxy endpoint
    over this. This returns the bare HF URL which only works for *public* repos
    when loaded in a browser.
    """
    return object_url(key)


def list_keys(prefix: str = "", limit: Optional[int] = None) -> Iterator[str]:
    files = _known_files()
    if not files:
        try:
            files = set(_api().list_repo_files(_repo(), repo_type="dataset"))
            _known_files.cache_clear()
            _known_files()  # repopulate
        except Exception:
            return
    n = 0
    for f in sorted(files):
        if f.startswith(prefix):
            yield f
            n += 1
            if limit and n >= limit:
                return
