"""Append-only per-source JSONL manifest, mergeable to one Parquet."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd

from ..common import manifests_dir


@dataclass(frozen=True)
class ManifestRow:
    image_id: str
    source: str
    source_id: str
    r2_key: str
    url_original: str
    width: int
    height: int
    format: str
    file_size: int
    phash: str
    license: str
    collected_at: str


def _source_path(source: str) -> Path:
    d = manifests_dir() / "collection"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{source}.jsonl"


def append(source: str, row: ManifestRow) -> None:
    path = _source_path(source)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(row)) + "\n")


def append_many(source: str, rows: Iterable[ManifestRow]) -> int:
    path = _source_path(source)
    n = 0
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(asdict(r)) + "\n")
            n += 1
    return n


def known_ids(source: str) -> set[str]:
    """Image ids already present in the manifest for this source. Used for resumability."""
    path = _source_path(source)
    if not path.exists():
        return set()
    seen: set[str] = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                seen.add(json.loads(line)["image_id"])
            except Exception:
                continue
    return seen


def count(source: str) -> int:
    path = _source_path(source)
    if not path.exists():
        return 0
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def merge_to_parquet(output_name: str = "collection.parquet") -> Path:
    rows: list[dict] = []
    for jsonl in (manifests_dir() / "collection").glob("*.jsonl"):
        with open(jsonl, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rows.append(json.loads(line))
    if not rows:
        raise RuntimeError("No manifest rows found. Run a collector first.")
    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=["image_id"], keep="first")
    out = manifests_dir() / output_name
    df.to_parquet(out, index=False)
    return out


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
