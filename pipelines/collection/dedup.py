"""Perceptual-hash deduplication against the merged manifest."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ..common import get_logger, load_config, manifests_dir

log = get_logger(__name__)


def hamming(a: str, b: str) -> int:
    """Hamming distance between two hex-encoded phashes."""
    if len(a) != len(b):
        return 64
    ai = int(a, 16)
    bi = int(b, 16)
    return bin(ai ^ bi).count("1")


def find_duplicates(parquet_path: Path | None = None) -> pd.DataFrame:
    """Returns a DataFrame of (image_id_keep, image_id_drop) pairs for near-duplicates."""
    threshold = load_config()["collection"]["dedup"]["hamming_threshold"]
    if parquet_path is None:
        parquet_path = manifests_dir() / "collection.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(parquet_path)

    df = pd.read_parquet(parquet_path)
    df = df.sort_values("collected_at").reset_index(drop=True)

    by_hash: dict[str, list[int]] = {}
    drops: list[tuple[str, str]] = []
    for idx, row in df.iterrows():
        ph = row["phash"]
        for cand_hash, cand_idxs in by_hash.items():
            if hamming(ph, cand_hash) <= threshold:
                drops.append((df.loc[cand_idxs[0], "image_id"], row["image_id"]))
                break
        else:
            by_hash.setdefault(ph, []).append(idx)

    return pd.DataFrame(drops, columns=["image_id_keep", "image_id_drop"])


def apply_dedup(parquet_path: Path | None = None) -> Path:
    """Writes a deduped manifest next to the original."""
    if parquet_path is None:
        parquet_path = manifests_dir() / "collection.parquet"
    df = pd.read_parquet(parquet_path)
    dupes = find_duplicates(parquet_path)
    drop_ids = set(dupes["image_id_drop"])
    log.info("dedup: dropping %d of %d", len(drop_ids), len(df))
    kept = df[~df["image_id"].isin(drop_ids)].copy()
    out = parquet_path.with_name(parquet_path.stem + "_deduped.parquet")
    kept.to_parquet(out, index=False)
    return out
