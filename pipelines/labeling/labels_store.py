"""Append-only labels store: JSONL per round, mergeable to Parquet.

Schema mirrors pipelines.labeling.schema.ImageLabel, plus provenance fields:
  - image_id, source, r2_key (from manifest)
  - labeler ("falconsai" | "nim" | "gemini" | "human")
  - labeled_at (ISO timestamp)
  - human_reviewed (bool)
  - flagged_for_review (bool, with reason)
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

from ..common import manifests_dir


@dataclass
class LabelRecord:
    image_id: str
    r2_key: str
    labeler: str
    labeled_at: str
    label_json: str
    block: bool
    confidence: float
    violations_json: str
    flagged_for_review: bool = False
    review_reason: Optional[str] = None
    human_reviewed: bool = False


def _labels_dir() -> Path:
    p = manifests_dir() / "labels"
    p.mkdir(parents=True, exist_ok=True)
    return p


def path_for_round(round_name: str) -> Path:
    return _labels_dir() / f"{round_name}.jsonl"


_path = path_for_round  # legacy alias


def append(round_name: str, record: LabelRecord) -> None:
    with open(_path(round_name), "a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(record)) + "\n")


def append_many(round_name: str, records: Iterable[LabelRecord]) -> int:
    n = 0
    with open(_path(round_name), "a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(asdict(r)) + "\n")
            n += 1
    return n


def known_ids(round_name: str) -> set[str]:
    path = _path(round_name)
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


def merge_to_parquet(output_name: str = "labels.parquet") -> Path:
    rows: list[dict] = []
    for jsonl in _labels_dir().glob("*.jsonl"):
        with open(jsonl, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rows.append(json.loads(line))
    if not rows:
        raise RuntimeError("No label rows yet — run a labeler first.")
    df = pd.DataFrame(rows)
    df = df.sort_values(["image_id", "labeled_at"]).drop_duplicates(
        subset=["image_id", "labeler"], keep="last"
    )
    out = manifests_dir() / output_name
    df.to_parquet(out, index=False)
    return out
