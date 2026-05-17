"""Populates the local review queue from labels.parquet on HuggingFace.

Selection criteria (configurable):
  - flagged_for_review = True (NIM/Gemini disagreement, safety refusal, etc.)
  - labeler_confidence < threshold
  - VLM result with edge-case attributes (sleeve=elbow, lower_length=at_knee, fit=fitted)
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from pipelines.common import get_logger, load_config, require_env
from . import db

log = get_logger(__name__)


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _pull_labels_from_hf(local_path: Path) -> Path:
    from huggingface_hub import hf_hub_download

    local_path.parent.mkdir(parents=True, exist_ok=True)
    return Path(
        hf_hub_download(
            repo_id=require_env("HF_DATASET_REPO"),
            filename="labels.parquet",
            repo_type="dataset",
            local_dir=str(local_path.parent),
        )
    )


def _is_edge_case(raw: str) -> bool:
    try:
        lbl = json.loads(raw or "{}")
    except json.JSONDecodeError:
        return False
    p = lbl.get("primary_person") or {}
    return (
        p.get("sleeve_length") == "elbow"
        or p.get("lower_length") == "at_knee"
        or p.get("fit") == "fitted"
    )


def _selection_mask(df: pd.DataFrame) -> pd.Series:
    cfg = load_config()["labeling"]["flag_for_review"]
    low_conf_threshold = cfg["low_confidence_threshold"]
    flagged = df["flagged_for_review"].fillna(False).astype(bool)
    low_conf = df["confidence"].fillna(0.0) < low_conf_threshold
    edge = df["label_json"].fillna("{}").apply(_is_edge_case)
    return flagged | low_conf | edge


def populate(limit: Optional[int] = None, from_local: Optional[str] = None) -> int:
    if from_local:
        path = Path(from_local)
    else:
        path = _pull_labels_from_hf(Path("manifests/labels.parquet"))

    df = pd.read_parquet(path)
    df = df[df["labeler"].isin({"nim", "gemini"})]  # human review is over VLM labels
    df = df.sort_values("labeled_at").drop_duplicates(subset=["image_id"], keep="last")

    mask = _selection_mask(df)
    picked = df[mask]
    if limit:
        picked = picked.head(limit)

    rows = []
    for _, r in picked.iterrows():
        rows.append(
            {
                "image_id": r["image_id"],
                "r2_key": r["r2_key"],
                "ai_label_json": r["label_json"],
                "ai_confidence": float(r["confidence"]),
                "flag_reason": r.get("review_reason") or None,
                "enqueued_at": _utcnow(),
            }
        )
    n = db.enqueue_many(rows)
    log.info("queue populated: %d new items (selected %d total)", n, len(rows))
    return n


def push_to_hf() -> int:
    rows = db.export_for_hf()
    if not rows:
        return 0
    df = pd.DataFrame(rows)
    out = Path("manifests/human_review.parquet")
    if out.exists():
        prev = pd.read_parquet(out)
        df = pd.concat([prev, df]).drop_duplicates(subset=["image_id"], keep="last")
    df.to_parquet(out, index=False)

    from huggingface_hub import HfApi

    api = HfApi(token=require_env("HF_TOKEN"))
    repo = require_env("HF_DATASET_REPO")
    api.upload_file(
        path_or_fileobj=str(out),
        path_in_repo="human_review.parquet",
        repo_id=repo,
        repo_type="dataset",
    )
    db.mark_pushed([r["image_id"] for r in rows])
    log.info("pushed %d human reviews to HF (%s)", len(rows), repo)
    return len(rows)
