"""One full active-learning round:
  1. Score the unlabeled pool, pick most uncertain
  2. Run NSFW oracle + VLM labelers on the selection
  3. Append to labels.parquet
  4. Retrain (warm start from previous best)
  5. Re-tune thresholds + recalibrate
  6. Re-eval on holdout
  7. Append metrics to active_learning/history.json
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ..common import REPO_ROOT, get_logger
from ..labeling import nsfw_oracle, vlm_labeler
from ..labeling import labels_store
from ..training import calibrator, threshold_tuner, train
from ..eval import benchmark
from . import uncertainty_sampling

log = get_logger(__name__)


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def run_round(round_idx: int) -> dict:
    log.info("=== AL round %d ===", round_idx)
    round_name = f"vlm_al_round_{round_idx}"

    selection_path = uncertainty_sampling.select()
    selection = pd.read_parquet(selection_path)
    log.info("labeling %d selected images", len(selection))

    sub_manifest_path = REPO_ROOT / "manifests" / f"al_round_{round_idx}_manifest.parquet"
    selection[["image_id", "r2_key"]].assign(
        source="al_round",
        source_id="",
        url_original="",
        width=224,
        height=224,
        format="webp",
        file_size=0,
        phash="",
        license="",
        collected_at=_utcnow(),
    ).to_parquet(sub_manifest_path, index=False)

    nsfw_oracle.run(str(sub_manifest_path))
    vlm_labeler.run(manifest_path=str(sub_manifest_path), round_name=round_name)
    labels_store.merge_to_parquet("labels.parquet")

    log.info("retraining (warm-start)")
    best = train.train(epochs=None)
    log.info("threshold tuning")
    threshold_tuner.tune()
    log.info("calibration")
    calibrator.calibrate()
    log.info("benchmark")
    metrics = benchmark.run()

    history_path = REPO_ROOT / "models" / "eval" / "history.json"
    history = json.loads(history_path.read_text()) if history_path.exists() else {"rounds": []}
    history["rounds"].append(
        {
            "round": round_idx,
            "completed_at": _utcnow(),
            "metrics": metrics,
            "selection_count": int(len(selection)),
        }
    )
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(json.dumps(history, indent=2))
    log.info("round %d done; recall=%.3f precision=%.3f", round_idx, metrics["block_recall_not_acceptable"], metrics["block_precision"])
    return metrics


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--round", type=int, required=True)
    a = p.parse_args()
    run_round(a.round)
