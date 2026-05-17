"""Evaluate the trained model on the holdout test set.

Reports the critical recall-on-NOT_ACCEPTABLE number, plus per-attribute metrics,
confusion matrices, calibration plot data, and a failure list (FP + FN with
their image_ids) for the HTML report.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from ..common import REPO_ROOT, get_logger, load_config, load_thresholds
from ..training.dataset import TzniutDataset, load_labels_for_training, stratified_split
from ..training.heads import HEAD_DEFINITIONS, head_specs
from ..training.model import TzniutMultiTask
from ..training.train import _TorchDataset, _build_transforms, _load_checkpoint

log = get_logger(__name__)


@torch.no_grad()
def _infer_all(model, loader, device, head_names):
    rows: list[dict] = []
    for imgs, targets in loader:
        imgs = imgs.to(device)
        out = model(imgs)
        bs = imgs.shape[0]
        for i in range(bs):
            row = {}
            for h in head_names:
                row[f"logit_{h}"] = out[h][i].detach().cpu().numpy().tolist()
                v = targets[h][i].item()
                row[f"target_{h}"] = v
            rows.append(row)
    return rows


def _f_beta(prec, rec, beta=2.0):
    if prec == 0 and rec == 0:
        return 0.0
    return (1 + beta * beta) * prec * rec / (beta * beta * prec + rec)


def run(checkpoint_path: str | None = None, holdout_size: int | None = None) -> dict:
    cfg = load_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_size = cfg["training"]["input_size"]

    df = load_labels_for_training(
        REPO_ROOT / "manifests" / "labels.parquet",
        REPO_ROOT / "manifests" / "human_review.parquet",
    )
    _, _, test_df = stratified_split(df)
    if holdout_size:
        test_df = test_df.head(holdout_size)
    log.info("holdout test size: %d", len(test_df))

    tf = _build_transforms(image_size, train=False)
    ds = _TorchDataset(TzniutDataset(test_df, Path("/tmp/tzniut_cache"), tf, image_size))
    loader = DataLoader(ds, batch_size=cfg["training"]["batch_size"], num_workers=2)

    heads = head_specs(cfg["training"]["loss"]["head_weights"])
    model = TzniutMultiTask(cfg["training"]["backbone"], heads).to(device)
    ck = Path(checkpoint_path) if checkpoint_path else REPO_ROOT / "models" / "checkpoints" / "best.pt"
    _load_checkpoint(ck, model)
    model.eval()

    head_names = list(heads.keys())
    rows = _infer_all(model, loader, device, head_names)
    thresholds = load_thresholds()
    block_threshold = float(thresholds.get("block_overall", 0.5))
    temp = float(thresholds.get("temperature", 1.0))

    preds_block = []
    targets_block = []
    failures: list[dict] = []
    per_head_correct: dict[str, int] = {h: 0 for h in head_names}
    per_head_n: dict[str, int] = {h: 0 for h in head_names}

    for r, src_row in zip(rows, test_df.itertuples(index=False)):
        for h in head_names:
            logit = np.array(r[f"logit_{h}"])
            target = r[f"target_{h}"]
            spec = heads[h]
            if spec.kind == "binary":
                prob = 1.0 / (1.0 + np.exp(-(logit[0] / max(temp, 0.01))))
                t = float(target)
                if t < 0:
                    continue
                pred = int(prob >= 0.5)
                per_head_correct[h] += int(pred == int(t))
                per_head_n[h] += 1
                if h == "block":
                    preds_block.append(pred)
                    targets_block.append(int(t))
                    if pred != int(t):
                        failures.append({
                            "image_id": getattr(src_row, "image_id"),
                            "r2_key": getattr(src_row, "r2_key", ""),
                            "predicted_block": bool(pred),
                            "true_block": bool(int(t)),
                            "score": float(prob),
                        })
            else:
                if int(target) == -100:
                    continue
                pred = int(np.argmax(logit))
                per_head_correct[h] += int(pred == int(target))
                per_head_n[h] += 1

    preds_arr = np.array(preds_block)
    targets_arr = np.array(targets_block)
    tp = int(((preds_arr == 1) & (targets_arr == 1)).sum())
    fp = int(((preds_arr == 1) & (targets_arr == 0)).sum())
    fn = int(((preds_arr == 0) & (targets_arr == 1)).sum())
    tn = int(((preds_arr == 0) & (targets_arr == 0)).sum())

    metrics = {
        "block_precision": tp / max(1, tp + fp),
        "block_recall_not_acceptable": tp / max(1, tp + fn),
        "block_specificity": tn / max(1, tn + fp),
        "block_f2": _f_beta(tp / max(1, tp + fp), tp / max(1, tp + fn), beta=2.0),
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "false_negative_rate": fn / max(1, fn + tp),
        "per_head_accuracy": {h: per_head_correct[h] / max(1, per_head_n[h]) for h in head_names},
        "holdout_size": len(test_df),
        "block_threshold": block_threshold,
    }

    out_dir = REPO_ROOT / "models" / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    pd.DataFrame(failures).to_parquet(out_dir / "failures.parquet", index=False)
    pd.DataFrame(rows).to_parquet(out_dir / "raw_predictions.parquet", index=False)
    log.info("metrics → %s", out_dir / "metrics.json")
    log.info(
        "recall(not_acceptable)=%.3f precision=%.3f F2=%.3f FN=%d FP=%d",
        metrics["block_recall_not_acceptable"],
        metrics["block_precision"],
        metrics["block_f2"],
        fn,
        fp,
    )

    req_rec = cfg["eval"]["required_recall_not_acceptable"]
    req_prec = cfg["eval"]["required_precision_not_acceptable"]
    if metrics["block_recall_not_acceptable"] < req_rec:
        log.warning(
            "FAILS recall gate: %.3f < %.3f",
            metrics["block_recall_not_acceptable"],
            req_rec,
        )
    if metrics["block_precision"] < req_prec:
        log.warning("FAILS precision gate: %.3f < %.3f", metrics["block_precision"], req_prec)

    return metrics


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--holdout-size", type=int, default=None)
    a = p.parse_args()
    run(a.checkpoint, a.holdout_size)
