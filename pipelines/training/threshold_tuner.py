"""Per-attribute threshold tuning on validation set.

For each binary head + each violation class within a categorical head, find
the threshold that maximizes F2 (recall-weighted). Write to config/thresholds.yaml.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from ..common import REPO_ROOT, get_logger, load_config
from .dataset import TzniutDataset, load_labels_for_training, stratified_split
from .heads import HEAD_DEFINITIONS, head_specs
from .model import TzniutMultiTask
from .train import _TorchDataset, _build_transforms, _load_checkpoint

log = get_logger(__name__)

VIOLATION_CLASSES = {
    "visible_nudity": ("partial", "full"),
    "sleeve_length": ("none", "short", "elbow"),
    "neckline": ("cleavage_visible",),
    "lower_garment": ("pants", "shorts", "swimwear", "underwear", "none"),
    "lower_length": ("above_knee", "at_knee"),
    "fit": ("tight",),
}


def _f_beta(prec: float, rec: float, beta: float) -> float:
    if prec == 0 and rec == 0:
        return 0.0
    return (1 + beta * beta) * prec * rec / (beta * beta * prec + rec)


def _tune_binary(probs: np.ndarray, labels: np.ndarray, beta: float = 2.0) -> tuple[float, float]:
    best_t, best_score = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 91):
        pred = probs >= t
        tp = int(((pred == 1) & (labels == 1)).sum())
        fp = int(((pred == 1) & (labels == 0)).sum())
        fn = int(((pred == 0) & (labels == 1)).sum())
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        score = _f_beta(prec, rec, beta)
        if score > best_score:
            best_score = score
            best_t = float(t)
    return best_t, best_score


@torch.no_grad()
def tune(checkpoint_path: str | None = None) -> dict:
    cfg = load_config()["training"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    df = load_labels_for_training(
        REPO_ROOT / "manifests" / "labels.parquet",
        REPO_ROOT / "manifests" / "human_review.parquet",
    )
    _, val_df, _ = stratified_split(df)
    tf = _build_transforms(cfg["input_size"], train=False)
    ds = _TorchDataset(TzniutDataset(val_df, Path("/tmp/tzniut_cache"), tf, cfg["input_size"]))
    loader = DataLoader(ds, batch_size=cfg["batch_size"], num_workers=2)

    heads = head_specs(cfg["loss"]["head_weights"])
    model = TzniutMultiTask(cfg["backbone"], heads).to(device)
    ckpt = Path(checkpoint_path) if checkpoint_path else REPO_ROOT / "models" / "checkpoints" / "best.pt"
    _load_checkpoint(ckpt, model)
    model.eval()

    bin_probs: Dict[str, List[float]] = {}
    bin_labels: Dict[str, List[float]] = {}
    cat_probs: Dict[str, List[np.ndarray]] = {}
    cat_labels: Dict[str, List[int]] = {}

    for imgs, targets in loader:
        imgs = imgs.to(device)
        out = model(imgs)
        for name, spec in heads.items():
            if spec.kind == "binary":
                p = torch.sigmoid(out[name]).squeeze(-1).cpu().numpy()
                t = targets[name].numpy()
                bin_probs.setdefault(name, []).extend(p[t >= 0].tolist())
                bin_labels.setdefault(name, []).extend(t[t >= 0].tolist())
            else:
                p = torch.softmax(out[name], dim=-1).cpu().numpy()
                t = targets[name].numpy()
                mask = t != -100
                cat_probs.setdefault(name, []).extend(p[mask].tolist())
                cat_labels.setdefault(name, []).extend(t[mask].tolist())

    result: dict = {"per_attribute": {"female_violations": {}}}
    for name in ("shirtless_male", "romantic_contact", "suggestive_pose"):
        if name in bin_probs:
            t, score = _tune_binary(np.array(bin_probs[name]), np.array(bin_labels[name]))
            result["per_attribute"][name] = round(t, 3)
            log.info("%s: threshold=%.3f F2=%.3f", name, t, score)
    if "block" in bin_probs:
        t, score = _tune_binary(np.array(bin_probs["block"]), np.array(bin_labels["block"]))
        result["block_overall"] = round(t, 3)
        log.info("block: threshold=%.3f F2=%.3f", t, score)

    classes_map = {n: list(d["classes"]) for n, d in HEAD_DEFINITIONS.items()}
    for head, violating in VIOLATION_CLASSES.items():
        if head not in cat_probs:
            continue
        probs_arr = np.array(cat_probs[head])
        labels_arr = np.array(cat_labels[head])
        for cls in violating:
            cls_idx = classes_map[head].index(cls)
            cls_prob = probs_arr[:, cls_idx]
            cls_label = (labels_arr == cls_idx).astype(int)
            if cls_label.sum() < 10:
                continue
            t, score = _tune_binary(cls_prob, cls_label)
            key = f"{head}_{cls}"
            result["per_attribute"]["female_violations"][key] = round(t, 3)
            log.info("%s: threshold=%.3f F2=%.3f", key, t, score)

    out_path = REPO_ROOT / "config" / "thresholds.yaml"
    existing = yaml.safe_load(out_path.read_text()) if out_path.exists() else {}
    merged = {**existing, **result, "per_attribute": {**existing.get("per_attribute", {}), **result["per_attribute"]}}
    out_path.write_text(yaml.safe_dump(merged, sort_keys=False))
    log.info("thresholds → %s", out_path)
    return merged


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=None)
    a = p.parse_args()
    tune(a.checkpoint)
