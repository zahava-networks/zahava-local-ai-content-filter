"""Temperature scaling on the validation set.

Logits get divided by a learned temperature T so that
softmax/sigmoid outputs are properly calibrated as probabilities.
Important for the active learning loop (uncertainty sampling
needs trustworthy confidence).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from ..common import REPO_ROOT, get_logger, load_config
from .dataset import TzniutDataset, load_labels_for_training, stratified_split
from .heads import head_specs
from .model import TzniutMultiTask
from .train import _TorchDataset, _build_transforms, _load_checkpoint

log = get_logger(__name__)


@torch.no_grad()
def _collect(model, loader, device):
    logits = {}
    labels = {}
    for imgs, t in loader:
        imgs = imgs.to(device)
        out = model(imgs)
        for k, v in out.items():
            logits.setdefault(k, []).append(v.detach().cpu())
            labels.setdefault(k, []).append(t[k])
    for k in logits:
        logits[k] = torch.cat(logits[k])
        labels[k] = torch.cat(labels[k])
    return logits, labels


def _fit_T(logits: torch.Tensor, targets: torch.Tensor, kind: str) -> float:
    valid = (targets != -100) if kind == "categorical" else (targets >= 0)
    if valid.sum() < 32:
        return 1.0
    logits = logits[valid]
    targets = targets[valid]
    T = nn.Parameter(torch.ones(1))
    opt = torch.optim.LBFGS([T], lr=0.05, max_iter=50)
    if kind == "categorical":
        loss_fn = nn.CrossEntropyLoss()
        targets_long = targets.long()

        def closure():
            opt.zero_grad()
            l = logits / T.clamp_min(0.05)
            loss = loss_fn(l, targets_long)
            loss.backward()
            return loss
    else:
        loss_fn = nn.BCEWithLogitsLoss()
        targets_f = targets.float()

        def closure():
            opt.zero_grad()
            l = logits / T.clamp_min(0.05)
            loss = loss_fn(l.squeeze(-1), targets_f)
            loss.backward()
            return loss
    opt.step(closure)
    return float(T.clamp_min(0.05).item())


def calibrate(checkpoint_path: str | None = None) -> dict:
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
    ck_path = Path(checkpoint_path) if checkpoint_path else REPO_ROOT / "models" / "checkpoints" / "best.pt"
    _load_checkpoint(ck_path, model)
    model.eval()

    logits, labels = _collect(model, loader, device)
    temps = {}
    for name, spec in heads.items():
        T = _fit_T(logits[name], labels[name], spec.kind)
        temps[name] = round(T, 3)
        log.info("%s: T=%.3f", name, T)

    thr_path = REPO_ROOT / "config" / "thresholds.yaml"
    existing = yaml.safe_load(thr_path.read_text()) if thr_path.exists() else {}
    existing["temperatures_per_head"] = temps
    existing["temperature"] = float(np.mean(list(temps.values())))
    thr_path.write_text(yaml.safe_dump(existing, sort_keys=False))
    log.info("temperatures → %s", thr_path)
    return temps


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=None)
    a = p.parse_args()
    calibrate(a.checkpoint)
