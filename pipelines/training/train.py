"""Multi-task training loop with checkpoint resume.

Designed for Colab free tier — saves checkpoint every N steps so disconnects
don't lose progress. Resume just re-runs the same script.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from ..common import REPO_ROOT, get_logger, load_config
from .dataset import TzniutDataset, load_labels_for_training, stratified_split
from .heads import head_specs
from .model import TzniutMultiTask, build_loss_fn

log = get_logger(__name__)


class _TorchDataset(Dataset):
    def __init__(self, base: TzniutDataset) -> None:
        self.base = base

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        return self.base[idx]


def _build_transforms(image_size: int, train: bool):
    from torchvision import transforms

    if train:
        return transforms.Compose(
            [
                transforms.Resize(int(image_size * 1.15)),
                transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.4, 0.4, 0.4),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                transforms.RandomErasing(p=0.25),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.15)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def _checkpoint_dir() -> Path:
    p = REPO_ROOT / "models" / "checkpoints"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _save_checkpoint(model, optimizer, scheduler, scaler, step, epoch, val_metrics: dict, name: str = "latest"):
    ck = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler else None,
        "scaler": scaler.state_dict() if scaler else None,
        "step": step,
        "epoch": epoch,
        "val_metrics": val_metrics,
    }
    path = _checkpoint_dir() / f"{name}.pt"
    tmp = path.with_suffix(".pt.tmp")
    torch.save(ck, tmp)
    tmp.replace(path)
    log.info("checkpoint saved → %s (step=%d epoch=%d)", path, step, epoch)


def _load_checkpoint(path: Path, model, optimizer=None, scheduler=None, scaler=None):
    ck = torch.load(path, map_location="cpu")
    model.load_state_dict(ck["model"])
    if optimizer and "optimizer" in ck:
        optimizer.load_state_dict(ck["optimizer"])
    if scheduler and ck.get("scheduler"):
        scheduler.load_state_dict(ck["scheduler"])
    if scaler and ck.get("scaler"):
        scaler.load_state_dict(ck["scaler"])
    return ck.get("step", 0), ck.get("epoch", 0)


@torch.no_grad()
def evaluate(model, loader, device, head_names, max_batches: Optional[int] = None) -> dict:
    model.eval()
    correct: dict[str, int] = {h: 0 for h in head_names}
    n: dict[str, int] = {h: 0 for h in head_names}
    block_tp = block_fp = block_fn = block_tn = 0
    for i, (imgs, targets) in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        imgs = imgs.to(device, non_blocking=True)
        logits = model(imgs)
        for h in head_names:
            t = targets[h]
            l = logits[h].detach().cpu()
            if t.dtype == torch.float32:
                pred = (torch.sigmoid(l).squeeze(-1) > 0.5).long()
                mask = t >= 0
                tt = t[mask].long()
                pp = pred[mask]
                if h == "block":
                    block_tp += int(((pp == 1) & (tt == 1)).sum())
                    block_fp += int(((pp == 1) & (tt == 0)).sum())
                    block_fn += int(((pp == 0) & (tt == 1)).sum())
                    block_tn += int(((pp == 0) & (tt == 0)).sum())
                correct[h] += int((pp == tt).sum())
                n[h] += int(mask.sum())
            else:
                pred = l.argmax(dim=-1)
                mask = t != -100
                correct[h] += int((pred[mask] == t[mask]).sum())
                n[h] += int(mask.sum())
    metrics: dict = {f"acc_{h}": (correct[h] / max(1, n[h])) for h in head_names}
    metrics["block_precision"] = block_tp / max(1, block_tp + block_fp)
    metrics["block_recall"] = block_tp / max(1, block_tp + block_fn)
    metrics["block_f2"] = (
        5 * metrics["block_precision"] * metrics["block_recall"]
        / max(1e-9, 4 * metrics["block_precision"] + metrics["block_recall"])
    )
    return metrics


def train(
    labels_parquet: Optional[str] = None,
    human_review_parquet: Optional[str] = None,
    image_cache: str = "/tmp/tzniut_cache",
    resume: bool = True,
    backbone: Optional[str] = None,
    epochs: Optional[int] = None,
) -> Path:
    cfg = load_config()["training"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("device=%s", device)

    labels_parquet = labels_parquet or str(REPO_ROOT / "manifests" / "labels.parquet")
    human_review_parquet = human_review_parquet or str(REPO_ROOT / "manifests" / "human_review.parquet")
    df = load_labels_for_training(Path(labels_parquet), Path(human_review_parquet))
    train_df, val_df, _ = stratified_split(df)

    image_size = cfg["input_size"]
    cache_dir = Path(image_cache)
    train_tf = _build_transforms(image_size, train=True)
    val_tf = _build_transforms(image_size, train=False)

    train_ds = _TorchDataset(TzniutDataset(train_df, cache_dir, train_tf, image_size))
    val_ds = _TorchDataset(TzniutDataset(val_df, cache_dir, val_tf, image_size))

    train_loader = DataLoader(
        train_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=4,
        pin_memory=(device == "cuda"), drop_last=True, persistent_workers=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=2,
        pin_memory=(device == "cuda"), persistent_workers=True,
    )

    heads = head_specs(cfg["loss"]["head_weights"])
    backbone_name = backbone or cfg["backbone"]
    try:
        model = TzniutMultiTask(backbone_name, heads)
    except Exception as e:
        log.warning("backbone %s failed (%s); falling back to %s", backbone_name, e, cfg["fallback_backbone"])
        model = TzniutMultiTask(cfg["fallback_backbone"], heads)
    model = model.to(device)
    log.info("model params: %d", model.num_parameters())

    loss_fn = build_loss_fn(heads, pos_weight_block=cfg["loss"]["head_weights"].get("block", 3.0))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"]
    )
    total_epochs = epochs or cfg["epochs_round1"]
    total_steps = max(1, total_epochs * max(1, len(train_loader)))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=cfg["learning_rate"],
        total_steps=total_steps,
        pct_start=cfg["warmup_epochs"] / max(1, total_epochs),
        anneal_strategy="cos",
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    step = 0
    start_epoch = 0
    latest = _checkpoint_dir() / "latest.pt"
    if resume and latest.exists():
        step, start_epoch = _load_checkpoint(latest, model, optimizer, scheduler, scaler)
        log.info("resumed from step=%d epoch=%d", step, start_epoch)

    save_every = cfg["checkpoint"]["save_steps"]
    head_names = list(heads.keys())
    best_f2 = 0.0
    for epoch in range(start_epoch, total_epochs):
        model.train()
        t0 = time.time()
        running = 0.0
        for imgs, targets in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                logits = model(imgs)
                total, _ = loss_fn(logits, targets)
            scaler.scale(total).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running += float(total.item())
            step += 1
            if step % 50 == 0:
                log.info("epoch=%d step=%d loss=%.4f lr=%.5g", epoch, step, running / 50, scheduler.get_last_lr()[0])
                running = 0.0
            if step % save_every == 0:
                _save_checkpoint(model, optimizer, scheduler, scaler, step, epoch, {}, name="latest")
        metrics = evaluate(model, val_loader, device, head_names, max_batches=200)
        log.info("epoch=%d eval: %s", epoch, json.dumps({k: round(v, 4) for k, v in metrics.items()}))
        _save_checkpoint(model, optimizer, scheduler, scaler, step, epoch + 1, metrics, name="latest")
        if metrics["block_f2"] > best_f2:
            best_f2 = metrics["block_f2"]
            _save_checkpoint(model, optimizer, scheduler, scaler, step, epoch + 1, metrics, name="best")
        log.info("epoch %d done in %.1fs", epoch, time.time() - t0)

    return _checkpoint_dir() / "best.pt"


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--labels", default=None)
    p.add_argument("--human-review", default=None)
    p.add_argument("--cache", default="/tmp/tzniut_cache")
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--backbone", default=None)
    p.add_argument("--epochs", type=int, default=None)
    a = p.parse_args()
    train(a.labels, a.human_review, a.cache, resume=not a.no_resume, backbone=a.backbone, epochs=a.epochs)
