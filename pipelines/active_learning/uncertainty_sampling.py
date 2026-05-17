"""Rank unlabeled images by uncertainty and pick the most informative subset.

Uncertainty score combines:
  - entropy on the block (binary) head
  - max entropy across categorical heads (sleeve, fit, lower_length, etc.)
  - softmax margin (1 - top1) on the most violating head
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from ..common import REPO_ROOT, get_logger, load_config, load_thresholds
from ..collection import r2_client
from ..training.dataset import TzniutDataset
from ..training.heads import head_specs
from ..training.model import TzniutMultiTask
from ..training.train import _TorchDataset, _build_transforms, _load_checkpoint

log = get_logger(__name__)


def _binary_entropy(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return -(p * math.log(p) + (1 - p) * math.log(1 - p))


def _cat_entropy(probs: np.ndarray) -> float:
    p = np.clip(probs, 1e-6, 1.0)
    return float(-np.sum(p * np.log(p)))


@torch.no_grad()
def score_pool(
    model,
    pool_df: pd.DataFrame,
    image_size: int,
    device: str,
    batch_size: int,
    head_names: List[str],
    head_kinds: dict[str, str],
    temperature: float = 1.0,
) -> pd.DataFrame:
    tf = _build_transforms(image_size, train=False)
    ds = _TorchDataset(TzniutDataset(pool_df, Path("/tmp/tzniut_cache"), tf, image_size))
    loader = DataLoader(ds, batch_size=batch_size, num_workers=2)

    scores: list[dict] = []
    iterator = iter(loader)
    df_idx = 0
    for imgs, _ in iterator:
        imgs = imgs.to(device)
        out = model(imgs)
        bs = imgs.shape[0]
        for i in range(bs):
            row = {"image_id": pool_df.iloc[df_idx]["image_id"], "r2_key": pool_df.iloc[df_idx]["r2_key"]}
            df_idx += 1
            ent_sum = 0.0
            block_p = 0.5
            margin_max = 0.0
            for h in head_names:
                logit = out[h][i].cpu().numpy().astype(np.float64) / max(temperature, 0.05)
                if head_kinds[h] == "binary":
                    p = 1.0 / (1.0 + math.exp(-float(logit[0])))
                    ent_sum += _binary_entropy(p)
                    if h == "block":
                        block_p = p
                else:
                    e = np.exp(logit - logit.max())
                    probs = e / e.sum()
                    ent_sum += _cat_entropy(probs) / math.log(len(probs))
                    top = float(probs.max())
                    margin_max = max(margin_max, 1.0 - top)
            row["uncertainty"] = ent_sum + margin_max
            row["block_prob"] = block_p
            scores.append(row)
    return pd.DataFrame(scores)


def select(
    checkpoint_path: str | None = None,
    pool_size: int | None = None,
    selection_size: int | None = None,
) -> Path:
    cfg = load_config()
    al = cfg["active_learning"]
    pool_size = pool_size or al["pool_size"]
    selection_size = selection_size or al["selection_size"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    collection = pd.read_parquet(REPO_ROOT / "manifests" / "collection_deduped.parquet")
    labels_path = REPO_ROOT / "manifests" / "labels.parquet"
    labeled_ids: set[str] = set()
    if labels_path.exists():
        labels = pd.read_parquet(labels_path)
        labeled_ids = set(labels["image_id"].unique())
    unlabeled = collection[~collection["image_id"].isin(labeled_ids)]
    pool = unlabeled.sample(min(pool_size, len(unlabeled)), random_state=int(__import__("time").time()))
    log.info("active learning pool: %d (from %d unlabeled)", len(pool), len(unlabeled))

    heads = head_specs(cfg["training"]["loss"]["head_weights"])
    model = TzniutMultiTask(cfg["training"]["backbone"], heads).to(device)
    ck = Path(checkpoint_path) if checkpoint_path else REPO_ROOT / "models" / "checkpoints" / "best.pt"
    _load_checkpoint(ck, model)
    model.eval()
    head_kinds = {h: spec.kind for h, spec in heads.items()}

    thr = load_thresholds()
    temp = float(thr.get("temperature", 1.0))
    scored = score_pool(
        model, pool, cfg["training"]["input_size"], device,
        cfg["training"]["batch_size"], list(heads.keys()), head_kinds, temperature=temp,
    )
    picked = scored.sort_values("uncertainty", ascending=False).head(selection_size)
    out = REPO_ROOT / "manifests" / f"al_round_selection.parquet"
    picked.to_parquet(out, index=False)
    log.info("selected %d uncertain images → %s", len(picked), out)
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--pool-size", type=int, default=None)
    p.add_argument("--selection-size", type=int, default=None)
    a = p.parse_args()
    select(a.checkpoint, a.pool_size, a.selection_size)
