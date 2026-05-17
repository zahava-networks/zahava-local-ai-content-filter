"""Build the INT8 quantization calibration set.

Pulls a representative ~500-image sample (class-balanced) from the training
labels and saves them as a NumPy array of normalized tensors. Used by both
TFLite (post-training quantization) and CoreML (weight palettization).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ..common import REPO_ROOT, get_logger, load_config
from ..collection import r2_client
from ..training.dataset import TzniutDataset, load_labels_for_training
from ..training.train import _build_transforms

log = get_logger(__name__)


def build(n: int = None, output: str | None = None) -> Path:
    cfg = load_config()
    n = n or cfg["export"]["tflite"]["calibration_samples"]
    image_size = cfg["training"]["input_size"]

    df = load_labels_for_training(
        REPO_ROOT / "manifests" / "labels.parquet",
        REPO_ROOT / "manifests" / "human_review.parquet",
    )
    blocked = df[df["block"]]
    allowed = df[~df["block"]]
    half = n // 2
    sample = pd.concat([blocked.sample(min(half, len(blocked)), random_state=7),
                        allowed.sample(min(n - half, len(allowed)), random_state=7)])
    log.info("calibration: %d images (block=%d allow=%d)", len(sample), len(blocked), len(allowed))

    tf = _build_transforms(image_size, train=False)
    ds = TzniutDataset(sample.reset_index(drop=True), Path("/tmp/tzniut_cache"), tf, image_size)
    arr = np.zeros((len(ds), 3, image_size, image_size), dtype=np.float32)
    for i in range(len(ds)):
        x, _ = ds[i]
        arr[i] = x.numpy() if hasattr(x, "numpy") else x

    out = Path(output) if output else REPO_ROOT / "models" / "calibration_set.npy"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, arr)
    log.info("calibration set → %s (shape=%s)", out, arr.shape)
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=None)
    p.add_argument("--output", default=None)
    a = p.parse_args()
    build(a.n, a.output)
