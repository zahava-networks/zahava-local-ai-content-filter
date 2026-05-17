"""PyTorch dataset: streams labels from parquet, images from R2 (cached locally).

Encoding:
  - categorical heads → integer class index (long), with -100 ignore_index for "not_visible"
  - binary heads → float 0/1 with valid_mask (some labels may be missing)
"""
from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from PIL import Image

from ..common import get_logger
from ..collection import r2_client
from .heads import HEAD_DEFINITIONS

log = get_logger(__name__)


class TzniutDataset:
    """Dataset that yields (image_tensor, labels_dict).

    Subclassed for PyTorch via torch.utils.data.Dataset in the train script.
    Kept framework-light here so it can be imported without torch installed.
    """

    def __init__(
        self,
        labels_df: pd.DataFrame,
        image_cache_dir: Path,
        transform=None,
        image_size: int = 224,
    ):
        self.df = labels_df.reset_index(drop=True)
        self.cache = Path(image_cache_dir)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.transform = transform
        self.image_size = image_size
        self._encoders = self._build_encoders()

    @staticmethod
    def _build_encoders() -> dict[str, dict[str, int]]:
        out: dict[str, dict[str, int]] = {}
        for name, defn in HEAD_DEFINITIONS.items():
            if defn["kind"] == "categorical":
                out[name] = {c: i for i, c in enumerate(defn["classes"])}
        return out

    def __len__(self) -> int:
        return len(self.df)

    def _cached_image(self, image_id: str, r2_key: str) -> Image.Image:
        path = self.cache / f"{image_id}.webp"
        if not path.exists():
            raw = r2_client.download_bytes(r2_key)
            with open(path, "wb") as f:
                f.write(raw)
        return Image.open(path).convert("RGB")

    def _encode(self, label: dict) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        primary = label.get("primary_person") or {}
        for name, defn in HEAD_DEFINITIONS.items():
            if defn["kind"] == "categorical":
                if name in {"medium"}:
                    val = label.get(name)
                else:
                    val = primary.get(name)
                if val is None or val == "not_visible":
                    out[name] = np.array(-100, dtype=np.int64)
                else:
                    out[name] = np.array(self._encoders[name].get(val, -100), dtype=np.int64)
            else:
                if name in {"romantic_contact", "suggestive_pose", "block"}:
                    val = label.get(name)
                else:
                    val = primary.get(name)
                if val is None:
                    out[name] = np.array(-1.0, dtype=np.float32)
                else:
                    out[name] = np.array(1.0 if bool(val) else 0.0, dtype=np.float32)
        return out

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img = self._cached_image(row["image_id"], row["r2_key"])
        if self.transform is not None:
            img = self.transform(img)
        try:
            label = json.loads(row["corrected_label_json"]) if row.get("corrected_label_json") else json.loads(row["label_json"])
        except (TypeError, KeyError, json.JSONDecodeError):
            label = {}
        targets = self._encode(label)
        return img, targets


def load_labels_for_training(
    parquet_path: Path,
    human_review_path: Optional[Path] = None,
    prefer_human: bool = True,
) -> pd.DataFrame:
    df = pd.read_parquet(parquet_path)
    df = df[df["labeler"].isin({"nim", "gemini"})]
    df = df.sort_values("labeled_at").drop_duplicates(subset=["image_id"], keep="last")

    if human_review_path and human_review_path.exists() and prefer_human:
        hr = pd.read_parquet(human_review_path)
        hr = hr[hr["decision"].isin({"accept", "correct"})].copy()
        hr["corrected_label_json"] = hr["corrected_label_json"].fillna(hr["ai_label_json"])
        df = df.set_index("image_id")
        for _, row in hr.iterrows():
            iid = row["image_id"]
            if iid in df.index:
                df.at[iid, "label_json"] = row["corrected_label_json"]
                df.at[iid, "corrected_label_json"] = row["corrected_label_json"]
        df = df.reset_index()
    log.info("training labels: %d rows", len(df))
    return df


def stratified_split(
    df: pd.DataFrame,
    val_frac: float = 0.1,
    test_frac: float = 0.05,
    seed: int = 17,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Block-aware split: maintains acceptable/not-acceptable balance in each split."""
    rng = np.random.default_rng(seed)
    df = df.copy()
    df["__block"] = df["block"].fillna(False).astype(bool)
    parts: list[tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]] = []
    for _, grp in df.groupby("__block"):
        idx = np.arange(len(grp))
        rng.shuffle(idx)
        n_val = int(len(idx) * val_frac)
        n_test = int(len(idx) * test_frac)
        test = grp.iloc[idx[:n_test]]
        val = grp.iloc[idx[n_test : n_test + n_val]]
        train = grp.iloc[idx[n_test + n_val :]]
        parts.append((train, val, test))
    train = pd.concat([p[0] for p in parts]).sample(frac=1, random_state=seed).reset_index(drop=True)
    val = pd.concat([p[1] for p in parts]).sample(frac=1, random_state=seed).reset_index(drop=True)
    test = pd.concat([p[2] for p in parts]).sample(frac=1, random_state=seed).reset_index(drop=True)
    log.info("split: train=%d val=%d test=%d", len(train), len(val), len(test))
    return train.drop(columns="__block"), val.drop(columns="__block"), test.drop(columns="__block")
