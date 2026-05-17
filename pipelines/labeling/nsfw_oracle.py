"""Falconsai/nsfw_image_detection — deterministic nudity oracle.

Apache 2.0, ViT-base, binary classifier. Used as a fast pre-filter before VLM
labeling: images scoring above `threshold_skip_vlm` skip VLM entirely (already
known to be NOT_ACCEPTABLE for the nudity reason).
"""
from __future__ import annotations

import argparse
import io
from datetime import datetime, timezone
from typing import Iterable

import pandas as pd
from PIL import Image
from tqdm import tqdm

from ..common import get_logger, load_config, manifests_dir, require_env
from ..collection import r2_client
from .labels_store import LabelRecord, append, known_ids

log = get_logger(__name__)
ROUND = "nsfw_oracle"
LABELER = "falconsai"


def _load_model():
    from transformers import AutoModelForImageClassification, AutoProcessor
    import torch

    cfg = load_config()["labeling"]["nsfw_oracle"]
    name = cfg["model"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("loading %s on %s", name, device)
    model = AutoModelForImageClassification.from_pretrained(name).to(device).eval()
    processor = AutoProcessor.from_pretrained(name)
    return model, processor, device


def _iter_images(manifest_df: pd.DataFrame, batch_size: int):
    done = known_ids(ROUND)
    pending = manifest_df[~manifest_df["image_id"].isin(done)].copy()
    log.info("nsfw oracle: %d pending of %d total", len(pending), len(manifest_df))
    batch: list[tuple[str, str, Image.Image]] = []
    for _, row in pending.iterrows():
        try:
            raw = r2_client.download_bytes(row["r2_key"])
            img = Image.open(io.BytesIO(raw)).convert("RGB")
            batch.append((row["image_id"], row["r2_key"], img))
        except Exception as e:
            log.warning("download %s: %s", row["r2_key"], e)
            continue
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def run(manifest_path: str | None = None) -> None:
    import torch

    cfg = load_config()["labeling"]["nsfw_oracle"]
    batch_size = cfg["batch_size"]
    threshold_block = cfg["threshold_block"]
    if manifest_path is None:
        manifest_path = str(manifests_dir() / "collection_deduped.parquet")
    manifest_df = pd.read_parquet(manifest_path)

    model, processor, device = _load_model()

    pbar = tqdm(total=len(manifest_df) - len(known_ids(ROUND)), desc="nsfw_oracle")
    nsfw_label_idx = None
    if hasattr(model.config, "id2label"):
        for idx, name in model.config.id2label.items():
            if name.lower() in {"nsfw", "porn", "explicit", "sexy"}:
                nsfw_label_idx = idx
                break
    if nsfw_label_idx is None:
        nsfw_label_idx = 1  # convention for binary models

    for batch in _iter_images(manifest_df, batch_size):
        ids = [b[0] for b in batch]
        keys = [b[1] for b in batch]
        imgs = [b[2] for b in batch]
        inputs = processor(images=imgs, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        for iid, key, p in zip(ids, keys, probs):
            score = float(p[nsfw_label_idx])
            block = score >= threshold_block
            label_json = {
                "person_present": True,
                "person_count": 1,
                "medium": "unknown",
                "primary_person": None,
                "additional_people": [],
                "romantic_contact": False,
                "suggestive_pose": False,
                "block": block,
                "violations": ["nudity_full"] if block else [],
                "confidence": score if block else 1.0 - score,
                "reasoning": f"nsfw_score={score:.3f}, threshold={threshold_block}",
                "_nsfw_score": score,
            }
            append(
                ROUND,
                LabelRecord(
                    image_id=iid,
                    r2_key=key,
                    labeler=LABELER,
                    labeled_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    label_json=__import__("json").dumps(label_json),
                    block=block,
                    confidence=float(label_json["confidence"]),
                    violations_json=__import__("json").dumps(label_json["violations"]),
                    flagged_for_review=False,
                ),
            )
        pbar.update(len(batch))
    pbar.close()
    log.info("nsfw_oracle done")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default=None)
    args = p.parse_args()
    run(args.manifest)
