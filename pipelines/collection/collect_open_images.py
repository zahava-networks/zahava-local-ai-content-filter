"""Collect from Open Images V7 — Person subset, CC BY 2.0.

Uses FiftyOne to query Open Images directly via Google Cloud without
downloading the whole 9M-image dataset. We pull image URLs and stream
them into R2.
"""
from __future__ import annotations

import argparse

from tqdm import tqdm

from ..common import get_logger, load_config, read_state, write_state
from . import manifest as mf
from .base import IngestResult, fetch_url, ingest

log = get_logger(__name__)

SOURCE = "open_images_v7"
LICENSE = "CC BY 2.0"


def run(max_to_collect: int | None = None) -> None:
    import fiftyone.zoo as foz

    cap = max_to_collect or load_config()["collection"]["per_source_caps"]["open_images"]
    already = mf.count(SOURCE)
    if already >= cap:
        log.info("open_images already at %d (cap %d) — skipping", already, cap)
        return

    state = read_state(f"collect_{SOURCE}")
    seed = state.get("seed", 7)
    start_idx = state.get("next_idx", 0)
    remaining = cap - already

    log.info("loading %d Open Images V7 'Person' samples (start_idx=%d)", remaining, start_idx)
    dataset = foz.load_zoo_dataset(
        "open-images-v7",
        split="train",
        classes=["Person"],
        max_samples=remaining + start_idx,
        seed=seed,
        only_matching=True,
        shuffle=True,
        label_types=["classifications"],
    )

    n_ingested = 0
    pbar = tqdm(total=remaining, desc="open_images")
    for i, sample in enumerate(dataset.skip(start_idx)):
        try:
            with open(sample.filepath, "rb") as f:
                raw = f.read()
            url = sample.get_field("open_images_id") or sample.filepath
            res: IngestResult = ingest(SOURCE, str(sample.id), raw, str(url), LICENSE)
            if not res.skipped:
                n_ingested += 1
                pbar.update(1)
        except Exception as e:
            log.warning("skip %s: %s", sample.id, e)
        if n_ingested % 200 == 0 and n_ingested > 0:
            write_state(f"collect_{SOURCE}", {"seed": seed, "next_idx": start_idx + i + 1})
        if n_ingested >= remaining:
            break
    write_state(f"collect_{SOURCE}", {"seed": seed, "next_idx": start_idx + n_ingested})
    pbar.close()
    log.info("open_images: ingested %d (manifest now %d)", n_ingested, mf.count(SOURCE))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--max", type=int, default=None)
    args = p.parse_args()
    run(args.max)
