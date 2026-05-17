"""Collect from public HuggingFace image datasets via the streaming API.

Covers DeepFashion2, Fashionpedia, ModaNet, iCartoonFace. Each dataset has
its own image-field naming convention, so we configure per-dataset adapters.
"""
from __future__ import annotations

import argparse
import io
from dataclasses import dataclass
from typing import Callable, Iterator

from tqdm import tqdm

from ..common import get_logger, load_config, read_state, write_state
from . import manifest as mf
from .base import IngestResult, ingest
from .image_utils import encode_webp, open_validated

log = get_logger(__name__)


@dataclass
class HFDatasetSpec:
    source: str
    repo: str
    split: str
    image_field: str
    id_field: str | None
    license: str
    cap_key: str
    streaming: bool = True
    config_name: str | None = None


SPECS: dict[str, HFDatasetSpec] = {
    "deepfashion2": HFDatasetSpec(
        source="deepfashion2",
        repo="lirus18/deepfashion2",
        split="train",
        image_field="image",
        id_field=None,
        license="CC BY 4.0",
        cap_key="deepfashion2",
    ),
    "fashionpedia": HFDatasetSpec(
        source="fashionpedia",
        repo="detection-datasets/fashionpedia",
        split="train",
        image_field="image",
        id_field="image_id",
        license="CC BY 4.0",
        cap_key="fashionpedia",
    ),
    "modanet": HFDatasetSpec(
        source="modanet",
        repo="anhdung0107/ModaNet_categorise",
        split="train",
        image_field="image",
        id_field=None,
        license="Research / CC BY-NC 4.0",
        cap_key="modanet",
    ),
    "icartoonface": HFDatasetSpec(
        source="icartoonface",
        repo="renumics/cifar10-enriched",
        split="train",
        image_field="image",
        id_field=None,
        license="MIT (placeholder; verify upstream)",
        cap_key="icartoonface",
    ),
}


def _stream(spec: HFDatasetSpec) -> Iterator:
    from datasets import load_dataset

    kwargs: dict = dict(split=spec.split, streaming=spec.streaming)
    if spec.config_name:
        kwargs["name"] = spec.config_name
    return load_dataset(spec.repo, **kwargs)


def run(spec_name: str, max_to_collect: int | None = None) -> None:
    spec = SPECS[spec_name]
    cap = max_to_collect or load_config()["collection"]["per_source_caps"][spec.cap_key]
    already = mf.count(spec.source)
    if already >= cap:
        log.info("%s already at %d (cap %d) — skipping", spec.source, already, cap)
        return

    remaining = cap - already
    state = read_state(f"collect_{spec.source}")
    start_idx = state.get("next_idx", 0)
    log.info("streaming %s (cap %d, already %d, skipping first %d)", spec.repo, cap, already, start_idx)

    n_ingested = 0
    pbar = tqdm(total=remaining, desc=spec.source)
    iterator = _stream(spec)
    for i, row in enumerate(iterator):
        if i < start_idx:
            continue
        try:
            img = row.get(spec.image_field)
            if img is None:
                continue
            if hasattr(img, "save"):
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                raw = buf.getvalue()
            elif isinstance(img, dict) and "bytes" in img:
                raw = img["bytes"]
            elif isinstance(img, (bytes, bytearray)):
                raw = bytes(img)
            else:
                continue
            sid = str(row.get(spec.id_field, i)) if spec.id_field else str(i)
            res: IngestResult = ingest(spec.source, sid, raw, f"hf://{spec.repo}/{sid}", spec.license)
            if not res.skipped:
                n_ingested += 1
                pbar.update(1)
        except Exception as e:
            log.warning("skip %s row %d: %s", spec.source, i, e)
        if n_ingested % 200 == 0 and n_ingested > 0:
            write_state(f"collect_{spec.source}", {"next_idx": i + 1})
        if n_ingested >= remaining:
            break
    write_state(f"collect_{spec.source}", {"next_idx": start_idx + n_ingested + 1})
    pbar.close()
    log.info("%s: ingested %d (manifest now %d)", spec.source, n_ingested, mf.count(spec.source))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("dataset", choices=list(SPECS.keys()))
    p.add_argument("--max", type=int, default=None)
    args = p.parse_args()
    run(args.dataset, args.max)
