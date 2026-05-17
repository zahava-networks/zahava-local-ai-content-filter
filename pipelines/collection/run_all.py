"""Single entry to run every collector + dedup + merge.

Sources gracefully skip when their API key isn't configured (Unsplash, Pexels).
The no-key sources (Open Images, HF datasets, Wikimedia) always run.
"""
from __future__ import annotations

import argparse
import os

from ..common import get_logger, load_env
from . import dedup
from . import manifest as mf

log = get_logger(__name__)

NO_KEY_SOURCES = ("open_images", "deepfashion2", "fashionpedia", "modanet", "icartoonface", "wikimedia")
KEY_GATED = {"unsplash": "UNSPLASH_ACCESS_KEY", "pexels": "PEXELS_API_KEY"}


def run(only: list[str] | None = None, skip_dedup: bool = False) -> None:
    load_env()
    runners = {
        "open_images": lambda: __import__("pipelines.collection.collect_open_images", fromlist=["run"]).run(),
        "deepfashion2": lambda: __import__("pipelines.collection.collect_huggingface", fromlist=["run"]).run("deepfashion2"),
        "fashionpedia": lambda: __import__("pipelines.collection.collect_huggingface", fromlist=["run"]).run("fashionpedia"),
        "modanet": lambda: __import__("pipelines.collection.collect_huggingface", fromlist=["run"]).run("modanet"),
        "icartoonface": lambda: __import__("pipelines.collection.collect_huggingface", fromlist=["run"]).run("icartoonface"),
        "unsplash": lambda: __import__("pipelines.collection.collect_unsplash", fromlist=["run"]).run(),
        "pexels": lambda: __import__("pipelines.collection.collect_pexels", fromlist=["run"]).run(),
        "wikimedia": lambda: __import__("pipelines.collection.collect_wikimedia", fromlist=["run"]).run(),
    }
    chosen = only or list(runners.keys())
    chosen = [
        s for s in chosen
        if s in NO_KEY_SOURCES or os.environ.get(KEY_GATED.get(s, ""), "")
    ]
    log.info("running collectors: %s", chosen)
    for name in chosen:
        if name not in runners:
            log.warning("unknown source %s", name)
            continue
        log.info("====== %s ======", name)
        try:
            runners[name]()
        except Exception as e:
            log.exception("%s failed: %s — moving on", name, e)
    out = mf.merge_to_parquet("collection.parquet")
    log.info("merged manifest → %s", out)
    if not skip_dedup:
        deduped = dedup.apply_dedup(out)
        log.info("deduped manifest → %s", deduped)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--only", nargs="*")
    p.add_argument("--skip-dedup", action="store_true")
    a = p.parse_args()
    run(only=a.only, skip_dedup=a.skip_dedup)
