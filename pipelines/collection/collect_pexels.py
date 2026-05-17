"""Collect from Pexels via the official API.

Pexels License is essentially CC0 — free for commercial use without attribution
(attribution still appreciated, stored in manifest).
"""
from __future__ import annotations

import argparse
import time

import requests
from tqdm import tqdm

from ..common import get_logger, load_config, read_state, require_env, write_state
from .base import IngestResult, fetch_url, ingest
from . import manifest as mf
from .collect_unsplash import SEARCH_QUERIES  # reuse the same diverse query set

log = get_logger(__name__)

SOURCE = "pexels"
LICENSE = "Pexels License (CC0-equivalent)"


def _search(query: str, page: int, per_page: int, token: str) -> list[dict]:
    r = requests.get(
        "https://api.pexels.com/v1/search",
        params={"query": query, "page": page, "per_page": per_page},
        headers={"Authorization": token},
        timeout=30,
    )
    if r.status_code == 429:
        raise RuntimeError("Pexels rate limit hit. Wait and resume.")
    r.raise_for_status()
    return r.json().get("photos", [])


def run(max_to_collect: int | None = None) -> None:
    cap = max_to_collect or load_config()["collection"]["per_source_caps"]["pexels"]
    already = mf.count(SOURCE)
    if already >= cap:
        log.info("pexels already at %d (cap %d) — skipping", already, cap)
        return
    token = require_env("PEXELS_API_KEY")

    state = read_state(f"collect_{SOURCE}")
    cursor = state.get("cursor", {})
    remaining = cap - already
    pbar = tqdm(total=remaining, desc=SOURCE)
    n_ingested = 0

    for query in SEARCH_QUERIES:
        if n_ingested >= remaining:
            break
        page = cursor.get(query, 1)
        while page <= 50:
            try:
                results = _search(query, page, per_page=30, token=token)
            except Exception as e:
                log.warning("search %s p%d failed: %s", query, page, e)
                time.sleep(5)
                break
            if not results:
                break
            for item in results:
                if n_ingested >= remaining:
                    break
                try:
                    url = item["src"]["large"]
                    raw = fetch_url(url)
                    sid = str(item["id"])
                    res: IngestResult = ingest(SOURCE, sid, raw, url, LICENSE)
                    if not res.skipped:
                        n_ingested += 1
                        pbar.update(1)
                except Exception as e:
                    log.warning("ingest %s: %s", item.get("id"), e)
            page += 1
            cursor[query] = page
            write_state(f"collect_{SOURCE}", {"cursor": cursor})
            time.sleep(1)
    pbar.close()
    log.info("pexels: ingested %d (manifest now %d)", n_ingested, mf.count(SOURCE))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--max", type=int, default=None)
    args = p.parse_args()
    run(args.max)
