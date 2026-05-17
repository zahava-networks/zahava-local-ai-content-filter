"""Collect from Unsplash via the official API.

Uses search-based queries to target clothing diversity (modest, athletic,
swimwear, formal, street, etc.). The Unsplash API license requires:
  - attribution to photographers (stored in manifest)
  - hotlinking permitted but local cache for ML training also allowed
"""
from __future__ import annotations

import argparse
import time

import requests
from tqdm import tqdm

from ..common import get_logger, load_config, read_state, require_env, write_state
from .base import IngestResult, fetch_url, ingest
from . import manifest as mf

log = get_logger(__name__)

SOURCE = "unsplash"
LICENSE = "Unsplash License (commercial-friendly with attribution)"

SEARCH_QUERIES = [
    # modest / acceptable coverage
    "modest fashion", "tzniut fashion", "hijab outfit", "abaya",
    "long sleeve maxi dress", "long sleeve dress", "long skirt outfit",
    "midi skirt outfit", "Mormon dress", "Amish woman",
    "business attire woman", "professional woman", "office outfit woman",
    "Orthodox Jewish family", "sheitel wig",
    # non-modest / violation coverage
    "bikini", "swimsuit", "beach photo", "yoga pants", "leggings outfit",
    "bodycon dress", "mini skirt", "crop top", "tank top",
    "athletic wear woman", "gym outfit", "tennis player",
    "shirtless man", "shirtless beach", "boxer underwear",
    # mixed / neutral
    "office", "wedding photo", "graduation", "family photo",
    "street fashion", "street photography people",
    # cultural diversity
    "Indian saree", "Korean traditional dress", "African dress",
    "Pakistani shalwar kameez", "Vietnamese ao dai",
]


def _search(query: str, page: int, per_page: int, token: str) -> list[dict]:
    r = requests.get(
        "https://api.unsplash.com/search/photos",
        params={"query": query, "page": page, "per_page": per_page, "orientation": "portrait"},
        headers={"Authorization": f"Client-ID {token}"},
        timeout=30,
    )
    if r.status_code == 403:
        raise RuntimeError("Unsplash rate limit hit (403). Wait an hour and resume.")
    r.raise_for_status()
    return r.json().get("results", [])


def run(max_to_collect: int | None = None) -> None:
    cap = max_to_collect or load_config()["collection"]["per_source_caps"]["unsplash"]
    already = mf.count(SOURCE)
    if already >= cap:
        log.info("unsplash already at %d (cap %d) — skipping", already, cap)
        return
    token = require_env("UNSPLASH_ACCESS_KEY")

    state = read_state(f"collect_{SOURCE}")
    cursor = state.get("cursor", {})  # query -> page already done
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
                    url = item["urls"]["regular"]
                    raw = fetch_url(url)
                    sid = item["id"]
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
    log.info("unsplash: ingested %d (manifest now %d)", n_ingested, mf.count(SOURCE))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--max", type=int, default=None)
    args = p.parse_args()
    run(args.max)
