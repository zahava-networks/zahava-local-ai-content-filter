"""Collect from Wikimedia Commons via the MediaWiki API.

License: CC BY-SA / CC0 / various — stored per-image in the manifest.
No API rate limit beyond reasonable use; we still throttle.
"""
from __future__ import annotations

import argparse
import time

import requests
from tqdm import tqdm

from ..common import get_logger, load_config, read_state, write_state
from .base import IngestResult, fetch_url, ingest
from . import manifest as mf

log = get_logger(__name__)

SOURCE = "wikimedia"

CATEGORY_QUERIES = [
    "Category:Women_by_clothing",
    "Category:Men_by_clothing",
    "Category:Beachwear",
    "Category:Swimwear",
    "Category:Wedding_dresses",
    "Category:Modest_clothing",
    "Category:Hijab",
    "Category:Burqas",
    "Category:Athletic_wear",
    "Category:Yoga_pants",
    "Category:Mini_skirts",
    "Category:Long_dresses",
    "Category:Long_skirts",
    "Category:Tznius",
    "Category:Orthodox_Jewish_women",
    "Category:Hasidic_men",
    "Category:Fashion_photography",
    "Category:Catwalk",
    "Category:People_walking",
    "Category:Cartoons",
]

WIKI_API = "https://commons.wikimedia.org/w/api.php"


def _list_category(category: str, cm_continue: str | None = None) -> dict:
    params = {
        "action": "query",
        "format": "json",
        "list": "categorymembers",
        "cmtitle": category,
        "cmtype": "file",
        "cmlimit": 50,
    }
    if cm_continue:
        params["cmcontinue"] = cm_continue
    r = requests.get(WIKI_API, params=params, timeout=30, headers={"User-Agent": "TzniutClassifier/0.1"})
    r.raise_for_status()
    return r.json()


def _image_url(title: str) -> str | None:
    r = requests.get(
        WIKI_API,
        params={
            "action": "query",
            "titles": title,
            "prop": "imageinfo",
            "iiprop": "url|extmetadata",
            "format": "json",
        },
        timeout=30,
        headers={"User-Agent": "TzniutClassifier/0.1"},
    )
    r.raise_for_status()
    pages = r.json().get("query", {}).get("pages", {})
    for _, p in pages.items():
        infos = p.get("imageinfo", [])
        if infos:
            return infos[0].get("url")
    return None


def run(max_to_collect: int | None = None) -> None:
    cap = max_to_collect or load_config()["collection"]["per_source_caps"]["wikimedia"]
    already = mf.count(SOURCE)
    if already >= cap:
        log.info("wikimedia already at %d (cap %d) — skipping", already, cap)
        return

    state = read_state(f"collect_{SOURCE}")
    cursor = state.get("cursor", {})  # category -> cmcontinue
    remaining = cap - already
    pbar = tqdm(total=remaining, desc=SOURCE)
    n_ingested = 0

    for cat in CATEGORY_QUERIES:
        if n_ingested >= remaining:
            break
        cm_continue = cursor.get(cat)
        while True:
            try:
                data = _list_category(cat, cm_continue)
            except Exception as e:
                log.warning("list %s: %s", cat, e)
                break
            for item in data.get("query", {}).get("categorymembers", []):
                if n_ingested >= remaining:
                    break
                title = item["title"]
                if not any(title.lower().endswith(ext) for ext in (".jpg", ".jpeg", ".png", ".webp")):
                    continue
                try:
                    url = _image_url(title)
                    if not url:
                        continue
                    raw = fetch_url(url)
                    res: IngestResult = ingest(SOURCE, title, raw, url, "Wikimedia (varies; see file page)")
                    if not res.skipped:
                        n_ingested += 1
                        pbar.update(1)
                except Exception as e:
                    log.warning("ingest %s: %s", title, e)
                time.sleep(0.2)
            cm_continue = data.get("continue", {}).get("cmcontinue")
            cursor[cat] = cm_continue
            write_state(f"collect_{SOURCE}", {"cursor": cursor})
            if not cm_continue or n_ingested >= remaining:
                break
    pbar.close()
    log.info("wikimedia: ingested %d (manifest now %d)", n_ingested, mf.count(SOURCE))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--max", type=int, default=None)
    args = p.parse_args()
    run(args.max)
