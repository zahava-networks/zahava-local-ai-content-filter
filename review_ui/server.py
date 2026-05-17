"""Local review UI server.

Run with:  python -m review_ui.server [--port 8765] [--populate N]
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from cachetools import LRUCache

from pipelines.collection import r2_client
from pipelines.common import get_logger, load_config

from . import db, queue_manager

log = get_logger(__name__)
STATIC_DIR = Path(__file__).resolve().parent / "static"
app = FastAPI(title="Tzniut Review UI", version="0.1.0")

_image_cache: LRUCache = LRUCache(maxsize=256)


class Decision(BaseModel):
    image_id: str
    decision: str
    corrected_label: Optional[dict] = None
    notes: Optional[str] = None


@app.get("/")
def root():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/next")
def api_next():
    item = db.next_pending()
    if not item:
        return JSONResponse({"empty": True})
    try:
        ai_label = json.loads(item["ai_label_json"])
    except Exception:
        ai_label = {}
    return {
        "image_id": item["image_id"],
        "r2_key": item["r2_key"],
        "image_url": f"/api/image/{item['image_id']}",
        "ai_label": ai_label,
        "ai_confidence": item["ai_confidence"],
        "flag_reason": item.get("flag_reason"),
    }


@app.get("/api/image/{image_id}")
def api_image(image_id: str):
    cached = _image_cache.get(image_id)
    if cached is not None:
        body, ctype = cached
        return Response(content=body, media_type=ctype)
    with db.connect() as c:
        row = c.execute("SELECT r2_key FROM queue WHERE image_id = ?", (image_id,)).fetchone()
    if not row:
        raise HTTPException(404, "unknown image_id")
    try:
        body = r2_client.download_bytes(row["r2_key"])
    except Exception as e:
        raise HTTPException(502, f"fetch failed: {e}")
    ctype = "image/webp" if row["r2_key"].endswith(".webp") else "image/jpeg"
    _image_cache[image_id] = (body, ctype)
    return Response(content=body, media_type=ctype)


@app.post("/api/decision")
def api_decision(d: Decision):
    if d.decision not in {"accept", "correct", "skip", "bad_image"}:
        raise HTTPException(400, f"unknown decision: {d.decision}")
    db.record_decision(
        d.image_id,
        d.decision,
        d.corrected_label,
        d.notes,
        datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    return {"ok": True, "stats": db.stats()}


@app.get("/api/stats")
def api_stats():
    return db.stats()


@app.post("/api/push")
def api_push():
    n = queue_manager.push_to_hf()
    return {"pushed": n}


@app.post("/api/populate")
def api_populate(limit: Optional[int] = None):
    n = queue_manager.populate(limit=limit)
    return {"new_in_queue": n, "stats": db.stats()}


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=load_config()["review_ui"]["port"])
    parser.add_argument("--populate", type=int, default=None, help="pre-populate N items from HF labels")
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    if args.populate is not None:
        queue_manager.populate(limit=args.populate)

    log.info("Review UI at http://%s:%d", args.host, args.port)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
