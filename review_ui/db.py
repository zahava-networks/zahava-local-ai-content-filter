"""Local SQLite store for the review UI queue and decisions."""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

from pipelines.common import load_config


_SCHEMA = """
CREATE TABLE IF NOT EXISTS queue (
    image_id TEXT PRIMARY KEY,
    r2_key TEXT NOT NULL,
    ai_label_json TEXT NOT NULL,
    ai_confidence REAL NOT NULL,
    flag_reason TEXT,
    enqueued_at TEXT NOT NULL,
    reviewed_at TEXT,
    decision TEXT,
    corrected_label_json TEXT,
    notes TEXT,
    pushed_to_hf INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_queue_unreviewed ON queue (reviewed_at) WHERE reviewed_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_queue_pending_push ON queue (pushed_to_hf) WHERE pushed_to_hf = 0;
"""


def db_path() -> Path:
    p = Path(load_config()["paths"]["review_db"])
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(db_path(), isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.executescript("PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL;")
    conn.executescript(_SCHEMA)
    try:
        yield conn
    finally:
        conn.close()


def enqueue_many(rows: list[dict]) -> int:
    if not rows:
        return 0
    with connect() as c:
        sql = (
            "INSERT OR IGNORE INTO queue "
            "(image_id, r2_key, ai_label_json, ai_confidence, flag_reason, enqueued_at) "
            "VALUES (?, ?, ?, ?, ?, ?)"
        )
        cur = c.executemany(
            sql,
            [
                (
                    r["image_id"],
                    r["r2_key"],
                    r["ai_label_json"],
                    r["ai_confidence"],
                    r.get("flag_reason"),
                    r["enqueued_at"],
                )
                for r in rows
            ],
        )
        return cur.rowcount


def next_pending() -> Optional[dict]:
    with connect() as c:
        row = c.execute(
            "SELECT * FROM queue WHERE reviewed_at IS NULL ORDER BY enqueued_at LIMIT 1"
        ).fetchone()
        return dict(row) if row else None


def record_decision(
    image_id: str,
    decision: str,
    corrected_label: Optional[dict],
    notes: Optional[str],
    reviewed_at: str,
) -> None:
    with connect() as c:
        c.execute(
            "UPDATE queue SET decision = ?, corrected_label_json = ?, notes = ?, reviewed_at = ? "
            "WHERE image_id = ?",
            (
                decision,
                json.dumps(corrected_label) if corrected_label else None,
                notes,
                reviewed_at,
                image_id,
            ),
        )


def stats() -> dict:
    with connect() as c:
        total = c.execute("SELECT COUNT(*) FROM queue").fetchone()[0]
        reviewed = c.execute(
            "SELECT COUNT(*) FROM queue WHERE reviewed_at IS NOT NULL"
        ).fetchone()[0]
        pending = total - reviewed
        agreed = c.execute(
            "SELECT COUNT(*) FROM queue WHERE decision = 'accept'"
        ).fetchone()[0]
        return {"total": total, "reviewed": reviewed, "pending": pending, "accepted": agreed}


def export_for_hf() -> list[dict]:
    with connect() as c:
        rows = c.execute(
            "SELECT image_id, r2_key, ai_label_json, decision, corrected_label_json, notes, reviewed_at "
            "FROM queue WHERE reviewed_at IS NOT NULL AND pushed_to_hf = 0"
        ).fetchall()
        return [dict(r) for r in rows]


def mark_pushed(image_ids: list[str]) -> None:
    if not image_ids:
        return
    with connect() as c:
        c.executemany(
            "UPDATE queue SET pushed_to_hf = 1 WHERE image_id = ?",
            [(i,) for i in image_ids],
        )
