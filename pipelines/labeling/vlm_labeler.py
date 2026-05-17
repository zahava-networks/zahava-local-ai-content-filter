"""VLM labelers: NVIDIA NIM (primary) and Google Gemini (secondary).

Both consume the same halachic prompt (prompts/labeler_prompt.txt) and return
JSON conforming to schema.ImageLabel. The deterministic block_rule.reconcile()
overrides the VLM's self-reported `block` with the rule's verdict.

Refusal handling:
  - Safety filter refusal → record as flagged_for_review, do not crash
  - Disagreement between NIM and Gemini on the same image → also flagged
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import random
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import os

import pandas as pd
import requests
from PIL import Image
from pydantic import ValidationError
from tqdm import tqdm

from ..common import get_logger, load_config, load_env, manifests_dir, require_env
from ..collection import r2_client
from .block_rule import reconcile
from .labels_store import LabelRecord, append, known_ids
from .rate_limiter import RateLimiter
from .schema import ImageLabel

log = get_logger(__name__)

PROMPT_PATH = Path(__file__).resolve().parent.parent.parent / "prompts" / "labeler_prompt.txt"


def _prompt() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8")


def _b64(raw: bytes) -> str:
    return base64.b64encode(raw).decode("ascii")


def _strip_json(text: str) -> str:
    """VLMs sometimes wrap JSON in code fences. Strip them."""
    t = text.strip()
    m = re.search(r"\{.*\}", t, re.DOTALL)
    if m:
        return m.group(0)
    return t


def _parse_label(raw_text: str) -> Optional[ImageLabel]:
    try:
        data = json.loads(_strip_json(raw_text))
    except json.JSONDecodeError:
        return None
    try:
        return ImageLabel.model_validate(data)
    except ValidationError as e:
        log.warning("schema validation failed: %s", e)
        return None


class _NIMClient:
    def __init__(self) -> None:
        cfg = load_config()["labeling"]["nim"]
        self.model = require_env("NIM_MODEL")
        self.api_key = require_env("NIM_API_KEY")
        self.base_url = require_env("NIM_BASE_URL")
        self.timeout = cfg["timeout_s"]
        self.max_retries = cfg["max_retries"]
        self.limiter = RateLimiter(rpm=cfg["rpm"])

    def label(self, img_bytes: bytes) -> tuple[Optional[ImageLabel], str]:
        b64 = _b64(img_bytes)
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": _prompt()},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/webp;base64,{b64}"},
                        },
                    ],
                }
            ],
            "max_tokens": 1024,
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
        }
        last_err = ""
        for attempt in range(self.max_retries):
            self.limiter.acquire()
            try:
                r = requests.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    timeout=self.timeout,
                )
                if r.status_code == 429:
                    time.sleep(2 ** attempt + random.random())
                    last_err = "429"
                    continue
                if r.status_code in (400, 422):
                    body = r.text[:300]
                    if "safety" in body.lower() or "policy" in body.lower():
                        return None, f"safety_refusal:{body[:80]}"
                r.raise_for_status()
                data = r.json()
                content = data["choices"][0]["message"]["content"]
                lbl = _parse_label(content)
                if lbl is None:
                    last_err = "parse_failed"
                    continue
                return lbl, "ok"
            except requests.RequestException as e:
                last_err = str(e)
                time.sleep(2 ** attempt + random.random())
        return None, f"failed_after_retries:{last_err}"


def gemini_available() -> bool:
    load_env()
    return bool(os.environ.get("GEMINI_API_KEY"))


class _GeminiClient:
    def __init__(self) -> None:
        cfg = load_config()["labeling"]["gemini"]
        self.model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash-lite")
        self.api_key = require_env("GEMINI_API_KEY")
        self.timeout = cfg["timeout_s"]
        self.max_retries = cfg["max_retries"]
        self.limiter = RateLimiter(rpm=cfg["rpm"])

    def label(self, img_bytes: bytes) -> tuple[Optional[ImageLabel], str]:
        from google import genai
        from google.genai import types as gtypes

        client = genai.Client(api_key=self.api_key)
        last_err = ""
        for attempt in range(self.max_retries):
            self.limiter.acquire()
            try:
                resp = client.models.generate_content(
                    model=self.model,
                    contents=[
                        gtypes.Part.from_text(_prompt()),
                        gtypes.Part.from_bytes(data=img_bytes, mime_type="image/webp"),
                    ],
                    config=gtypes.GenerateContentConfig(
                        temperature=0.0,
                        max_output_tokens=1024,
                        response_mime_type="application/json",
                    ),
                )
                if resp.prompt_feedback and getattr(resp.prompt_feedback, "block_reason", None):
                    return None, f"safety_refusal:{resp.prompt_feedback.block_reason}"
                text = resp.text or ""
                lbl = _parse_label(text)
                if lbl is None:
                    last_err = "parse_failed"
                    continue
                return lbl, "ok"
            except Exception as e:
                msg = str(e)
                if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                    time.sleep(2 ** attempt + random.random())
                    last_err = "429"
                    continue
                if "SAFETY" in msg.upper() or "BLOCKED" in msg.upper():
                    return None, f"safety_refusal:{msg[:80]}"
                last_err = msg
                time.sleep(2 ** attempt + random.random())
        return None, f"failed_after_retries:{last_err}"


def _record(image_id: str, r2_key: str, labeler: str, label: ImageLabel, flagged: bool, reason: str | None):
    return LabelRecord(
        image_id=image_id,
        r2_key=r2_key,
        labeler=labeler,
        labeled_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        label_json=label.model_dump_json(),
        block=label.block,
        confidence=label.confidence,
        violations_json=json.dumps(label.violations),
        flagged_for_review=flagged,
        review_reason=reason,
    )


def run(
    manifest_path: str | None = None,
    skip_already_nsfw: bool = True,
    round_name: str = "vlm_round_1",
    use_gemini_qa_rate: Optional[float] = None,
) -> None:
    cfg = load_config()["labeling"]
    if manifest_path is None:
        manifest_path = str(manifests_dir() / "collection_deduped.parquet")
    manifest_df = pd.read_parquet(manifest_path)

    if skip_already_nsfw:
        from .labels_store import path_for_round

        nsfw_path = path_for_round("nsfw_oracle")
        if nsfw_path.exists():
            blocked = set()
            threshold_skip = cfg["nsfw_oracle"]["threshold_skip_vlm"]
            with open(nsfw_path, "r", encoding="utf-8") as f:
                for line in f:
                    rec = json.loads(line)
                    lbl = json.loads(rec["label_json"])
                    if lbl.get("_nsfw_score", 0) >= threshold_skip:
                        blocked.add(rec["image_id"])
            log.info("skipping %d images already known-nsfw", len(blocked))
            manifest_df = manifest_df[~manifest_df["image_id"].isin(blocked)]

    done = known_ids(round_name)
    pending = manifest_df[~manifest_df["image_id"].isin(done)]
    log.info("vlm labeler: %d pending of %d", len(pending), len(manifest_df))

    qa_rate = use_gemini_qa_rate if use_gemini_qa_rate is not None else cfg["vlm_agreement_check_rate"]
    nim = _NIMClient()
    gemini = _GeminiClient() if gemini_available() else None
    if gemini is None:
        log.info("GEMINI_API_KEY not set — running NIM-only; safety refusals route to human review")

    pbar = tqdm(total=len(pending), desc=round_name)
    for _, row in pending.iterrows():
        try:
            raw = r2_client.download_bytes(row["r2_key"])
        except Exception as e:
            log.warning("download %s: %s", row["r2_key"], e)
            pbar.update(1)
            continue

        nim_label, nim_status = nim.label(raw)
        if nim_label is not None:
            nim_label = reconcile(nim_label)
            flagged = False
            reason = None

            if gemini is not None and random.random() < qa_rate:
                gemini_label, _ = gemini.label(raw)
                if gemini_label is not None:
                    gemini_label = reconcile(gemini_label)
                    if gemini_label.block != nim_label.block:
                        flagged = True
                        reason = "nim_gemini_disagreement"
                    append(round_name, _record(row["image_id"], row["r2_key"], "gemini", gemini_label, flagged, reason))

            append(round_name, _record(row["image_id"], row["r2_key"], "nim", nim_label, flagged, reason))
        elif nim_status.startswith("safety_refusal"):
            gemini_label = None
            gstatus = "gemini_unavailable"
            if gemini is not None:
                gemini_label, gstatus = gemini.label(raw)
            if gemini_label is not None:
                gemini_label = reconcile(gemini_label)
                append(round_name, _record(row["image_id"], row["r2_key"], "gemini", gemini_label, True, "nim_refused"))
            else:
                append(
                    round_name,
                    LabelRecord(
                        image_id=row["image_id"],
                        r2_key=row["r2_key"],
                        labeler="none",
                        labeled_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                        label_json="{}",
                        block=True,  # conservative default — "when in doubt, block"
                        confidence=0.0,
                        violations_json="[]",
                        flagged_for_review=True,
                        review_reason=f"nim_refused_no_fallback:{nim_status}|{gstatus}",
                    ),
                )
        else:
            log.warning("nim failed on %s: %s", row["image_id"], nim_status)
            append(
                round_name,
                LabelRecord(
                    image_id=row["image_id"],
                    r2_key=row["r2_key"],
                    labeler="none",
                    labeled_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    label_json="{}",
                    block=True,
                    confidence=0.0,
                    violations_json="[]",
                    flagged_for_review=True,
                    review_reason=f"nim_failed:{nim_status}",
                ),
            )
        pbar.update(1)
    pbar.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default=None)
    p.add_argument("--round", default="vlm_round_1")
    p.add_argument("--qa-rate", type=float, default=None)
    a = p.parse_args()
    run(a.manifest, round_name=a.round, use_gemini_qa_rate=a.qa_rate)
