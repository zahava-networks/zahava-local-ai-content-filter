"""Connectivity + import smoke test.

Run after filling in `.env` to confirm everything is wired up:
  python scripts/smoke_test.py
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def _ok(msg: str) -> None:
    print(f"  \033[32m✓\033[0m {msg}")


def _err(msg: str) -> None:
    print(f"  \033[31m✗\033[0m {msg}")


def _section(name: str) -> None:
    print(f"\n\033[1m{name}\033[0m")


def check_imports() -> bool:
    _section("imports")
    modules = [
        "pipelines.common",
        "pipelines.collection.r2_client",
        "pipelines.collection.manifest",
        "pipelines.collection.image_utils",
        "pipelines.collection.dedup",
        "pipelines.collection.base",
        "pipelines.collection.collect_open_images",
        "pipelines.collection.collect_huggingface",
        "pipelines.collection.collect_unsplash",
        "pipelines.collection.collect_pexels",
        "pipelines.collection.collect_wikimedia",
        "pipelines.collection.run_all",
        "pipelines.labeling.schema",
        "pipelines.labeling.block_rule",
        "pipelines.labeling.rate_limiter",
        "pipelines.labeling.labels_store",
        "pipelines.labeling.nsfw_oracle",
        "pipelines.labeling.vlm_labeler",
        "pipelines.labeling.run",
        "pipelines.training.heads",
        "pipelines.training.dataset",
        "pipelines.training.model",
        "pipelines.training.train",
        "pipelines.training.threshold_tuner",
        "pipelines.training.calibrator",
        "pipelines.export.calibration_set",
        "pipelines.export.export_tflite",
        "pipelines.export.export_coreml",
        "pipelines.export.export_full",
        "pipelines.eval.benchmark",
        "pipelines.eval.html_report",
        "pipelines.eval.latency_on_device",
        "pipelines.active_learning.uncertainty_sampling",
        "pipelines.active_learning.round_runner",
        "review_ui.db",
        "review_ui.queue_manager",
        "review_ui.server",
    ]
    all_ok = True
    for m in modules:
        try:
            importlib.import_module(m)
            _ok(m)
        except Exception as e:
            _err(f"{m}: {type(e).__name__}: {e}")
            all_ok = False
    return all_ok


def check_env() -> bool:
    _section(".env")
    from pipelines.common import load_env

    load_env()
    import os

    required = [
        "R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET", "R2_ENDPOINT",
        "HF_TOKEN", "HF_DATASET_REPO",
        "NIM_API_KEY", "NIM_MODEL",
        "GEMINI_API_KEY", "GEMINI_MODEL",
    ]
    ok = True
    for k in required:
        if os.environ.get(k):
            _ok(k)
        else:
            _err(f"{k} missing")
            ok = False
    return ok


def check_r2() -> bool:
    _section("Cloudflare R2")
    try:
        from pipelines.collection import r2_client
        b = r2_client.bucket()
        keys = list(r2_client.list_keys("", limit=1))
        _ok(f"bucket={b} (listing ok)")
        return True
    except Exception as e:
        _err(f"{type(e).__name__}: {e}")
        return False


def check_hf() -> bool:
    _section("HuggingFace")
    try:
        from huggingface_hub import HfApi
        from pipelines.common import require_env
        api = HfApi(token=require_env("HF_TOKEN"))
        user = api.whoami()
        _ok(f"whoami: {user.get('name','?')}")
        return True
    except Exception as e:
        _err(f"{type(e).__name__}: {e}")
        return False


def check_nim() -> bool:
    _section("NVIDIA NIM")
    try:
        import requests
        from pipelines.common import require_env

        url = require_env("NIM_BASE_URL").rstrip("/") + "/models"
        r = requests.get(url, headers={"Authorization": f"Bearer {require_env('NIM_API_KEY')}"}, timeout=10)
        r.raise_for_status()
        _ok(f"models endpoint reachable")
        return True
    except Exception as e:
        _err(f"{type(e).__name__}: {e}")
        return False


def check_gemini() -> bool:
    _section("Google Gemini")
    try:
        from google import genai
        from pipelines.common import require_env

        client = genai.Client(api_key=require_env("GEMINI_API_KEY"))
        models = client.models.list()
        names = [m.name for m in models]
        _ok(f"models listed: {len(names)} available")
        return True
    except Exception as e:
        _err(f"{type(e).__name__}: {e}")
        return False


def run() -> int:
    print("Tzniut classifier — smoke test\n")
    ok_imports = check_imports()
    ok_env = check_env()
    if ok_env:
        check_r2()
        check_hf()
        check_nim()
        check_gemini()
    print()
    if not ok_imports:
        print("\033[31mFAIL\033[0m: imports broken — fix before doing anything else.")
        return 1
    if not ok_env:
        print("\033[33mWARN\033[0m: .env incomplete — set keys before running pipelines.")
        return 2
    print("\033[32mOK\033[0m")
    return 0


if __name__ == "__main__":
    sys.exit(run())
