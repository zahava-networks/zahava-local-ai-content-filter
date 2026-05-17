"""Connectivity + import smoke test.

Run after filling in `.env` to confirm everything is wired up:
  python scripts/smoke_test.py
"""
from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def _ok(msg: str) -> None:
    print(f"  \033[32m✓\033[0m {msg}")


def _err(msg: str) -> None:
    print(f"  \033[31m✗\033[0m {msg}")


def _info(msg: str) -> None:
    print(f"  \033[2m·\033[0m {msg}")


def _section(name: str) -> None:
    print(f"\n\033[1m{name}\033[0m")


REQUIRED_ENV = ["HF_TOKEN", "HF_DATASET_REPO", "HF_MODEL_REPO", "NIM_API_KEY", "NIM_BASE_URL", "NIM_MODEL"]
OPTIONAL_ENV = ["GEMINI_API_KEY", "UNSPLASH_ACCESS_KEY", "PEXELS_API_KEY", "KAGGLE_KEY", "LIGHTNING_USER_KEY"]


def check_imports() -> bool:
    _section("imports")
    modules = [
        "pipelines.common",
        "pipelines.collection.r2_client",
        "pipelines.collection.manifest",
        "pipelines.collection.image_utils",
        "pipelines.collection.dedup",
        "pipelines.collection.base",
        "pipelines.collection.run_all",
        "pipelines.labeling.schema",
        "pipelines.labeling.block_rule",
        "pipelines.labeling.rate_limiter",
        "pipelines.labeling.labels_store",
        "pipelines.training.heads",
        "review_ui.db",
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
    ok = True
    for k in REQUIRED_ENV:
        if os.environ.get(k):
            _ok(k)
        else:
            _err(f"{k} (REQUIRED) missing")
            ok = False
    for k in OPTIONAL_ENV:
        if os.environ.get(k):
            _ok(f"{k} (optional, configured)")
        else:
            _info(f"{k} (optional, blank)")
    return ok


def check_hf_dataset() -> bool:
    _section("HuggingFace (storage + auth)")
    try:
        from huggingface_hub import HfApi
        from pipelines.common import require_env

        api = HfApi(token=require_env("HF_TOKEN"))
        user = api.whoami()
        _ok(f"whoami: {user.get('name','?')}")
        repo = require_env("HF_DATASET_REPO")
        try:
            api.repo_info(repo, repo_type="dataset")
            _ok(f"dataset repo exists: {repo}")
        except Exception:
            api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
            _ok(f"dataset repo created (private): {repo}")
        model_repo = require_env("HF_MODEL_REPO")
        try:
            api.repo_info(model_repo)
            _ok(f"model repo exists: {model_repo}")
        except Exception:
            api.create_repo(model_repo, private=True, exist_ok=True)
            _ok(f"model repo created (private): {model_repo}")
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
        _ok("models endpoint reachable")
        return True
    except Exception as e:
        _err(f"{type(e).__name__}: {e}")
        return False


def check_gemini_optional() -> None:
    _section("Gemini (optional)")
    from pipelines.common import load_env

    load_env()
    if not os.environ.get("GEMINI_API_KEY"):
        _info("GEMINI_API_KEY not set — NIM-only mode; safety refusals route to human review")
        return
    try:
        from google import genai

        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        list(client.models.list())
        _ok("Gemini reachable")
    except Exception as e:
        _err(f"Gemini configured but unreachable: {type(e).__name__}: {e}")


def run() -> int:
    print("Tzniut classifier — smoke test\n")
    ok_imports = check_imports()
    ok_env = check_env()
    if ok_env:
        check_hf_dataset()
        check_nim()
        check_gemini_optional()
    print()
    if not ok_imports:
        print("\033[31mFAIL\033[0m: imports broken")
        return 1
    if not ok_env:
        print("\033[33mWARN\033[0m: required .env values missing")
        return 2
    print("\033[32mOK\033[0m")
    return 0


if __name__ == "__main__":
    sys.exit(run())
