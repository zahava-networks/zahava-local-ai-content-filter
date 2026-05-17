"""Shared config loader, logger, and small utilities."""
from __future__ import annotations

import json
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "config" / "config.yaml"
THRESHOLDS_PATH = REPO_ROOT / "config" / "thresholds.yaml"
ENV_PATH = REPO_ROOT / ".env"


@lru_cache(maxsize=1)
def load_env() -> None:
    if ENV_PATH.exists():
        load_dotenv(ENV_PATH)
    else:
        load_dotenv()


@lru_cache(maxsize=1)
def load_config() -> dict[str, Any]:
    load_env()
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@lru_cache(maxsize=1)
def load_thresholds() -> dict[str, Any]:
    with open(THRESHOLDS_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def require_env(name: str) -> str:
    load_env()
    val = os.environ.get(name)
    if not val:
        raise RuntimeError(
            f"Missing environment variable {name}. "
            f"Set it in {ENV_PATH} (copy from config/.env.example)."
        )
    return val


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    h = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s", "%H:%M:%S")
    h.setFormatter(fmt)
    logger.addHandler(h)
    logger.propagate = False
    return logger


def state_dir() -> Path:
    p = REPO_ROOT / load_config()["paths"]["state_dir"]
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_state(name: str) -> dict[str, Any]:
    f = state_dir() / f"{name}.json"
    if not f.exists():
        return {}
    return json.loads(f.read_text())


def write_state(name: str, state: dict[str, Any]) -> None:
    f = state_dir() / f"{name}.json"
    tmp = f.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True))
    tmp.replace(f)


def manifests_dir() -> Path:
    p = REPO_ROOT / load_config()["paths"]["manifests_dir"]
    p.mkdir(parents=True, exist_ok=True)
    return p
