"""Export the full FP32 PyTorch checkpoint + head metadata for future fine-tuning."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from ..common import REPO_ROOT, get_logger, load_config
from ..training.heads import HEAD_DEFINITIONS

log = get_logger(__name__)


def export(checkpoint_path: str | None = None, output: str | None = None) -> Path:
    src = Path(checkpoint_path) if checkpoint_path else REPO_ROOT / "models" / "checkpoints" / "best.pt"
    out = Path(output) if output else REPO_ROOT / "models" / "tzniut_full.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, out)

    cfg = load_config()
    meta = {
        "backbone": cfg["training"]["backbone"],
        "input_size": cfg["training"]["input_size"],
        "heads": {
            name: {"kind": d["kind"], "classes": list(d["classes"])}
            for name, d in HEAD_DEFINITIONS.items()
        },
        "preprocessing": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "interpolation": "bilinear",
        },
    }
    (out.with_suffix(".json")).write_text(json.dumps(meta, indent=2))
    log.info("full model → %s", out)
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--output", default=None)
    a = p.parse_args()
    export(a.checkpoint, a.output)
