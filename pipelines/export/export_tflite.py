"""PyTorch → TFLite via ai_edge_torch (Google's official converter, 2024+).

The model gets INT8-quantized using a representative dataset built by
calibration_set.py. Target: <5.5 MB per config.yaml.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from ..common import REPO_ROOT, get_logger, load_config
from ..training.heads import head_specs
from ..training.model import TzniutMultiTask
from ..training.train import _load_checkpoint

log = get_logger(__name__)


class _ExportWrapper(torch.nn.Module):
    """Wraps the multi-task model so its forward returns a single concatenated tensor.

    TFLite handles single-tensor outputs more cleanly than dicts. We concat
    head outputs in a fixed order and the inference runtime splits them.
    """

    def __init__(self, model: TzniutMultiTask):
        super().__init__()
        self.model = model
        self.head_order = sorted(model.heads.keys())
        self.head_sizes = [
            (1 if model.head_specs[n].kind == "binary" else len(model.head_specs[n].classes))
            for n in self.head_order
        ]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        parts = []
        for n in self.head_order:
            t = out[n]
            if t.ndim == 1:
                t = t.unsqueeze(-1)
            parts.append(t)
        return torch.cat(parts, dim=-1)


def export(checkpoint_path: str | None = None, output: str | None = None) -> Path:
    cfg = load_config()
    image_size = cfg["training"]["input_size"]
    heads = head_specs(cfg["training"]["loss"]["head_weights"])
    model = TzniutMultiTask(cfg["training"]["backbone"], heads)
    ck = Path(checkpoint_path) if checkpoint_path else REPO_ROOT / "models" / "checkpoints" / "best.pt"
    _load_checkpoint(ck, model)
    model.eval()
    wrapped = _ExportWrapper(model).eval()

    calib_path = REPO_ROOT / "models" / "calibration_set.npy"
    if not calib_path.exists():
        from .calibration_set import build
        build()
    calib = np.load(calib_path)
    log.info("calibration set shape: %s", calib.shape)

    try:
        import ai_edge_torch
        from ai_edge_torch.quantize import pt2e_quantizer
    except ImportError as e:
        raise RuntimeError(
            "ai_edge_torch not installed. In Colab: `pip install ai-edge-torch`"
        ) from e

    sample_input = (torch.from_numpy(calib[:1]),)

    def rep_dataset():
        bs = 1
        for i in range(0, min(len(calib), 256), bs):
            yield [torch.from_numpy(calib[i : i + bs])]

    quantizer = pt2e_quantizer.PT2EQuantizer().set_global(
        pt2e_quantizer.get_symmetric_quantization_config()
    )
    quant_config = ai_edge_torch.quantize.quant_config.QuantConfig(
        pt2e_quantizer=quantizer,
        calibration_data=rep_dataset(),
    )
    edge_model = ai_edge_torch.convert(wrapped, sample_input, quant_config=quant_config)
    out = Path(output) if output else REPO_ROOT / "models" / "tzniut.tflite"
    out.parent.mkdir(parents=True, exist_ok=True)
    edge_model.export(str(out))

    size_kb = out.stat().st_size / 1024
    log.info("tflite → %s (%.1f KB)", out, size_kb)
    budget = cfg["export"]["tflite"]["target_max_kb"]
    if size_kb > budget:
        log.warning("tflite exceeded budget (%.1f KB > %d KB) — consider pruning", size_kb, budget)

    head_meta = {
        "head_order": wrapped.head_order,
        "head_sizes": wrapped.head_sizes,
        "input_size": image_size,
        "preprocessing": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
    }
    import json as _json
    (out.with_suffix(".json")).write_text(_json.dumps(head_meta, indent=2))
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--output", default=None)
    a = p.parse_args()
    export(a.checkpoint, a.output)
