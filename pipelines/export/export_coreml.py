"""PyTorch → CoreML via coremltools.

INT8 weight quantization with palettization. iOS 16+ deployment target.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ..common import REPO_ROOT, get_logger, load_config
from ..training.heads import head_specs
from ..training.model import TzniutMultiTask
from ..training.train import _load_checkpoint
from .export_tflite import _ExportWrapper

log = get_logger(__name__)


def export(checkpoint_path: str | None = None, output: str | None = None) -> Path:
    cfg = load_config()
    image_size = cfg["training"]["input_size"]
    heads = head_specs(cfg["training"]["loss"]["head_weights"])
    model = TzniutMultiTask(cfg["training"]["backbone"], heads)
    ck = Path(checkpoint_path) if checkpoint_path else REPO_ROOT / "models" / "checkpoints" / "best.pt"
    _load_checkpoint(ck, model)
    model.eval()
    wrapped = _ExportWrapper(model).eval()

    try:
        import coremltools as ct
    except ImportError as e:
        raise RuntimeError("coremltools not installed. `pip install coremltools`") from e

    example = torch.randn(1, 3, image_size, image_size)
    traced = torch.jit.trace(wrapped, example)

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    scale = 1.0 / (np.array(std) * 255.0)
    bias = (-np.array(mean) / np.array(std)).tolist()

    mlmodel = ct.convert(
        traced,
        inputs=[ct.ImageType(
            name="image",
            shape=(1, 3, image_size, image_size),
            scale=float(scale[0]),
            bias=bias,
            color_layout=ct.colorlayout.RGB,
        )],
        outputs=[ct.TensorType(name="logits")],
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS16,
        compute_precision=ct.precision.FLOAT16,
    )

    from coremltools.optimize.coreml import (
        OptimizationConfig,
        OpLinearQuantizerConfig,
        linear_quantize_weights,
    )
    op_cfg = OpLinearQuantizerConfig(mode="linear_symmetric", weight_threshold=512)
    opt_cfg = OptimizationConfig(global_config=op_cfg)
    mlmodel_q = linear_quantize_weights(mlmodel, config=opt_cfg)

    out = Path(output) if output else REPO_ROOT / "models" / "tzniut.mlpackage"
    out.parent.mkdir(parents=True, exist_ok=True)
    mlmodel_q.save(str(out))

    head_meta = {
        "head_order": wrapped.head_order,
        "head_sizes": wrapped.head_sizes,
        "input_size": image_size,
    }
    (out.parent / "tzniut_mlpackage_heads.json").write_text(json.dumps(head_meta, indent=2))
    log.info("CoreML → %s", out)
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--output", default=None)
    a = p.parse_args()
    export(a.checkpoint, a.output)
