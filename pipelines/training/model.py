"""Multi-task model: timm backbone + per-head linear heads.

Why timm: MobileNetV4 is in timm, EfficientNet-Lite0 is in timm, and timm models
export cleanly via ai_edge_torch (PyTorch→LiteRT) and coremltools (PyTorch→CoreML).
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from .heads import HeadSpec


class TzniutMultiTask(nn.Module):
    def __init__(self, backbone_name: str, heads: Dict[str, HeadSpec]):
        super().__init__()
        import timm

        try:
            self.backbone = timm.create_model(
                backbone_name, pretrained=True, num_classes=0, global_pool="avg"
            )
        except Exception:
            # Fallback for offline or missing weights — random init still usable
            self.backbone = timm.create_model(
                backbone_name, pretrained=False, num_classes=0, global_pool="avg"
            )
        feat_dim = self.backbone.num_features

        self.heads = nn.ModuleDict()
        self.head_specs = heads
        for name, spec in heads.items():
            n_out = 1 if spec.kind == "binary" else len(spec.classes)
            self.heads[name] = nn.Linear(feat_dim, n_out)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        feat = self.backbone(x)
        return {name: head(feat) for name, head in self.heads.items()}

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_loss_fn(heads: Dict[str, HeadSpec], pos_weight_block: float = 3.0):
    """Returns a callable (logits_dict, targets_dict) -> (total_loss, per_head_losses)."""
    ce = nn.CrossEntropyLoss(ignore_index=-100, reduction="mean")
    bce_block = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_block]))
    bce_other = nn.BCEWithLogitsLoss()

    def _fn(logits: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]):
        device = next(iter(logits.values())).device
        total = torch.tensor(0.0, device=device)
        per_head: Dict[str, torch.Tensor] = {}
        for name, spec in heads.items():
            t = targets[name].to(device)
            l = logits[name].to(device)
            if spec.kind == "categorical":
                if (t != -100).any():
                    loss = ce(l, t.long())
                else:
                    loss = torch.tensor(0.0, device=device)
            else:
                mask = t >= 0
                if mask.any():
                    bce = bce_block if name == "block" else bce_other
                    bce.to(device)
                    loss = bce(l[mask].squeeze(-1), t[mask].float())
                else:
                    loss = torch.tensor(0.0, device=device)
            per_head[name] = loss.detach()
            total = total + spec.loss_weight * loss
        return total, per_head

    return _fn
