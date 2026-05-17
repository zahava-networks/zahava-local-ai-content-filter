"""Multi-task head schema. Single source of truth for what the model predicts.

Each head is either:
  - "categorical": softmax over N named classes
  - "binary": sigmoid (single logit, BCE loss)
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HeadSpec:
    name: str
    kind: str           # "categorical" | "binary"
    classes: tuple[str, ...]  # for categorical; for binary, len==1 (label name)
    loss_weight: float  # passed from config.yaml at training time


HEAD_DEFINITIONS: dict[str, dict] = {
    "gender": {
        "kind": "categorical",
        "classes": ("female", "male", "unknown"),
    },
    "age_group": {
        "kind": "categorical",
        "classes": ("adult", "child", "unknown"),
    },
    "sleeve_length": {
        "kind": "categorical",
        "classes": ("none", "short", "elbow", "three_quarter", "long", "not_visible"),
    },
    "neckline": {
        "kind": "categorical",
        "classes": ("modest", "cleavage_visible", "no_top", "not_visible"),
    },
    "lower_garment": {
        "kind": "categorical",
        "classes": ("skirt", "pants", "shorts", "swimwear", "underwear", "none", "not_visible"),
    },
    "lower_length": {
        "kind": "categorical",
        "classes": ("above_knee", "at_knee", "below_knee", "full", "not_visible"),
    },
    "fit": {
        "kind": "categorical",
        "classes": ("loose", "fitted", "tight", "not_visible"),
    },
    "visible_nudity": {
        "kind": "categorical",
        "classes": ("none", "partial", "full"),
    },
    "shirtless_male": {"kind": "binary", "classes": ("shirtless_male",)},
    "romantic_contact": {"kind": "binary", "classes": ("romantic_contact",)},
    "suggestive_pose": {"kind": "binary", "classes": ("suggestive_pose",)},
    "medium": {
        "kind": "categorical",
        "classes": ("photo", "cartoon", "illustration", "anime", "drawing", "mixed"),
    },
    "block": {"kind": "binary", "classes": ("block",)},
}


def head_specs(weights: dict[str, float]) -> dict[str, HeadSpec]:
    out: dict[str, HeadSpec] = {}
    for name, defn in HEAD_DEFINITIONS.items():
        out[name] = HeadSpec(
            name=name,
            kind=defn["kind"],
            classes=tuple(defn["classes"]),
            loss_weight=float(weights.get(name, 1.0)),
        )
    return out
