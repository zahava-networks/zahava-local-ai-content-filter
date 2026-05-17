"""Deterministic halachic block rule.

Single source of truth — used by:
  - the VLM labeler (post-check on the model's `block` field)
  - the trained model's inference path (over per-attribute predictions)
  - the human review UI (to surface the block reason)
"""
from __future__ import annotations

from .schema import (
    Fit,
    Gender,
    ImageLabel,
    LowerGarment,
    LowerLength,
    Neckline,
    Nudity,
    PersonAttributes,
    SleeveLength,
)


def _evaluate_person(p: PersonAttributes) -> list[str]:
    violations: list[str] = []

    if p.visible_nudity == Nudity.FULL:
        violations.append("nudity_full")
    elif p.visible_nudity == Nudity.PARTIAL:
        violations.append("nudity_partial")

    if p.shirtless_male:
        violations.append("shirtless_male")

    treat_as_female = p.gender == Gender.FEMALE or p.gender == Gender.UNKNOWN

    if treat_as_female:
        if p.sleeve_length in {SleeveLength.NONE, SleeveLength.SHORT, SleeveLength.ELBOW}:
            violations.append("female_sleeve_too_short")
        if p.neckline == Neckline.CLEAVAGE_VISIBLE:
            violations.append("female_cleavage_visible")

        if p.lower_garment == LowerGarment.PANTS:
            violations.append("female_pants")
        elif p.lower_garment == LowerGarment.SHORTS:
            violations.append("female_shorts")
        elif p.lower_garment == LowerGarment.SWIMWEAR:
            violations.append("female_swimwear")
        elif p.lower_garment == LowerGarment.UNDERWEAR:
            violations.append("female_underwear_visible")
        elif p.lower_garment == LowerGarment.NONE:
            violations.append("female_no_lower_garment")

        if p.lower_garment in {LowerGarment.SKIRT, LowerGarment.PANTS} and p.lower_length in {
            LowerLength.ABOVE_KNEE,
            LowerLength.AT_KNEE,
        }:
            violations.append("female_skirt_too_short")

        if p.fit == Fit.TIGHT:
            violations.append("female_tight_fit")

    return violations


def evaluate(label: ImageLabel) -> tuple[bool, list[str]]:
    """Return (block, violations) computed from per-person attributes.

    The pipeline calls this AFTER getting the VLM's labels. If the VLM's
    `block` disagrees with our deterministic rule, we trust the rule.
    """
    violations: list[str] = []

    if label.primary_person is not None:
        violations.extend(_evaluate_person(label.primary_person))
    for p in label.additional_people:
        violations.extend(_evaluate_person(p))

    if label.romantic_contact:
        violations.append("romantic_contact")
    if label.suggestive_pose:
        violations.append("suggestive_pose")

    seen: set[str] = set()
    deduped = [v for v in violations if not (v in seen or seen.add(v))]
    return (len(deduped) > 0, deduped)


def reconcile(label: ImageLabel) -> ImageLabel:
    """Overwrite `block` and `violations` on the label using the deterministic rule.

    Use when the VLM's self-reported block is unreliable. The rule's verdict wins.
    """
    block, violations = evaluate(label)
    return label.model_copy(update={"block": block, "violations": violations})
