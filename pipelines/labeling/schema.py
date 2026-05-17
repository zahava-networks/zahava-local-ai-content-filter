"""Label schema + Pydantic validators + canonical violation tags."""
from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class Gender(str, Enum):
    FEMALE = "female"
    MALE = "male"
    UNKNOWN = "unknown"


class AgeGroup(str, Enum):
    ADULT = "adult"
    CHILD = "child"
    UNKNOWN = "unknown"


class SleeveLength(str, Enum):
    NONE = "none"
    SHORT = "short"
    ELBOW = "elbow"
    THREE_QUARTER = "three_quarter"
    LONG = "long"
    NOT_VISIBLE = "not_visible"


class Neckline(str, Enum):
    MODEST = "modest"
    CLEAVAGE_VISIBLE = "cleavage_visible"
    NO_TOP = "no_top"
    NOT_VISIBLE = "not_visible"


class LowerGarment(str, Enum):
    SKIRT = "skirt"
    PANTS = "pants"
    SHORTS = "shorts"
    SWIMWEAR = "swimwear"
    UNDERWEAR = "underwear"
    NONE = "none"
    NOT_VISIBLE = "not_visible"


class LowerLength(str, Enum):
    ABOVE_KNEE = "above_knee"
    AT_KNEE = "at_knee"
    BELOW_KNEE = "below_knee"
    FULL = "full"
    NOT_VISIBLE = "not_visible"


class Fit(str, Enum):
    LOOSE = "loose"
    FITTED = "fitted"
    TIGHT = "tight"
    NOT_VISIBLE = "not_visible"


class Nudity(str, Enum):
    NONE = "none"
    PARTIAL = "partial"
    FULL = "full"


class Medium(str, Enum):
    PHOTO = "photo"
    CARTOON = "cartoon"
    ILLUSTRATION = "illustration"
    ANIME = "anime"
    DRAWING = "drawing"
    MIXED = "mixed"


VIOLATION_TAGS = frozenset(
    {
        "nudity_full",
        "nudity_partial",
        "shirtless_male",
        "romantic_contact",
        "suggestive_pose",
        "female_sleeve_too_short",
        "female_cleavage_visible",
        "female_pants",
        "female_shorts",
        "female_swimwear",
        "female_underwear_visible",
        "female_no_lower_garment",
        "female_skirt_too_short",
        "female_tight_fit",
    }
)


class PersonAttributes(BaseModel):
    gender: Gender
    age_group: AgeGroup
    sleeve_length: SleeveLength
    neckline: Neckline
    lower_garment: LowerGarment
    lower_length: LowerLength
    fit: Fit
    visible_nudity: Nudity
    shirtless_male: bool


class ImageLabel(BaseModel):
    person_present: bool
    person_count: int = Field(ge=0)
    medium: Medium
    primary_person: Optional[PersonAttributes]
    additional_people: list[PersonAttributes] = Field(default_factory=list)
    romantic_contact: bool
    suggestive_pose: bool
    block: bool
    violations: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = Field(max_length=400)

    @field_validator("violations")
    @classmethod
    def _violations_known(cls, v: list[str]) -> list[str]:
        unknown = [x for x in v if x not in VIOLATION_TAGS]
        if unknown:
            raise ValueError(f"unknown violation tags: {unknown}")
        return v

    @field_validator("person_count")
    @classmethod
    def _count_matches_present(cls, v: int) -> int:
        return max(0, v)
