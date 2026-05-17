"""Image processing helpers: load, validate, resize, hash, encode."""
from __future__ import annotations

import hashlib
import io
from typing import Optional

import imagehash
from PIL import Image, ImageOps, UnidentifiedImageError

from ..common import load_config


MIN_SIDE_PX = 128


class BadImage(Exception):
    """Raised when an image is unreadable, too small, or otherwise unusable."""


def open_validated(raw: bytes) -> Image.Image:
    """Decode bytes → RGB PIL image, validating size and integrity."""
    try:
        img = Image.open(io.BytesIO(raw))
        img.load()
    except (UnidentifiedImageError, OSError) as e:
        raise BadImage(f"could not decode: {e}") from e
    img = ImageOps.exif_transpose(img)
    if img.mode != "RGB":
        img = img.convert("RGB")
    if min(img.size) < MIN_SIDE_PX:
        raise BadImage(f"too small: {img.size}")
    return img


def resize_for_storage(img: Image.Image, target: Optional[int] = None) -> Image.Image:
    if target is None:
        target = load_config()["image"]["collection_resolution"]
    w, h = img.size
    if min(w, h) <= target:
        return img
    if w < h:
        new_w = target
        new_h = int(round(h * (target / w)))
    else:
        new_h = target
        new_w = int(round(w * (target / h)))
    return img.resize((new_w, new_h), Image.LANCZOS)


def encode_webp(img: Image.Image, quality: Optional[int] = None) -> bytes:
    if quality is None:
        quality = load_config()["image"]["jpeg_quality"]
    buf = io.BytesIO()
    img.save(buf, format="WEBP", quality=quality, method=6)
    return buf.getvalue()


def perceptual_hash(img: Image.Image) -> str:
    return str(imagehash.phash(img))


def content_id(raw: bytes, prefix: str = "img") -> str:
    """Deterministic short id from content hash."""
    h = hashlib.sha256(raw).hexdigest()[:20]
    return f"{prefix}_{h}"
