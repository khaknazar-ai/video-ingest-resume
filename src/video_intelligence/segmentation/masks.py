"""Canonical, bounded lossless binary mask transport; no model dependency."""

from __future__ import annotations

import base64
import hashlib

import numpy as np

ENCODING = "binary_packbits_base64.v1"
MAX_PIXELS = 8_294_400


def pack_mask(mask):
    if (not isinstance(mask, np.ndarray) or mask.dtype != np.bool_ or mask.ndim != 2
            or not 0 < mask.size <= MAX_PIXELS or min(mask.shape) <= 0):
        raise ValueError("mask must be a bounded full-resolution boolean array")
    binary = mask.astype(np.uint8)
    return {"encoding": ENCODING, "height": mask.shape[0], "width": mask.shape[1],
            "area": int(binary.sum()), "sha256": hashlib.sha256(binary.tobytes()).hexdigest(),
            "data": base64.b64encode(
                np.packbits(binary.reshape(-1), bitorder="big"),
            ).decode("ascii")}


def unpack_mask(value, *, width=None, height=None):
    if not isinstance(value, dict) or set(value) != {
        "encoding", "height", "width", "area", "sha256", "data",
    } or value["encoding"] != ENCODING:
        raise ValueError("invalid binary mask schema")
    w, h = value["width"], value["height"]
    if (any(isinstance(x, bool) or not isinstance(x, int) or x <= 0 for x in (w, h))
            or w * h > MAX_PIXELS or width is not None and w != width
            or height is not None and h != height):
        raise ValueError("mask dimensions differ or exceed pixel cap")
    count = (w * h + 7) // 8
    if not isinstance(value["data"], str) or len(value["data"]) != 4 * ((count + 2) // 3):
        raise ValueError("mask packed length exceeds declared dimensions")
    packed = base64.b64decode(value["data"], validate=True)
    if len(packed) != count:
        raise ValueError("mask packed size differs")
    binary = np.unpackbits(np.frombuffer(packed, dtype=np.uint8), bitorder="big")
    if binary[w * h:].any():
        raise ValueError("mask padding bits must be zero")
    binary = binary[:w * h].reshape(h, w)
    if (isinstance(value["area"], bool) or not isinstance(value["area"], int)
            or value["area"] != int(binary.sum())
            or hashlib.sha256(binary.tobytes()).hexdigest() != value["sha256"]):
        raise ValueError("mask area or binary hash differs")
    return binary.astype(bool)
