"""Fail-closed display policy until canonical transform mapping is implemented."""

from __future__ import annotations

import math
import struct
from fractions import Fraction

_IDENTITY_MATRIX = (65536, 0, 0, 0, 65536, 0, 0, 0, 1073741824)


class UnsupportedVideoTransformError(ValueError):
    """The source requires a display transform this decoder contract cannot represent."""


def require_identity_display(
    *, rotation: object = None, matrix: object = None, sample_aspect_ratio: object = None
) -> None:
    """Reject rotation, reflection, skew and non-square display pixels explicitly."""

    if rotation is not None:
        try:
            angle = float(rotation)
        except (TypeError, ValueError) as error:
            raise UnsupportedVideoTransformError("invalid display rotation") from error
        if not math.isfinite(angle) or not math.isclose(angle % 360, 0, abs_tol=1e-6):
            raise UnsupportedVideoTransformError("nonidentity display rotation is unsupported")
    if matrix is not None:
        try:
            if isinstance(matrix, bytes):
                values = struct.unpack("=9i", matrix)
            elif isinstance(matrix, str):
                values = tuple(
                    int(value)
                    for row in matrix.strip().splitlines()
                    for value in row.split(":", maxsplit=1)[1].split()
                )
            else:
                raise ValueError("unknown display matrix representation")
        except (IndexError, ValueError, struct.error) as error:
            raise UnsupportedVideoTransformError("invalid display matrix") from error
        if values != _IDENTITY_MATRIX:
            raise UnsupportedVideoTransformError("nonidentity display matrix is unsupported")
    if sample_aspect_ratio not in (None, "N/A", "0:1", 0):
        try:
            ratio = Fraction(str(sample_aspect_ratio).replace(":", "/"))
        except (ValueError, ZeroDivisionError) as error:
            raise UnsupportedVideoTransformError("invalid display sample aspect ratio") from error
        if ratio != 1:
            raise UnsupportedVideoTransformError("nonsquare display pixels are unsupported")
