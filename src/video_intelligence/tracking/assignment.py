"""Deterministic maximum-cardinality, maximum-IoU one-to-one assignment."""

from __future__ import annotations

import math
import time
from collections.abc import Sequence

MAX_ASSIGNMENT_SIZE = 256


def _check_deadline(deadline: float | None) -> None:
    if deadline is not None and time.perf_counter() >= deadline:
        raise TimeoutError("educational tracking exceeded its cooperative CPU deadline")


def maximum_iou_assignment(
    scores: Sequence[Sequence[float | None]], *, deadline: float | None = None,
) -> tuple[tuple[int, int], ...]:
    """Match admissible pairs globally; None forbids a pair before optimization.

    Cardinality has priority over summed IoU. Dummy columns allow every row to
    remain unmatched. Ties follow stable row/column iteration order. Rectangular
    Hungarian uses O(rows * (rows + columns)) storage and O(rows**2 *
    (rows + columns)) work. No SciPy dependency is required.
    """

    if isinstance(scores, (str, bytes)) or not isinstance(scores, Sequence):
        raise ValueError("scores must be a bounded rectangular sequence")
    rows = len(scores)
    if rows > MAX_ASSIGNMENT_SIZE:
        raise ValueError("assignment exceeds row cap")
    if deadline is not None and (
        isinstance(deadline, bool) or not isinstance(deadline, (int, float))
        or not math.isfinite(deadline)
    ):
        raise ValueError("deadline must be a finite clock value")
    _check_deadline(deadline)
    if not rows:
        return ()
    for row in scores:
        if isinstance(row, (str, bytes)) or not isinstance(row, Sequence):
            raise ValueError("scores must be a rectangular sequence")
    columns = len(scores[0])
    if columns > MAX_ASSIGNMENT_SIZE or any(len(row) != columns for row in scores):
        raise ValueError("assignment is ragged or exceeds column cap")
    for row in scores:
        for value in row:
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or not 0 <= value <= 1
            ):
                raise ValueError("IoU must be finite in [0,1], or None for a forbidden pair")
    if not columns:
        return ()
    # One extra match outweighs every possible difference in total IoU.
    bonus = rows + 1
    costs = [
        [bonus if value is None else -bonus - value for value in row] + [0.0] * rows
        for row in scores
    ]
    width = columns + rows
    u, v = [0.0] * (rows + 1), [0.0] * (width + 1)
    matched, previous = [0] * (width + 1), [0] * (width + 1)
    for row_number in range(1, rows + 1):
        _check_deadline(deadline)
        matched[0] = row_number
        current = 0
        distance, used = [math.inf] * (width + 1), [False] * (width + 1)
        while True:
            _check_deadline(deadline)
            used[current] = True
            active_row = matched[current]
            delta, next_column = math.inf, 0
            for column in range(1, width + 1):
                if used[column]:
                    continue
                reduced = costs[active_row - 1][column - 1] - u[active_row] - v[column]
                if reduced < distance[column]:
                    distance[column], previous[column] = reduced, current
                if distance[column] < delta:
                    delta, next_column = distance[column], column
            for column in range(width + 1):
                if used[column]:
                    u[matched[column]] += delta
                    v[column] -= delta
                else:
                    distance[column] -= delta
            current = next_column
            if matched[current] == 0:
                break
        while current:
            predecessor = previous[current]
            matched[current] = matched[predecessor]
            current = predecessor
    _check_deadline(deadline)
    return tuple(sorted(
        (matched[column] - 1, column - 1)
        for column in range(1, columns + 1)
        if matched[column] and scores[matched[column] - 1][column - 1] is not None
    ))
