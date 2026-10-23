"""Whole-view translation (a camera scroll) between the boards before and after an action.

Some games scroll the view: everything on screen moves together while the level itself stays put. As a raw
change this reads as hundreds of changed cells and makes every remembered screen position wrong. The detector
compares the two boards under every translation of up to 48 cells (per-colour FFT correlation) and reports
one when it explains the change much better than no translation.

Validation (2026-10-13, thresholds fixed before the second test): all logged pilot runs, 303/306 true shifts
found with the exact vector and 0 false detections in 12,590 other changed board pairs; random play on every
level of the 25 public games, 57/60 camera moves found and 0 false detections in 40,648 changed pairs. The
before/after misses are 52-row jumps that leave 12 rows of overlap and small follow-pans; `detect_action`
covers most of them with the action's animation frames. It uses only the boards the harness already has.
"""
from __future__ import annotations

import os
from typing import Any, Sequence

import numpy as np

MAX_SHIFT = 48
MIN_AGREEMENT = 0.80   # share of overlapping cells that match after the shift
MIN_GAIN = 0.10        # ... and how much better that is than no shift
MIN_CHANGED = 256      # smallest logged true shift changed 628 cells
MIN_OVERLAP_SHARE = 0.25


def enabled() -> bool:
    return os.environ.get("ARC3_VIEW_SHIFT", "1").strip().lower() in {"1", "true", "yes", "on"}


def detect(before: Sequence[Sequence[int]], after: Sequence[Sequence[int]]) -> dict[str, Any] | None:
    """{"dy", "dx", "other_cells"} when the whole view moved, else None.

    dy > 0: content moved down (the camera moved up); dx > 0: content moved right. `other_cells` counts
    cells that still differ inside the overlap once the shift is removed.
    """
    a = np.asarray(before)
    b = np.asarray(after)
    if a.ndim != 2 or a.shape != b.shape or min(a.shape) < 16:
        return None
    if int((a != b).sum()) < MIN_CHANGED:
        return None
    rows, cols = a.shape
    size = 1 << int(np.ceil(np.log2(2 * max(rows, cols))))
    matches = np.zeros((size, size))
    for colour in np.union1d(np.unique(a), np.unique(b)):
        fa = np.zeros((size, size)); fa[:rows, :cols] = (a == colour)
        fb = np.zeros((size, size)); fb[:rows, :cols] = (b == colour)
        matches += np.real(np.fft.ifft2(np.fft.fft2(fb) * np.conj(np.fft.fft2(fa))))
    matches = np.rint(matches)
    offsets = np.array([i if i < size // 2 else i - size for i in range(size)])
    dy_grid, dx_grid = np.meshgrid(offsets, offsets, indexing="ij")
    overlap = np.clip(rows - np.abs(dy_grid), 0, None) * np.clip(cols - np.abs(dx_grid), 0, None)
    valid = ((np.abs(dy_grid) <= min(MAX_SHIFT, rows - 1)) & (np.abs(dx_grid) <= min(MAX_SHIFT, cols - 1))
             & (overlap >= MIN_OVERLAP_SHARE * rows * cols) & ~((dy_grid == 0) & (dx_grid == 0)))
    agreement = np.where(valid, matches / np.maximum(overlap, 1), -1.0)
    index = np.unravel_index(int(np.argmax(agreement)), agreement.shape)
    best = float(agreement[index])
    unshifted = float(matches[0, 0]) / float(rows * cols)
    if best < MIN_AGREEMENT or best - unshifted < MIN_GAIN:
        return None
    dy, dx = int(offsets[index[0]]), int(offsets[index[1]])
    return {"dy": dy, "dx": dx, "other_cells": int(overlap[index] - matches[index])}


def residual(before, after, dy: int, dx: int) -> int:
    """Cells that differ inside the overlap once a (dy, dx) shift is removed."""
    a = np.asarray(before)
    b = np.asarray(after)
    rows, cols = a.shape
    if abs(dy) >= rows or abs(dx) >= cols:
        return 0
    a_part = a[max(0, -dy):rows - max(0, dy), max(0, -dx):cols - max(0, dx)]
    b_part = b[max(0, dy):rows - max(0, -dy), max(0, dx):cols - max(0, -dx)]
    return int((a_part != b_part).sum())


def detect_action(before, after, frames: Sequence[Sequence[Sequence[int]]] = ()) -> dict[str, Any] | None:
    """The view shift of one action: the before/after boards first, else the sum over its animation.

    The before/after comparison misses large jumps (little overlap left, e.g. a 52-row camera move) and
    small follow-pans whose gain stays under the threshold. Frame to frame, both are ordinary scrolls, so
    the shifts between consecutive animation frames are summed instead, and the sum is kept only if it
    also explains the two boards. All logged runs: 412/412 true scrolls with the exact vector (before/after
    alone 403), 0 false detections in 16,336 actions without a scroll. Random play on the 25 public games:
    59/60 camera moves (before/after alone 57), 0 false detections in 40,648 changed pairs; without the
    check against the two boards, an sp80 animation that slides away and cuts back gave 4 false ones.
    """
    shift = detect(before, after)
    if shift or len(frames) < 2:
        return shift
    a = np.asarray(before)
    b = np.asarray(after)
    if a.ndim != 2 or a.shape != b.shape:
        return None
    changed = int((a != b).sum())
    if changed < MIN_CHANGED:
        return None  # a scroll changes many cells; an animation that ends where it began is not one
    dy = dx = 0
    previous = before
    for frame in frames:
        step = detect(previous, frame)
        if step:
            dy += step["dy"]
            dx += step["dx"]
        previous = frame
    if not dy and not dx:
        return None
    rows, cols = a.shape
    overlap = max(0, rows - abs(dy)) * max(0, cols - abs(dx))
    if not overlap:
        return None
    other = residual(before, after, dy, dx)
    # the summed shift must also explain the two boards: most of the overlap matches, and better than no shift
    if 1 - other / overlap < MIN_AGREEMENT or 1 - other / overlap <= 1 - changed / (rows * cols):
        return None
    return {"dy": dy, "dx": dx, "other_cells": other, "from_animation": True}


def describe(dy: int, dx: int, *, short: bool = False) -> str:
    """'24 rows down', '8 columns left', '30 rows up and 15 columns right' (short: '24 down')."""
    parts = []
    if dy:
        parts.append(f"{abs(dy)}{'' if short else ' rows'} {'down' if dy > 0 else 'up'}")
    if dx:
        parts.append(f"{abs(dx)}{'' if short else ' columns'} {'right' if dx > 0 else 'left'}")
    return (", " if short else " and ").join(parts) or "0"
