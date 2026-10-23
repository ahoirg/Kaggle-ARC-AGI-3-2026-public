"""Exploration coverage for the turn opener: what has not been tested on this level.

Two facts only: keys never pressed on this level (lf52 arrows, cd82 vertical keys) and spots where
a click did nothing but the object there has since changed (vc33 lit gate, bp35 toggle marks). It
reads only the recorded history (boards and results), never game rules.
"""
from __future__ import annotations

import os
import re
from typing import Any, Iterable

from inference.utils.grid_utils import ARC_COLOR_CHARS

_MOUSE = re.compile(r"MOUSE\(row=(\d+), col=(\d+)\)")
_SKIP = {"RESET"}
_MAX_COMPONENT = 400  # larger connected regions are backgrounds or frames, not objects


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def enabled() -> bool:
    return os.environ.get("ARC3_EXPLORATION_NOTE", "1").strip().lower() in {"1", "true", "yes", "on"}


def _interior_key(grid, border: int) -> int:
    rows = len(grid)
    return hash(tuple(tuple(row[border:len(row) - border]) for row in grid[border:rows - border]))


def _component(grid, r: int, c: int, border: int):
    """4-connected same-colour region containing (r, c), inside the border band."""
    rows, cols = len(grid), len(grid[0]) if grid else 0
    if not (0 <= r < rows and 0 <= c < cols):
        return None
    colour = grid[r][c]
    seen = {(r, c)}
    stack = [(r, c)]
    while stack:
        y, x = stack.pop()
        for ny, nx in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
            if (ny, nx) in seen or not (border <= ny < rows - border and border <= nx < cols - border):
                continue
            if grid[ny][nx] == colour:
                seen.add((ny, nx))
                stack.append((ny, nx))
                if len(seen) > _MAX_COMPONENT:
                    return None
    r0 = min(y for y, _ in seen)
    c0 = min(x for _, x in seen)
    shape = frozenset((y - r0, x - c0) for y, x in seen)
    return {"colour": colour, "shape": shape, "at": (r0, c0), "cells": len(seen),
            "size": (max(y for y, _ in seen) - r0 + 1, max(x for _, x in seen) - c0 + 1)}


def _components(grid, border: int) -> list[dict[str, Any]]:
    rows, cols = len(grid), len(grid[0]) if grid else 0
    done = set()
    out = []
    for r in range(border, rows - border):
        for c in range(border, cols - border):
            if (r, c) in done:
                continue
            comp = _component(grid, r, c, border)
            if comp is None:
                # oversized region: mark it with a cheap fill so it is not rescanned
                colour = grid[r][c]
                stack = [(r, c)]
                done.add((r, c))
                while stack:
                    y, x = stack.pop()
                    for ny, nx in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
                        if (ny, nx) not in done and border <= ny < rows - border and border <= nx < cols - border \
                                and grid[ny][nx] == colour:
                            done.add((ny, nx))
                            stack.append((ny, nx))
                continue
            r0, c0 = comp["at"]
            done.update((r0 + y, c0 + x) for y, x in comp["shape"])
            out.append(comp)
    return out


def _name(display: str) -> str:
    return "MOUSE" if display.startswith("MOUSE") else display.split("(")[0].strip()


def _changed(result: dict[str, Any]) -> bool:
    if result.get("level_completed") or result.get("game_over"):
        return True
    scope = result.get("change_scope")
    if scope is not None:
        return scope in {"interior_changed", "shape_changed"}
    return bool(result.get("gameplay_changed"))


def _describe(comp: dict[str, Any]) -> str:
    colour = ARC_COLOR_CHARS[comp["colour"]] if 0 <= comp["colour"] < len(ARC_COLOR_CHARS) else str(comp["colour"])
    h, w = comp["size"]
    return f"{colour} {h}x{w} at ({comp['at'][0]},{comp['at'][1]})"


def level_entries(history: Iterable[Any], level: int) -> list[Any]:
    """History entries of this level: those after the last board of a different level."""
    entries = list(history)
    start = 0
    for index, entry in enumerate(entries):
        frame = getattr(entry, "frame", None)
        if frame is not None and getattr(frame, "level", None) != level:
            start = index + 1
    return entries[start:]


def _level_steps(history: Iterable[Any], level: int):
    """(name, display, before_grid, result, view_offset_before) per executed action on this level.

    Harness-initiated resets are skipped.
    """
    entries = list(history)
    start = len(entries) - len(level_entries(entries, level))
    steps = []
    offset = (0, 0)  # view offset (dy, dx) on this level before each step; a scroll moves screen positions
    for index in range(start + 1, len(entries)):
        entry = entries[index]
        result = getattr(entry, "result", None) or {}
        display = str(getattr(entry, "action", "") or "")
        if _name(display) == "RESET":
            offset = (0, 0)  # the level restarts from its first view
            continue
        if not display or not result.get("executed", True) or _name(display) in _SKIP or result.get("automatic"):
            continue
        steps.append((_name(display), display, entries[index - 1].frame.grid, result, offset))
        shift = result.get("view_shift")
        if shift and len(shift) == 2:
            offset = (offset[0] + int(shift[0]), offset[1] + int(shift[1]))
    return steps, offset


def coverage_lines(history: Iterable[Any], current_grid, level: int, valid_actions: Iterable[str],
                   border: int = 4) -> list[str]:
    """Opener lines: keys never pressed on this level, and spots where a click did nothing but the
    object there has since changed. Two lines at most (r12: the longer note of r5-r11 was read only
    when the model had run out of ideas; the state-count and never-clicked-type lines never changed a
    decision in the logs). Empty until the level has had `ARC3_EXPLORATION_MIN_ACTIONS` actions.
    """
    steps, offset = _level_steps(history, level)
    if len(steps) < _env_int("ARC3_EXPLORATION_MIN_ACTIONS", 6) or not current_grid:
        return []

    valid = [str(a) for a in valid_actions or []]
    tried = {name for name, _, _, _, _ in steps}
    lines: list[str] = []
    untried = [a for a in valid if a not in _SKIP and a not in tried]
    if untried:
        lines.append("- Not tried yet on this level: " + ", ".join(untried) + ".")

    if "MOUSE" in valid:
        changed_since = []
        last_noop: dict[tuple[int, int], tuple[Any, Any, tuple[int, int]]] = {}
        for name, display, grid, result, view in steps:
            match = _MOUSE.match(display)
            if not match:
                continue
            r, c = int(match.group(1)), int(match.group(2))
            comp = _component(grid, r, c, border)
            # keyed by position in the level's first view, so a scroll does not make it a different object
            key = (r - view[0], c - view[1])
            if not _changed(result):
                last_noop[key] = (result.get("action_num"), comp, (r, c))
            else:
                last_noop.pop(key, None)
        rows, cols = len(current_grid), len(current_grid[0]) if current_grid else 0
        for (kr, kc), (action_num, before, (cr, cc)) in sorted(last_noop.items()):
            r, c = kr + offset[0], kc + offset[1]  # where that spot is on the current view
            if not (0 <= r < rows and 0 <= c < cols):
                continue  # scrolled out of view
            if (r, c) != (cr, cc) and not (border <= r < rows - border and border <= c < cols - border):
                continue  # a scroll moved it into the edge band (counters, bars)
            now = _component(current_grid, r, c, border)
            before_key = (before["colour"], before["shape"]) if before else None
            now_key = (now["colour"], now["shape"]) if now else None
            if before_key != now_key and now is not None:
                where = "" if (r, c) == (cr, cc) else f" (that spot is now at row={r}, col={c} after the view scrolled)"
                in_band = not (border <= cr < rows - border and border <= cc < cols - border)
                changed_since.append((in_band, f"MOUSE(row={cr}, col={cc}) did nothing at action {action_num}{where}; "
                                                f"the object there is now {_describe(now)}"))
        if changed_since:
            # board objects first; spots in the edge band (counters, step bars) change on their own
            changed_since = [text for _, text in sorted(changed_since, key=lambda item: item[0])]
            lines.append("- Changed since a click on it did nothing; it may respond now, and one click tests it: "
                         + "; ".join(changed_since[:3]) + ".")
    if not lines:
        return []
    return [f"Exploration on this level ({len(steps)} actions):"] + lines
