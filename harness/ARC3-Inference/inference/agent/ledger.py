"""Level ledger: the facts of this level, rebuilt from the recorded history (r10, lever 5).

Why: the context is trimmed 1-4 times per game (to 25-60% of the messages), silently, and the
stuck games are trimmed on the level they are stuck on. Retained functions survive a trim; action
results, deaths and dead ends do not, and the model is not told. This module rebuilds, from the
boards and results only, what each action did on this level, which attempts ended and how, and
how far the view has scrolled, so the first turn after a trim restates what was lost. Nothing here
reads game rules.
"""
from __future__ import annotations

import os
import re
from collections import defaultdict
from typing import Any, Iterable

from inference.agent.exploration import _changed, _interior_key, _name
from inference.utils import motion

_MOUSE = re.compile(r"MOUSE\(row=(\d+), col=(\d+)\)")


def mode() -> str:
    """'drain' (default): shown on the first turn after a context trim; 'always'; 'off'."""
    value = os.environ.get("ARC3_LEVEL_LEDGER", "drain").strip().lower()
    if value in {"0", "off", "false", "no"}:
        return "off"
    if value in {"1", "always", "on", "true", "yes"}:
        return "always"
    return "drain"


def _level_steps(history: Iterable[Any], level: int):
    entries = list(history)
    start = 0
    for index, entry in enumerate(entries):
        frame = getattr(entry, "frame", None)
        if frame is not None and getattr(frame, "level", None) != level:
            start = index + 1
    steps = []  # (name, display, before_grid, after_grid, result)
    for index in range(start + 1, len(entries)):
        entry = entries[index]
        result = getattr(entry, "result", None) or {}
        display = str(getattr(entry, "action", "") or "")
        if not display or result.get("automatic") or _name(display) == "RESET" or not result.get("executed", True):
            continue
        before = getattr(entries[index - 1], "frame", None)
        after = getattr(entry, "frame", None)
        if before is None or after is None:
            continue
        steps.append((_name(display), display, before.grid, after.grid, result))
    return steps


def ledger_lines(history: Iterable[Any], level: int, *, border: int = 4, max_lines: int = 8) -> list[str]:
    steps = _level_steps(history, level)
    if not steps:
        return []
    lines: list[str] = []
    by_key: dict[str, list] = defaultdict(list)
    for step in steps:
        by_key[step[0]].append(step)
    deaths = [step for step in steps if (step[4] or {}).get("game_over")]
    offset = [0, 0]
    for step in steps:
        shift = (step[4] or {}).get("view_shift")
        if shift and len(shift) == 2:
            offset = [offset[0] + int(shift[0]), offset[1] + int(shift[1])]
    order = sorted(by_key.items(), key=lambda kv: -len(kv[1]))
    for key, items in order:
        states = {_interior_key(grid, border) for _, _, grid, _, _ in items}
        changing = [step for step in items if _changed(step[4] or {}) and not (step[4] or {}).get("game_over")]
        changing_states = {_interior_key(step[2], border) for step in changing}
        if key == "MOUSE":
            text = f"MOUSE: {len(items)} click(s), {len(changing)} changed the board"
        else:
            text = f"{key}: {len(items)} press(es); changed the board in {len(changing_states)} of {len(states)} board state(s) tried"
        fatal_here = sum(1 for step in items if (step[4] or {}).get("game_over"))
        if fatal_here:
            text += f"; ended the attempt {fatal_here} time(s)"
        if changing:
            last = changing[-1]
            phrases = motion.motion_lines(last[2], last[3], border=border, max_items=1)
            if phrases:
                text += f"; last change ({last[1]}, action {(last[4] or {}).get('action_num', '?')}): {phrases[0]}"
        lines.append("- " + text + ".")
    if deaths:
        fatal = [step[1] for step in deaths][-3:]
        text = f"- Attempts ended on this level: {len(deaths)}; fatal action(s): {', '.join(fatal)}"
        last = deaths[-1]
        phrases = motion.motion_lines(last[2], last[3], border=border, max_items=2)
        if phrases:
            text += f"; in the fatal step: {'; '.join(phrases)}"
        lines.append(text + ".")
    if any(offset):
        parts = []
        if offset[0]:
            parts.append(f"{abs(offset[0])} rows {'down' if offset[0] > 0 else 'up'}")
        if offset[1]:
            parts.append(f"{abs(offset[1])} columns {'right' if offset[1] > 0 else 'left'}")
        lines.append(f"- Total view shift on this level: {', '.join(parts)}; positions noted before a scroll moved by that much.")
    states_all = {_interior_key(grid, border) for _, _, grid, _, _ in steps}
    header = (f"Level ledger (rebuilt from the recorded history of this level: {len(steps)} actions, "
              f"{len(states_all)} distinct board states):")
    return [header] + lines[:max_lines]


def drain_notice(dropped_replies: int, notes: bool = False) -> str:
    survives = ("Your notes at the top of the context, your retained functions and the ledger below are what survives"
                if notes else "Your retained functions and the ledger below are what survives")
    return (f"Context note: your context was trimmed; your oldest {dropped_replies} repl{'y' if dropped_replies == 1 else 'ies'} "
            f"on this game and their tool results are no longer in it. {survives}; do not assume you tested something "
            "unless it is listed.")
