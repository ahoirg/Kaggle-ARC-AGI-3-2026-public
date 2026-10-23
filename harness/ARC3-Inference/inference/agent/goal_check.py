"""Goal check for the turn opener: run the model's own `goal_violations(frame)`.

Run9/run10 and the 60-minute run showed goals kept in prose: hand-transcribed targets
(ft09), an exact-match goal carried over after it stopped fitting (ar25 L7), goals found
by executing wrong guesses (sk48). When the model retains a function
`goal_violations(frame)` (a list of unmet goal conditions, empty when the goal is met),
the harness evaluates it on the current board every turn and reports the contradiction
that reveals a wrong or incomplete goal model: "0 unmet but the level is not complete".
The check covers only the level on which the function was last defined: a goal written for
one level went stale on later levels in the r4 runs. The function itself stays retained, so
helpers that call it keep working; on a new level the model is asked to define it again.
Until one exists for the level, a level with enough actions gets a one-line nudge instead.
A completion check was dropped: the board before the completing action is one action short
of the goal, and the last old-level frame of the completing action is not the success board
in every game (sk48 flashes, sb26 fades before the switch; replay of 388 logged completions).
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Callable

NAME = "goal_violations"
_DEFINES = re.compile(rf"^def {NAME}\s*\(", re.M)
_MARK = "__GOAL_CHECK__"
_SNIPPET = f"""
import json
_value = {NAME}(current_frame)
if _value is None:
    raise TypeError("{NAME} returned None; return a list of unmet conditions (empty when the goal is met)")
if isinstance(_value, bool):
    _items = ["goal not met"] if _value else []
elif isinstance(_value, int):
    _items = ["unmet"] * max(0, _value)
elif isinstance(_value, (str, bytes)):
    _items = [_value]
else:
    _items = list(_value)
print("{_MARK}" + json.dumps({{"n": len(_items), "first": [str(x)[:80] for x in _items[:3]]}}))
"""


def enabled() -> bool:
    return os.environ.get("ARC3_GOAL_CHECK", "1").strip().lower() in {"1", "true", "yes", "on"}


def _nudge_actions() -> int:
    try:
        return int(os.environ.get("ARC3_GOAL_CHECK_NUDGE_ACTIONS", "8"))
    except ValueError:
        return 8


def evaluate(run: Callable[..., dict[str, Any]], state: dict[str, Any], kept_functions: list[str],
             *, retain_imports: bool = False) -> dict[str, Any]:
    """Run goal_violations on the state's current_frame in the sandbox. No actions allowed.

    `retain_imports` must match the agent's own calls: a retained helper that relies on a
    retained top-level import is otherwise dropped at restore, and goal_violations with it.
    """

    def refuse(actions, stale_after=None):
        return {"action_result": {"executed": False, "error": "Actions are disabled during the goal check.",
                                  "valid_actions": []}, "state": state}

    out = run(code=_SNIPPET, timeout_seconds=5, initial_state=state, action_handler=refuse,
              kept_functions=kept_functions, retain_imports=retain_imports)
    for line in str(out.get("stdout") or "").splitlines():
        if line.startswith(_MARK):
            try:
                return json.loads(line[len(_MARK):])
            except ValueError:
                break
    error = str(out.get("error") or "no result").strip().splitlines()
    return {"error": (error[-1] if error else "no result")[:160]}


def _first(result: dict[str, Any]) -> str:
    items = result.get("first") or []
    return f" (first: {'; '.join(items)})" if items else ""


def level_actions(history, level: int) -> int:
    start = 0
    for index, entry in enumerate(history):
        frame = getattr(entry, "frame", None)
        if frame is not None and getattr(frame, "level", None) != level:
            start = index + 1
    return sum(1 for entry in history[start + 1:]
               if getattr(entry, "action", "") and not (getattr(entry, "result", None) or {}).get("automatic"))


def defines(code: str) -> bool:
    """True when a snippet defines goal_violations at top level (the retained kind)."""
    return bool(_DEFINES.search(code or ""))


def lines(*, defined: bool, current: dict[str, Any] | None, stale_from: int | None,
          level_transition: bool, actions_on_level: int, level: int = 0, nudged: set | None = None) -> list[str]:
    """`nudged` is the caller's per-game set: each nudge is shown once per level (r12; before, every turn)."""
    out: list[str] = []
    nudged = nudged if nudged is not None else set()

    def once(kind: str) -> bool:
        if (level, kind) in nudged:
            return False
        nudged.add((level, kind))
        return True

    if stale_from is not None:
        if level_transition:
            out.append(f"Goal check: your `{NAME}` was defined on level {stale_from}, so it is not checked on this "
                       "level. Define it again for this level's goal (reuse its code if the goal is unchanged).")
        elif actions_on_level >= _nudge_actions() and once("stale"):
            out.append(f"Goal check: `{NAME}` is still the one from level {stale_from}. Write your current goal "
                       "hypothesis for this level as one and plan from it. If a search under it finds no plan, relax "
                       "it (exact match, then covering, then partial) and test the relaxed version.")
        return out
    if current is not None:
        if "error" in current:
            out.append(f"Goal check: {NAME}(current_frame) failed: {current['error']}.")
        elif current["n"] == 0:
            out.append(f"Goal check: {NAME}(current_frame) reports 0 unmet, yet the level is not complete. Either the "
                       "goal model misses a condition, an element or a mechanic, or the function does not see the "
                       "whole goal (objects off-screen, hidden or not counted). Test what you have not tested before "
                       "changing other assumptions.")
        else:
            out.append(f"Goal check: {NAME}(current_frame) reports {current['n']} unmet{_first(current)}.")
    elif not defined and actions_on_level >= _nudge_actions() and once("define"):
        out.append(f"Goal check: no `{NAME}(frame)` is defined. Write your current goal hypothesis as one and plan "
                   "from it. If a search under it finds no plan, relax it (exact match, then covering, then partial) "
                   "and test the relaxed version.")
    return out
