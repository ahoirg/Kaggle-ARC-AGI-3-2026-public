"""Sim check for the turn opener: the model's own `observe(frame)` + `step(state, action)` replayed
over the level's recorded transitions every turn (r15; replaces the r13 `predict` check).

Why (r13 FULL): `predict(frame, action)`, a whole predicted board, was defined in 0 of 25 games
and its check was silent when absent; the models' simulators worked on states (`state(fr)` +
`apply_act(st, a)` in sk48) and were checked against the record by hand in 2 games of 25. The
contract and the replay itself live in `sim_contract` (it runs inside the sandbox, next to the
model's code); this module runs it from the host, renders the opener line and, when the pair is
missing, says so once per level after a few actions (the goal check's "no goal_violations is
defined" line preceded the first definition in 14 of 14 games that wrote one).

The same pair also drives `search(goal)` in the sandbox and the solver's batch guard
(`ARC3_SIM_GUARD`): a batch stops at the first action whose observed state differs from
`step(observe(before), action)`. Nothing here reads game rules.
"""
from __future__ import annotations

import json
import os
from typing import Any, Callable

from inference.agent.sim_contract import OBSERVE, STEP

NAMES = (OBSERVE, STEP)
_MARK = "__SIM_CHECK__"


def _flag(name: str, default: str = "1") -> bool:
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "on"}


def enabled() -> bool:
    return _flag("ARC3_SIM_CHECK")


def guard_enabled() -> bool:
    """Batch guard: on with the check unless ARC3_SIM_GUARD=0."""
    return enabled() and _flag("ARC3_SIM_GUARD")


def _int_env(name: str, default: int) -> int:
    try:
        return max(1, int(os.environ.get(name, str(default))))
    except ValueError:
        return default


def _limit() -> int:
    return _int_env("ARC3_SIM_CHECK_TRANSITIONS", 40)


def _nudge_actions() -> int:
    return _int_env("ARC3_SIM_CHECK_NUDGE_ACTIONS", 8)


def evaluate(run: Callable[..., dict[str, Any]], state: dict[str, Any], kept_functions: list[str],
             *, retain_imports: bool = False, timeout_seconds: int = 12, border: int = 4) -> dict[str, Any]:
    """Call the sandbox's sim_check() on the given state. No actions allowed."""

    def refuse(actions, stale_after=None, **_ignored):
        return {"action_result": {"executed": False, "error": "Actions are disabled during the sim check.",
                                  "valid_actions": []}, "state": state}

    code = f"print({_MARK!r} + json.dumps(sim_check({_limit()}), default=repr))\n"
    out = run(code=code, timeout_seconds=timeout_seconds, initial_state=state, action_handler=refuse,
              kept_functions=kept_functions, retain_imports=retain_imports, sim_border=border)
    for line in str(out.get("stdout") or "").splitlines():
        if line.startswith(_MARK):
            try:
                return json.loads(line[len(_MARK):])
            except ValueError:
                break
    error = str(out.get("error") or "no result").strip().splitlines()
    return {"host_error": (error[-1] if error else "no result")[:160]}


def lines(result: dict[str, Any] | None) -> list[str]:
    """The opener line for a retained observe/step pair."""
    if not result:
        return []
    if result.get("host_error"):
        return [f"Sim check: could not replay your observe/step: {result['host_error']}."]
    if not result.get("error") and not int(result.get("checked") or 0):
        return []  # nothing recorded on this level yet: no line
    summary = str(result.get("summary") or "").strip()
    return [summary] if summary else []


def missing_lines(missing: list[str], *, actions_on_level: int, level: int, nudged: set) -> list[str]:
    """Once per level, after a few actions on it, when the pair is not (fully) retained."""
    if actions_on_level < _nudge_actions() or (level, "sim") in nudged:
        return []
    nudged.add((level, "sim"))
    if len(missing) == len(NAMES):
        head = f"Sim check: no `{OBSERVE}(frame)` and `{STEP}(state, action)` are retained"
    else:
        have = [name for name in NAMES if name not in missing]
        head = f"Sim check: `{have[0]}` is retained but `{missing[0]}` is not"
    return [head + ", so no simulator of yours is checked against this level's recorded transitions. With both "
            "retained, each turn message reports where `step` disagrees with the record, `search(goal)` plans over "
            "them, and a batch stops at the first action they mispredict."]
