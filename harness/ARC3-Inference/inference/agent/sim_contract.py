"""The simulator contract: the model's own `observe(frame)` and `step(state, action)` (r15).

Why (r13 FULL, 25 games): the r13 check asked for `predict(frame, action)` returning a whole
64x64 board; it was defined in 0 of 25 games, while the models wrote state-level simulators in
many (`state(fr)` + `apply_act(st, a)` in sk48, `ap(st, a)` in lf52, `step3(state, act)` in
lp85, `succP(...)` in m0r0) and searched over them unchecked; only sk48 and r11l ever replayed
their simulator against the record, by hand. Wrong simulators then produced "impossible"
verdicts (m0r0 L5: its search treated the pieces as walls) and long batches executed on wrong
rules (su15 L4, sp80 L3).

The contract is the shape the models already write:
- `observe(frame)` returns the state the model cares about (any value: dict, tuple, list, ...);
- `step(state, action)` returns the state after `action` (the form `action(...)` takes: a key
  name or `{'action': 'MOUSE', 'row': r, 'col': c}`), or None for an action the model believes
  is impossible or fatal there.

Three uses of the same pair, so the simulator that is checked is the one that plans:
- `sim_replay`: step(observe(before), a) == observe(after) over the level's recorded
  transitions, mismatches as field differences, plus the transitions where observe() saw
  nothing although the board interior changed (a blind state passes every check otherwise);
- `sim_search`: breadth-first search over `step` from `observe(current_frame)` to `goal(state)`;
- `sim_guard_verdict`: one executed action of a batch against its prediction; the solver stops
  the batch at the first disagreement.

Stdlib only: the source is embedded into the sandbox child (python_tool_sandbox), where it runs
next to the model's code. Nothing here knows any game.
"""
import collections
import copy
import json
import re
import time

OBSERVE = "observe"
STEP = "step"

_MOUSE = re.compile(r"MOUSE\(row=(-?\d+), col=(-?\d+)\)")
_SKIP_ACTIONS = {"RESET", "UNDO"}


def sim_action_arg(action):
    """A recorded or requested action in the form `action(...)` takes: 'UP', or a MOUSE dict."""
    if isinstance(action, dict):
        name = str(action.get("action", "")).strip().upper()
        if name == "MOUSE":
            return {"action": "MOUSE", "row": int(action.get("row")), "col": int(action.get("col"))}
        return name
    text = str(action or "").strip()
    match = _MOUSE.match(text)
    if match:
        return {"action": "MOUSE", "row": int(match.group(1)), "col": int(match.group(2))}
    return text.upper()


def sim_fresh(state):
    """A private copy of a state for step(): a step that edits its input in place must not change the
    state it was given (a BFS parent, a cached observation). A state that cannot be copied is an
    error, never a silent fallback to the shared object (that would bring back mixed BFS branches)."""
    try:
        return copy.deepcopy(state)
    except Exception as exc:
        raise TypeError("the state from observe/step cannot be copied for step() (%s: %s); build it from "
                        "dicts, lists, tuples, sets, numbers and strings" % (type(exc).__name__, exc)) from None


def sim_action_text(action):
    if isinstance(action, dict):
        return "MOUSE(row=%s, col=%s)" % (action.get("row"), action.get("col"))
    return str(action)


def sim_canon(value, _depth=0):
    """A hashable, order-normalised image of a state, so lists equal tuples and dict order
    does not matter. Objects with a grid (frames) compare by their grid."""
    if _depth > 40:
        return ("deep", repr(value)[:200])
    if value is None or isinstance(value, (bool, int, float, str, bytes)):
        return value
    if isinstance(value, dict):
        items = [(sim_canon(k, _depth + 1), sim_canon(v, _depth + 1)) for k, v in value.items()]
        return ("d", tuple(sorted(items, key=repr)))
    if isinstance(value, (list, tuple)):
        return ("l", tuple(sim_canon(v, _depth + 1) for v in value))
    if isinstance(value, (set, frozenset)):
        return ("s", tuple(sorted((sim_canon(v, _depth + 1) for v in value), key=repr)))
    grid = getattr(value, "_grid", None)
    if grid is not None:
        return ("l", tuple(("l", tuple(row)) for row in grid))
    try:
        hash(value)
        return value
    except TypeError:
        return ("r", repr(value))


def _short(value, limit=60):
    text = repr(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def sim_diff(predicted, observed, limit=3):
    """Where two states differ, as up to `limit` phrases 'path: predicted X, observed Y'."""
    out = []

    def walk(p, o, path):
        if len(out) >= limit:
            return
        if sim_canon(p) == sim_canon(o):
            return
        if isinstance(p, dict) and isinstance(o, dict):
            keys = list(p.keys()) + [k for k in o.keys() if k not in p]
            for key in keys:
                if len(out) >= limit:
                    return
                if key not in o:
                    out.append("%s[%r]: predicted %s, not in observe()" % (path, key, _short(p[key])))
                elif key not in p:
                    out.append("%s[%r]: not predicted, observed %s" % (path, key, _short(o[key])))
                else:
                    walk(p[key], o[key], "%s[%r]" % (path, key))
            return
        if (isinstance(p, (list, tuple)) and isinstance(o, (list, tuple)) and len(p) == len(o)
                and len(p) <= 64):
            for index, (pi, oi) in enumerate(zip(p, o)):
                if len(out) >= limit:
                    return
                walk(pi, oi, "%s[%d]" % (path, index))
            return
        out.append("%s: predicted %s, observed %s" % (path, _short(p), _short(o)))

    walk(predicted, observed, "state")
    return out


def _interior_changed(before, after, border):
    grid_a = getattr(before, "_grid", None)
    grid_b = getattr(after, "_grid", None)
    if grid_a is None or grid_b is None:
        return None
    rows = len(grid_a)
    if rows != len(grid_b):
        return True
    for r in range(border, rows - border):
        row_a, row_b = grid_a[r], grid_b[r]
        cols = len(row_a)
        if cols != len(row_b):
            return True
        if row_a[border:cols - border] != row_b[border:cols - border]:
            return True
    return False


def sim_level_transitions(transitions, level, limit=40):
    """The level's recorded transitions a state simulator can be checked on:
    no automatic resets, no RESET/UNDO, nothing unexecuted, no game over, no level change."""
    rows = []
    for t in transitions:
        before, after = t.before_frame, t.after_frame
        result = t.result or {}
        if before is None or after is None:
            continue
        if before.level != level or after.level != level:
            continue
        if result.get("automatic") or result.get("executed") is False or result.get("game_over"):
            continue
        if str(t.action).strip().upper() in _SKIP_ACTIONS:
            continue
        rows.append(t)
    return rows[-max(1, int(limit)):]


def sim_replay(observe, step, transitions, level, limit=40, border=4, deadline=None):
    """step(observe(before), a) against observe(after) on the level's recorded transitions."""
    rows = sim_level_transitions(transitions, level, limit)
    seen = {}

    def obs(frame):
        key = id(frame)
        if key not in seen:
            seen[key] = (frame, observe(frame))  # keep the frame alive so the id stays unique
        return seen[key][1]

    checked, mismatches, blind = 0, [], []
    for t in rows:
        if deadline is not None and time.monotonic() > deadline:
            return {"checked": checked, "mismatches": mismatches, "blind": blind, "timed_out": True}
        num = (t.result or {}).get("action_num")
        arg = sim_action_arg(t.action)
        try:
            s0 = obs(t.before_frame)
            s1 = obs(t.after_frame)
            predicted = step(sim_fresh(s0), arg)
        except Exception as exc:
            return {"checked": checked, "mismatches": mismatches, "blind": blind,
                    "error": (type(exc).__name__ + ": " + str(exc))[:160], "at": num,
                    "action": sim_action_text(arg)}
        checked += 1
        if sim_canon(s0) == sim_canon(s1) and _interior_changed(t.before_frame, t.after_frame, border):
            blind.append(num)
        if predicted is None:
            mismatches.append({"action_num": num, "action": sim_action_text(arg),
                               "diff": ["step returned None (impossible in your model), but the action executed"]})
        elif sim_canon(predicted) != sim_canon(s1):
            mismatches.append({"action_num": num, "action": sim_action_text(arg), "diff": sim_diff(predicted, s1)})
    return {"checked": checked, "mismatches": mismatches, "blind": blind}


def sim_summary(result):
    """One line for the turn message and for search(): what the replay found."""
    if not result:
        return ""
    checked = int(result.get("checked") or 0)
    mis = list(result.get("mismatches") or [])
    head = "Sim check: "
    if result.get("error"):
        line = (head + "observe/step failed on action %s (%s): %s" % (result.get("at"), result.get("action"),
                                                                       result["error"]))
        if checked:
            line += "; before that, step matched %d of %d recorded transitions" % (checked - len(mis), checked)
        return line + "."
    if checked == 0:
        return head + "no recorded transition on this level to check step(state, action) against yet."
    span = " (time limit: only the first %d checked)" % checked if result.get("timed_out") else ""
    if not mis:
        line = head + "step(observe(before), action) equals observe(after) on all %d recorded transitions of this level%s." % (
            checked, span)
    else:
        parts = []
        for m in mis[:3]:
            parts.append("action %s (%s: %s)" % (m.get("action_num"), m.get("action"), "; ".join(m.get("diff") or [])))
        more = ", and %d more" % (len(mis) - 3) if len(mis) > 3 else ""
        line = (head + "step(observe(before), action) equals observe(after) on %d of %d recorded transitions of this "
                "level%s; mismatches at %s%s." % (checked - len(mis), checked, span, "; ".join(parts), more))
    blind = [b for b in (result.get("blind") or []) if b is not None]
    if blind:
        shown = ", ".join(str(b) for b in blind[:5]) + (", ..." if len(blind) > 5 else "")
        line += (" observe() returned the same state before and after %d action(s) that changed the board "
                 "interior (actions %s)." % (len(blind), shown))
    return line


def sim_search(step, start, goal, actions, max_depth=60, max_nodes=200000, time_limit=15.0):
    """Breadth-first search over the model's step from `start` until goal(state) is true.

    `actions` is a list of actions or a function state -> actions. A step that returns None
    prunes that move; step always receives a private copy of the state. The clock is checked
    before every move, so a slow step overruns `time_limit` by at most one call. Returns the
    shortest plan found, or why there is none."""
    t0 = time.monotonic()
    deadline = t0 + max(0.1, float(time_limit))

    def done(status, plan=None, **extra):
        out = {"status": status, "plan": plan, "length": None if plan is None else len(plan),
               "expanded": expanded, "states": len(parents), "depth": depth,
               "seconds": round(time.monotonic() - t0, 2)}
        out.update(extra)
        return out

    expanded, depth = 0, 0
    parents = {}
    try:
        start_key = sim_canon(start)
        parents[start_key] = None
        if goal(start):
            return done("found", [])
        frontier = collections.deque([(start, start_key, 0)])
        while frontier:
            state, key, d = frontier.popleft()
            if d >= max_depth:
                continue
            depth = max(depth, d + 1)
            expanded += 1
            if time.monotonic() > deadline:
                return done("time_limit")
            moves = actions(sim_fresh(state)) if callable(actions) else actions
            for move in moves:
                if time.monotonic() > deadline:
                    return done("time_limit")
                nxt = step(sim_fresh(state), move)
                if nxt is None:
                    continue
                nkey = sim_canon(nxt)
                if nkey in parents:
                    continue
                parents[nkey] = (key, move)
                if goal(nxt):
                    plan = [move]
                    back = parents[key]
                    while back is not None:
                        plan.append(back[1])
                        back = parents[back[0]]
                    plan.reverse()
                    return done("found", plan)
                if len(parents) >= max_nodes:
                    return done("max_nodes")
                frontier.append((nxt, nkey, d + 1))
    except Exception as exc:
        return done("error", error=(type(exc).__name__ + ": " + str(exc))[:200])
    return done("max_depth" if depth >= max_depth else "exhausted")


def sim_guard_verdict(observe, step, before, action, after):
    """One executed batch action against the model's prediction: ok True/False, or None when
    nothing could be checked (UNDO/RESET, or observe/step raised)."""
    arg = sim_action_arg(action)
    if sim_action_text(arg).upper() in _SKIP_ACTIONS:
        return {"ok": None}
    try:
        s0 = observe(before)
        predicted = step(sim_fresh(s0), arg)
        s1 = observe(after)
    except Exception as exc:
        return {"ok": None, "error": (type(exc).__name__ + ": " + str(exc))[:160]}
    if predicted is None:
        return {"ok": False, "detail": "step returned None (impossible in your model), but the game executed it"}
    if sim_canon(predicted) == sim_canon(s1):
        return {"ok": True}
    return {"ok": False, "detail": "; ".join(sim_diff(predicted, s1))}


def sim_json(value):
    return json.loads(json.dumps(value, default=repr))
