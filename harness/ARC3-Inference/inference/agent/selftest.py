"""CPU self-test of the harness contract, run by the notebook before the model loads.

Checks that the model-facing text and the code that implements it agree: the
system prompt builds and states the current rules, the Python sandbox exposes the
documented globals with the documented types, the stale-state guard fires only on
a true no-op, and the observation classifier and passive recorder behave as the
prompt describes. No network, GPU or game engine is needed.
"""
from __future__ import annotations

import gzip
import json
import time
import os
import tempfile
from pathlib import Path
from unittest.mock import patch


def _grid(points=()):
    grid = [[0] * 64 for _ in range(64)]
    for r, c in points:
        grid[r][c] = 1
    return grid


def _state(grid, step=0, level=1, valid=("LEFT", "RIGHT", "SPACE")):
    frame = {"ascii": "\n".join("W" * 64 for _ in range(64)), "step": step, "level": level,
             "shape": [64, 64], "grid": grid}
    return {"current_frame": frame, "history": [{"action": "", "frame": frame, "result": {}}],
            "valid_actions": list(valid), "last_action_call_result": {}}


def self_test() -> dict:
    checks: list[str] = []
    from inference.agent import prompts
    from inference.agent.action_observation import classify, stop_reason
    from inference.agent.python_tool_sandbox import run_sandboxed_python

    # 1. System prompt: builds, states the live rules, carries no retired concepts.
    text = prompts.build_system_prompt(output_chars=12288, border=4, undo=True, reset=False)
    for needed in ("StaleStateActionError", "`UNDO`", "bbox", "last_animation_timeline", "previous_frame",
                   "level_completed", "Classes may be defined", "12288", "Exploration on this level", "goal_violations(frame)",
                   "view_shift", "scrolls the view while clicks remain", "so is a click after the view scrolled",
                   "`observe(frame)`", "`step(state, action)`", "`sim_check(limit=40)`", "`search(goal, actions=None",
                   "`sim_mismatch`", "`found_unverified`", "`sim_guard_error`", "receives its own copy of the state",
                   "the call's time ran low", "`blind` lists the actions"):
        assert needed in text, needed
    for retired in ("world model", "BAR RULE", "ephemeral", "3072 tokens", "never use it to scan", "predict(",
                    "probe'", "harness probe"):
        assert retired not in text, retired
    checks.append("system prompt states the live contract and no retired concepts")

    # 2. Observation classifier semantics the prompt documents.
    before = _grid()
    assert classify(before, before)["no_op"] is True and stop_reason(classify(before, before)) == "no_visible_change"
    edge = classify(before, _grid([(1, 1)]))
    assert edge["no_op"] is None and edge["change_scope"] == "border_only"
    inside = classify(before, _grid([(20, 20)]))
    assert inside["no_op"] is False and inside["change_scope"] == "interior_changed"
    assert classify(before, before, animation={"transient_pixels": 5})["no_op"] is None
    checks.append("change_scope / no_op classification")

    # 3. Sandbox: documented globals and types, classes, imports, stale guard on a true no-op only.
    calls = []

    def action_handler(actions, stale_after=None):
        calls.append((actions, stale_after))
        if stale_after:
            return {"action_result": {"executed": False, "stop_reason": "stale_state",
                                      "stop_detail": "refused"}, "state": _state(before)}
        name = actions[0]["action"]
        no_op = name == "SPACE"
        result = {"executed": True, "executed_actions": [name], "no_op": no_op,
                  "gameplay_changed": not no_op, "level": 1}
        return {"action_result": result, "state": _state(before if no_op else _grid([(20, 20)]), step=len(calls))}

    def animation_handler(request):
        if request.get("kind") == "frames":
            return []
        return {"action": "SPACE", "frames": 0, "steps": []}

    code = """
import json, heapq, difflib
class Box:
    def __init__(self, v):
        self.v = v
print('cls', Box(2).v)
print('types', type(last_animation_frames).__name__, type(last_animation_timeline).__name__)
print('json', json.dumps(last_animation_timeline))
node = current_frame.segmentation['nodes'][0]
print('bbox', node['bbox'])
r = action(['LEFT'])
print('ran', r['executed_actions'])
r = action(['SPACE'])
try:
    action(['RIGHT'])
    print('NOT REFUSED')
except StaleStateActionError:
    print('refused')
"""
    out = run_sandboxed_python(code=code, timeout_seconds=20, initial_state=_state(before),
                               action_handler=action_handler, animation_handler=animation_handler,
                               kept_functions=[])
    stdout = out.get("stdout", "")
    assert not out.get("error"), out.get("error")
    for needed in ("cls 2", "types list dict", "bbox [0, 0, 63, 63]", "ran ['LEFT']", "refused"):
        assert needed in stdout, (needed, stdout)
    assert calls[-1][1] == "SPACE", calls
    checks.append("sandbox globals, classes, imports, animation types, stale guard after a true no-op")

    # 4. Border-only result must not arm the stale guard.
    calls.clear()

    def border_handler(actions, stale_after=None):
        calls.append((actions, stale_after))
        result = {"executed": True, "executed_actions": [actions[0]["action"]], "no_op": None,
                  "gameplay_changed": False, "level": 1}
        return {"action_result": result, "state": _state(before)}

    out = run_sandboxed_python(code="action(['LEFT'])\naction(['RIGHT'])\nprint('both ran')", timeout_seconds=20,
                               initial_state=_state(before), action_handler=border_handler)
    assert "both ran" in out.get("stdout", "") and all(s is None for _, s in calls), (out, calls)
    checks.append("border-only / inconclusive results do not arm the stale guard")

    # 5. An action outside valid_actions fails loudly instead of returning an empty result.
    def invalid_handler(actions, stale_after=None):
        return {"action_result": {"executed": False, "error": "UNDO is not valid right now.",
                                  "valid_actions": ["LEFT", "RIGHT"]}, "state": _state(before)}

    out = run_sandboxed_python(code="r = action(['UNDO'])\nprint('NO ERROR', r)", timeout_seconds=20,
                               initial_state=_state(before), action_handler=invalid_handler)
    assert "NO ERROR" not in out.get("stdout", "") and "ACTION NOT EXECUTED" in str(out.get("error")), out
    checks.append("invalid action reported as not executed")

    # 6. Exploration note: untried actions, state-specific no-ops, objects changed after a no-effect click.
    from inference.agent.exploration import coverage_lines
    from inference.agent.runtime_state import Frame, HistoryEntry

    def board(points, colour=1):
        grid = [[0] * 64 for _ in range(64)]
        for r, c in points:
            grid[r][c] = colour
        return tuple(tuple(row) for row in grid)

    a, b = board([(20, 20)]), board([(20, 21)])
    lit = tuple(tuple(12 if (r, c) == (40, 40) else v for c, v in enumerate(row)) for r, row in enumerate(b))
    dim = tuple(tuple(5 if (r, c) == (40, 40) else v for c, v in enumerate(row)) for r, row in enumerate(b))
    hist = [HistoryEntry("", Frame(a, 0, 1))]
    steps = [("UP", a, "no_visible_change"), ("UP", a, "no_visible_change"), ("RIGHT", b, "interior_changed"),
             ("MOUSE(row=40, col=40)", dim, "interior_changed"), ("MOUSE(row=40, col=40)", dim, "no_visible_change"),
             ("RIGHT", dim, "no_visible_change"), ("RIGHT", lit, "interior_changed")]
    for n, (name, grid, scope) in enumerate(steps, 1):
        hist.append(HistoryEntry(name, Frame(grid, n, 1), {"executed": True, "action_num": n, "change_scope": scope}))
    note = "\n".join(coverage_lines(hist, lit, 1, ["UP", "DOWN", "RIGHT", "MOUSE"]))
    for needed in ("Not tried yet on this level: DOWN", "one click tests it: MOUSE(row=40, col=40) did nothing at action 5"):
        assert needed in note, (needed, note)
    assert "tries, from" not in note and "never clicked" not in note, note  # r12: two lines only
    checks.append("exploration note")

    # 6b. View shift: a scroll is detected from the two boards only; clicks are tracked through it.
    import random as _random
    from inference.utils import view_shift
    from inference.agent.action_observation import verdict

    def blocks(spec, size=64):
        grid = [[0] * size for _ in range(size)]
        for (r0, c0, h, w, colour) in spec:
            for r in range(r0, r0 + h):
                for c in range(c0, c0 + w):
                    if 0 <= r < size and 0 <= c < size:
                        grid[r][c] = colour
        return tuple(tuple(row) for row in grid)

    rng = _random.Random(3)
    world = [(rng.randrange(-40, 100), rng.randrange(0, 60), rng.randrange(2, 7), rng.randrange(2, 7), rng.randrange(1, 15))
             for _ in range(140)]
    view = lambda dy, dx: blocks([(r + dy, c + dx, h, w, k) for r, c, h, w, k in world])
    for dy, dx in ((6, 0), (-24, 0), (0, 8), (-30, 15)):
        found = view_shift.detect(view(0, 0), view(dy, dx))
        assert found and (found["dy"], found["dx"]) == (dy, dx), (dy, dx, found)
    assert view_shift.detect(view(0, 0), view(0, 0)) is None
    moved_one = blocks([(r + (5 if i == 0 else 0), c, h, w, k) for i, (r, c, h, w, k) in enumerate(world)])
    assert view_shift.detect(view(0, 0), moved_one) is None
    recoloured = tuple(tuple((v + 1) % 16 for v in row) for row in view(0, 0))
    assert view_shift.detect(view(0, 0), recoloured) is None
    assert "view scrolled 24 down" in verdict({"change_scope": "interior_changed", "view_shift": [24, 0]})
    # a 52-row jump leaves too little overlap for the before/after test; the animation steps cover it
    assert view_shift.detect(view(0, 0), view(52, 0)) is None
    jump = view_shift.detect_action(view(0, 0), view(52, 0), [view(13, 0), view(26, 0), view(39, 0), view(52, 0)])
    assert jump and (jump["dy"], jump["dx"]) == (52, 0) and jump.get("from_animation"), jump
    blank = tuple(tuple(0 for _ in row) for row in view(0, 0))
    nearly = tuple(tuple(15 if (r, c) in ((40, 40), (40, 41)) else v for c, v in enumerate(row))
                   for r, row in enumerate(view(0, 0)))
    away_and_back = [view(0, -24), view(0, -48), blank, nearly]   # slides out, cuts back: not a scroll
    assert view_shift.detect_action(view(0, 0), nearly, away_and_back) is None
    walk = [moved_one, view(0, 0)]                     # one object moves and returns: no scroll
    assert view_shift.detect_action(view(0, 0), moved_one, walk[:1]) is None
    assert view_shift.residual(view(0, 0), view(6, 0), 6, 0) == 0
    a_block, b_block = (30, 30, 3, 3, 3), (24, 30, 3, 3, 4)
    s0 = blocks([a_block, b_block])
    s1 = blocks([(36, 30, 3, 3, 3), (30, 30, 3, 3, 4)])          # the view scrolled 6 rows down
    s2 = blocks([(36, 30, 3, 3, 12), (30, 30, 3, 3, 4)])         # ... and the clicked object lit up
    hist = [HistoryEntry("", Frame(s0, 0, 1)),
            HistoryEntry("MOUSE(row=31, col=31)", Frame(s0, 1, 1), {"executed": True, "action_num": 1, "change_scope": "frame_unchanged"})]
    for n in range(2, 7):
        hist.append(HistoryEntry("UP", Frame(s0, n, 1), {"executed": True, "action_num": n, "change_scope": "frame_unchanged"}))
    hist.append(HistoryEntry("DOWN", Frame(s1, 7, 1), {"executed": True, "action_num": 7, "change_scope": "interior_changed",
                                                       "view_shift": [6, 0]}))
    quiet = "\n".join(coverage_lines(hist, s1, 1, ["UP", "DOWN", "MOUSE"]))
    assert "did nothing at action 1" not in quiet, quiet   # same object, only scrolled
    lit_note = "\n".join(coverage_lines(hist, s2, 1, ["UP", "DOWN", "MOUSE"]))
    assert "MOUSE(row=31, col=31) did nothing at action 1 (that spot is now at row=37, col=31" in lit_note, lit_note
    edge = [HistoryEntry("", Frame(s0, 0, 1)),
            HistoryEntry("MOUSE(row=31, col=31)", Frame(s0, 1, 1), {"executed": True, "action_num": 1, "change_scope": "frame_unchanged"})]
    for n in range(2, 7):
        edge.append(HistoryEntry("UP", Frame(s0, n, 1), {"executed": True, "action_num": n, "change_scope": "frame_unchanged"}))
    edge.append(HistoryEntry("DOWN", Frame(blocks([(61, 30, 3, 3, 5)]), 7, 1),
                             {"executed": True, "action_num": 7, "change_scope": "interior_changed", "view_shift": [30, 0]}))
    edge_note = "\n".join(coverage_lines(edge, blocks([(61, 30, 3, 3, 5)]), 1, ["UP", "DOWN", "MOUSE"]))
    assert "did nothing at action 1" not in edge_note, edge_note   # moved into the edge band: not an object
    from inference.agent.tool_agent import ToolAgent as _Agent
    scrolled = {"view_shifts": [{"action": "RIGHT", "shift": [0, -6], "other_cells": 40}], "view_offset": [0, -30]}
    lines = _Agent._view_shift_lines(scrolled)
    assert lines[-1] == "Total view shift on this level: 30 columns left.", lines
    lines = _Agent._view_shift_lines(dict(scrolled, view_offset_uncertain=17))
    assert lines[-1].startswith("Total view shift on this level: 30 columns left (uncertain: action 17 changed"), lines
    assert _Agent._view_shift_lines({"view_offset_uncertain": 17}) == []   # no scroll in this call: no lines
    checks.append("view shift: detection, verdict, exploration positions, total line")

    # 7. Goal check: the model's retained goal_violations runs in the sandbox, with retained imports.
    from inference.agent import goal_check

    source = "def goal_violations(frame):\n    return [c for c in frame.ascii if c == 'Z'][:0] + ['a', 'b']\n"
    result = goal_check.evaluate(run_sandboxed_python, _state(before), [source])
    assert result.get("n") == 2 and result.get("first") == ["a", "b"], result
    helper = "import re\ndef runs(row):\n    return [m.group(0) for m in re.finditer(r'W{4}', row)]\n"
    uses_helper = "def goal_violations(frame):\n    return runs(frame.ascii.split(chr(10))[0])[:1]\n"
    with_import = goal_check.evaluate(run_sandboxed_python, _state(before), [helper, uses_helper], retain_imports=True)
    assert with_import == {"n": 1, "first": ["WWWW"]}, with_import
    empty = goal_check.evaluate(run_sandboxed_python, _state(before), ["def goal_violations(frame):\n    return []\n"])
    broken = goal_check.evaluate(run_sandboxed_python, _state(before), ["def goal_violations(frame):\n    return None\n"])
    assert empty == {"n": 0, "first": []} and "error" in broken, (empty, broken)
    text = "\n".join(goal_check.lines(defined=True, current=empty, stale_from=None, level_transition=False,
                                      actions_on_level=3))
    assert "0 unmet, yet the level is not complete" in text, text
    stale = goal_check.lines(defined=True, current=None, stale_from=2, level_transition=True, actions_on_level=0)
    assert stale and "defined on level 2" in stale[0], stale
    later = goal_check.lines(defined=True, current=None, stale_from=2, level_transition=False, actions_on_level=9)
    assert later and "from level 2" in later[0] and "relax it" in later[0], later
    assert goal_check.lines(defined=True, current=None, stale_from=2, level_transition=False, actions_on_level=3) == []
    nudge = goal_check.lines(defined=False, current=None, stale_from=None, level_transition=False, actions_on_level=9)
    assert nudge and "relax it" in nudge[0], nudge
    assert goal_check.defines("x = 1\ndef goal_violations(frame):\n    return []\n")
    assert not goal_check.defines("def helper():\n    def goal_violations(frame):\n        return []\n")

    # Agent side: the check covers the level of the last definition; the function stays retained;
    # a level change restores the per-level counters and the game-scope notice.
    from types import SimpleNamespace
    from inference.agent.tool_agent import ToolAgent
    kept = {"goal_violations": "def goal_violations(frame):\n    return []\n",
            "plan": "def plan(frame):\n    return goal_violations(frame)\n"}
    fake = SimpleNamespace(_kept_functions=dict(kept), _goal_defined_level=1, _retained_clear_notice="",
                           _actions_at_level_start=0, _tokens_at_level_start=0, _session_generated_tokens=500)
    fake._retained_sources = lambda: list(fake._kept_functions.values())
    from inference.agent.runtime_state import Frame
    frame2 = Frame(grid=tuple(tuple(row) for row in _grid()), step=9, level=2)
    env = {"ARC3_PERSISTENT_FUNCTIONS": "1", "ARC3_PERSISTENT_FUNCTIONS_SCOPE": "game", "ARC3_GOAL_CHECK": "1"}
    with patch.dict(os.environ, env):
        summary = {"level_transition": True, "level": 2, "end_action_num": 9}
        ToolAgent._note_level_transition(fake, summary)
        assert (fake._actions_at_level_start, fake._tokens_at_level_start) == (9, 500), vars(fake)
        assert fake._retained_clear_notice.startswith("Your retained functions remain available"), fake._retained_clear_notice
        opener = ToolAgent._goal_check_lines(fake, [], frame2, 2, summary)
        assert len(opener) == 1 and "defined on level 1" in opener[0], opener
        assert set(fake._kept_functions) == set(kept), "goal_violations and its callers stay retained"
        fake._goal_defined_level = 2
        opener = ToolAgent._goal_check_lines(fake, [], frame2, 2, {})
        assert len(opener) == 1 and "0 unmet" in opener[0], opener
    _gn = set()
    nudge_lines = goal_check.lines(defined=False, current=None, stale_from=None, level_transition=False, actions_on_level=9, level=2, nudged=_gn)
    assert nudge_lines and "no `goal_violations(frame)` is defined" in nudge_lines[0], nudge_lines
    assert goal_check.lines(defined=False, current=None, stale_from=None, level_transition=False, actions_on_level=10, level=2, nudged=_gn) == [], "nudge once per level"
    checks.append("goal check (nudge once per level)")

    # r9: a snippet that did not finish (timeout) keeps the functions retained before it.
    holder = SimpleNamespace(_kept_functions=dict(kept))
    with patch.dict(os.environ, {"ARC3_PERSISTENT_FUNCTIONS": "1"}):
        payload = {}
        ToolAgent._record_retained_functions(holder, {"error": "Tool timed out after 30s", "stdout": "", "action_results": []}, payload)
        assert holder._kept_functions == kept and "previously retained functions are unchanged" in payload["function_retention"], payload
        ToolAgent._record_retained_functions(holder, {"stdout": "", "keepable_functions": [], "action_results": []}, payload)
        assert holder._kept_functions == {}, "a finished snippet still replaces the retained set"
    checks.append("retained functions survive a timed-out snippet")

    # r10 lever 1: object motion phrases per action, inside a batch, and through an animation.
    from inference.utils import motion as _motion
    _a = blocks([(35, 33, 4, 4, 12)])
    _b = blocks([(35, 26, 4, 4, 12), (58, 28, 2, 2, 14)])
    _c = blocks([(35, 26, 4, 4, 12)])
    phrases = _motion.motion_lines(_a, _b)
    assert phrases[0] == "O 4x4 moved (35,33)->(35,26): 7 columns left", phrases
    batch = _motion.batch_motion([_a, _b, _c], ["LEFT", "RIGHT"])
    assert [p[1] for p in batch["per_action"]] == ["LEFT", "RIGHT"] and batch["per_action"][0][2][0].startswith("O 4x4 moved"), batch
    chain = [blocks([(20, 17, 6, 6, 5)]), blocks([(20, 23, 6, 6, 5)]), blocks([(20, 29, 6, 6, 5)]), blocks([(20, 29, 6, 6, 3)])]
    tracks = _motion.animation_tracks(chain)
    assert tracks and tracks[0].startswith("B 6x6 moved (20,17)->(20,23)->(20,29) over 2 step(s)") and "turned G" in tracks[0], tracks
    _entries = [SimpleNamespace(action="", frame=SimpleNamespace(grid=_a), result={}),
                SimpleNamespace(action="LEFT", frame=SimpleNamespace(grid=_b), result={"executed": True}),
                SimpleNamespace(action="RIGHT", frame=SimpleNamespace(grid=_c), result={"executed": True})]
    _holder = SimpleNamespace(_opener_history_entries=_entries, _last_animation_chain=chain)
    opener_lines = ToolAgent._motion_summary_lines(_holder, {"executed_count": 2})
    assert len(opener_lines) == 1 and opener_lines[0].startswith("Per action (objects matched by colour and shape): 1 LEFT: O 4x4 moved"), opener_lines
    assert ToolAgent._animation_track_lines(_holder)[0].startswith("Animation tracks"), "tracks line"
    with patch.dict(os.environ, {"ARC3_MOTION_SUMMARY": "0"}):
        assert ToolAgent._motion_summary_lines(_holder, {"executed_count": 2}) == [], "kill switch"
    checks.append("motion summary: per action, animation tracks")


    # r10 lever 5: the level ledger from history, shown with the drain notice once after a context trim.
    from inference.agent import ledger
    _E = lambda act, g, lvl=1, **r: SimpleNamespace(action=act, frame=SimpleNamespace(grid=g, level=lvl), result=r)
    _la, _lb, _lc = blocks([(35, 33, 4, 4, 12)]), blocks([(35, 26, 4, 4, 12)]), blocks([(35, 20, 4, 4, 12)])
    _lh = [_E("", _la, 0), _E("MOUSE(row=1, col=1)", _la, 1, executed=True, change_scope="interior_changed", level_completed=True),
           _E("LEFT", _lb, executed=True, change_scope="interior_changed", action_num=2),
           _E("UP", _lb, executed=True, change_scope="border_only", action_num=3),
           _E("LEFT", _lc, executed=True, change_scope="interior_changed", action_num=4, view_shift=[0, -6]),
           _E("DOWN", _la, executed=True, game_over=True, action_num=5), _E("RESET", _la, automatic=True)]
    _ll = ledger.ledger_lines(_lh, 1)
    assert _ll[0].startswith("Level ledger (rebuilt from the recorded history of this level: 4 actions") and \
        _ll[1].startswith("- LEFT: 2 press(es); changed the board in 2 of 2 board state(s) tried; last change (LEFT, action 4): O 4x4 moved (35,26)->(35,20): 6 columns left") and \
        any(line.startswith("- Attempts ended on this level: 1; fatal action(s): DOWN") for line in _ll) and \
        any("Total view shift on this level: 6 columns left" in line for line in _ll), _ll
    _lholder = SimpleNamespace(_drain_pending=12)
    _shown = ToolAgent._ledger_lines(_lholder, _lh, 1)
    assert _shown[0].startswith("Context note: your context was trimmed; your oldest 12 replies") and _shown[1].startswith("Level ledger"), _shown
    assert ToolAgent._ledger_lines(_lholder, _lh, 1) == [], "shown once per drain"
    with patch.dict(os.environ, {"ARC3_LEVEL_LEDGER": "always"}):
        assert ToolAgent._ledger_lines(SimpleNamespace(_drain_pending=0), _lh, 1)[0].startswith("Level ledger"), "always mode"
    checks.append("level ledger after a context trim")

    # 8. Passive recorder writes exact boards.
    from inference.agent import observation_recorder as recorder
    registry = dict(recorder.REGISTRY)
    try:
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"ARC3_OBSERVATION_DIR": tmp}):
            rec = recorder.ObservationRecorder("selftest")
            rec.transition(before, _grid([(20, 20)]), {"action_name": "LEFT", "action_display": "LEFT",
                                                         "action_num": 1, "level": 1}, 1)
            with gzip.open(Path(rec.root) / "evidence.jsonl.gz", "rt") as handle:
                rows = [json.loads(line) for line in handle]
            assert [r["kind"] for r in rows] == ["initial", "transition"] and rec.disk_errors == 0
    finally:
        recorder.REGISTRY.clear()
        recorder.REGISTRY.update(registry)
    checks.append("passive observation archive")

    # (13. auto-probe was removed in r16.)
    def block(r0, c0, h, w, colour, base=None):
        grid = [list(row) for row in base] if base is not None else [[0] * 64 for _ in range(64)]
        for r in range(r0, r0 + h):
            for c in range(c0, c0 + w):
                grid[r][c] = colour
        return tuple(tuple(row) for row in grid)

    # 14. Simulator contract (r15): the model's observe(frame) + step(state, action) replayed over the level's
    # transitions, search() over the same pair, the batch guard's per-action answer, the missing-pair line.
    from inference.agent import sim_check
    from inference.agent.tool_agent import _ascii_frame_view_payload, _ascii_history_view_payload

    def shifted(grid, dc):
        rows = [[0] * 64 for _ in range(64)]
        for r in range(64):
            for c in range(64):
                if grid[r][c] and 0 <= c + dc < 64:
                    rows[r][c + dc] = grid[r][c]
        return tuple(tuple(r) for r in rows)

    s0 = block(30, 30, 4, 4, 9)
    s1, s2, s3 = shifted(s0, 6), shifted(s0, 12), shifted(s0, 12)  # third step: RIGHT blocked at a wall
    sim_hist = [HistoryEntry("", Frame(s0, 0, 1)),
                HistoryEntry("RIGHT", Frame(s1, 1, 1), {"executed": True, "action_num": 1}),
                HistoryEntry("RIGHT", Frame(s2, 2, 1), {"executed": True, "action_num": 2}),
                HistoryEntry("RIGHT", Frame(s3, 3, 1), {"executed": True, "action_num": 3}),
                HistoryEntry("MOUSE(row=5, col=7)", Frame(s3, 4, 1), {"executed": True, "action_num": 4}),
                HistoryEntry("UNDO", Frame(s3, 5, 1), {"executed": True, "action_num": 5})]
    observe_src = """def observe(frame):
    for r, row in enumerate(frame.grid):
        for c, v in enumerate(row):
            if v == 9:
                return {'pos': (r, c)}
    return None
"""
    step_src = """def step(state, action):
    r, c = state['pos']
    if action == 'RIGHT':
        return {'pos': [r, c + 6]}
    if action == 'LEFT':
        return {'pos': [r, c - 6]}
    if isinstance(action, dict) and action.get('action') == 'MOUSE':
        return state
    return None
"""
    sim_state = {"current_frame": _ascii_frame_view_payload(sim_hist[-1].frame),
                 "history": _ascii_history_view_payload(sim_hist), "valid_actions": ["RIGHT", "LEFT", "MOUSE"],
                 "last_action_call_result": {}}
    res = sim_check.evaluate(run_sandboxed_python, sim_state, [observe_src, step_src])
    # UNDO is not a state transition a step(state, action) can model: skipped; lists equal tuples
    assert res.get("checked") == 4 and len(res.get("mismatches") or []) == 1, res
    assert res["mismatches"][0]["action_num"] == 3, res
    line = sim_check.lines(res)[0]
    assert ("equals observe(after) on 3 of 4 recorded transitions of this level; mismatches at action 3 "
            "(RIGHT: state['pos'][1]: predicted 48, observed 42)") in line, line
    blind_src = "def observe(frame):\n    return 0\n"
    blind = sim_check.evaluate(run_sandboxed_python, sim_state, [blind_src, "def step(state, action):\n    return state\n"])
    blind_line = sim_check.lines(blind)[0]
    assert "on all 4 recorded transitions" in blind_line and "changed the board interior (actions 1, 2)" in blind_line, blind_line
    bad = sim_check.evaluate(run_sandboxed_python, sim_state, [observe_src, "def step(state, action):\n    return state['x']\n"])
    assert "failed on action 1 (RIGHT): KeyError" in sim_check.lines(bad)[0], bad
    # search() plans over the same pair and carries the replay verdict; goal from the current position (30,42)
    found = run_sandboxed_python(
        code="r = search(lambda s: s is not None and s['pos'][1] == 30)\nprint(r['status'], r['plan'], r['sim_check'][:60])",
        timeout_seconds=20, initial_state=sim_state, action_handler=lambda a, stale_after=None: None,
        kept_functions=[observe_src, step_src])
    # the pair mispredicts action 3 (the wall), so the plan comes back marked unverified
    assert found.get("stdout", "").startswith("found_unverified ['LEFT', 'LEFT'] Sim check: step(observe(before)"), found
    wall_step = step_src.replace("return {'pos': [r, c + 6]}", "return {'pos': [r, min(c + 6, 42)]}")
    verified = run_sandboxed_python(
        code="r = search(lambda s: s['pos'][1] == 30)\nprint(r['status'], r['verified'], r['plan'], r['blind'])",
        timeout_seconds=20, initial_state=sim_state, action_handler=lambda a, stale_after=None: None,
        kept_functions=[observe_src, wall_step])
    assert verified.get("stdout", "").strip() == "found True ['LEFT', 'LEFT'] []", verified
    # a step that edits its input in place: branches must not leak into each other (r15 review)
    from inference.agent import sim_contract

    def inplace(state, act):
        state["x" if act == "RIGHT" else "y"] += 1
        return state

    assert sim_contract.sim_search(inplace, {"x": 0, "y": 0}, lambda s: s == {"x": 1, "y": 1},
                                   ["RIGHT", "UP"])["plan"] == ["RIGHT", "UP"]
    def gen():
        yield 1

    uncopyable = sim_contract.sim_search(inplace, {"x": 0, "y": 0, "g": gen()}, lambda s: False, ["RIGHT"])
    assert uncopyable["status"] == "error" and "cannot be copied" in uncopyable["error"], uncopyable  # no silent fallback
    inplace_src = wall_step.replace("return {'pos': [r, min(c + 6, 42)]}",
                                    "state['pos'] = [r, min(c + 6, 42)]\n        return state")
    clean = sim_check.evaluate(run_sandboxed_python, sim_state, [observe_src, inplace_src])
    assert clean.get("checked") == 4 and not clean.get("mismatches") and not clean.get("blind"), clean

    def slow_step(state, act):
        t0 = time.monotonic()
        while time.monotonic() - t0 < 0.02:
            pass
        return (state[0] + 1, state[1])

    t0 = time.monotonic()
    timed = sim_contract.sim_search(slow_step, (0, 0), lambda s: False, ["A", "B", "C", "D"], time_limit=0.2)
    assert timed["status"] == "time_limit" and time.monotonic() - t0 < 0.6, timed
    none = run_sandboxed_python(code="print(search(lambda s: False, max_depth=3)['status'])", timeout_seconds=20,
                                initial_state=sim_state, action_handler=lambda a, stale_after=None: None,
                                kept_functions=[observe_src, step_src])
    assert none.get("stdout", "").strip() == "max_depth", none
    missing = run_sandboxed_python(code="search(lambda s: True)", timeout_seconds=20, initial_state=sim_state,
                                   action_handler=lambda a, stale_after=None: None, kept_functions=[])
    assert "NameError" in missing.get("error", "") and "observe(frame) and step(state, action)" in missing["error"], missing

    # batch guard: the sandbox answers the solver's per-action query with the model's own pair
    answers = []

    def guarded_handler(actions, stale_after=None, sim_query=None):
        assert sim_query is not None, "sim_guard was not requested for a batch with observe/step defined"
        frame = lambda g: {"grid": [list(row) for row in g], "level": 1, "step": 0}
        answers.append(sim_query(1, frame(s0), frame(s1)))   # RIGHT moved 6: predicted
        answers.append(sim_query(2, frame(s2), frame(s3)))   # RIGHT blocked: mispredicted
        return {"action_result": {"executed": True, "executed_actions": ["RIGHT", "RIGHT"],
                                  "stop_reason": "sim_mismatch"}, "state": sim_state}

    guarded = run_sandboxed_python(code="r = action(['RIGHT', 'RIGHT', 'RIGHT'])\nprint(r['stop_reason'])",
                                   timeout_seconds=20, initial_state=sim_state, action_handler=guarded_handler,
                                   kept_functions=[observe_src, step_src], sim_guard=True)
    assert guarded.get("stdout", "").strip() == "sim_mismatch", guarded
    assert answers[0].get("ok") is True and answers[1].get("ok") is False, answers
    assert answers[1].get("detail") == "state['pos'][1]: predicted 48, observed 42", answers
    raising = []

    def raising_handler(actions, stale_after=None, sim_query=None):
        frame = lambda g: {"grid": [list(row) for row in g], "level": 1, "step": 0}
        raising.append(sim_query(1, frame(_grid()), frame(s1)))  # nothing of colour 9 before: step(None) raises
        return {"action_result": {"executed": True, "stop_reason": "sim_guard_error"}, "state": sim_state}

    run_sandboxed_python(code="action(['RIGHT', 'RIGHT'])", timeout_seconds=20, initial_state=sim_state,
                         action_handler=raising_handler, kept_functions=[observe_src, step_src], sim_guard=True)
    assert raising and raising[0].get("ok") is None and "TypeError" in (raising[0].get("error") or ""), raising
    unguarded = run_sandboxed_python(code="action(['RIGHT', 'RIGHT'])", timeout_seconds=20, initial_state=sim_state,
                                     action_handler=lambda a, stale_after=None: {"action_result": {"executed": True},
                                                                                 "state": sim_state},
                                     kept_functions=[], sim_guard=True)  # no pair defined: no query, old signature
    assert not unguarded.get("error"), unguarded
    # the opener's cache is keyed on every retained source: a changed helper is replayed again (r15 review)
    import types
    from inference.agent.tool_agent import ToolAgent
    helper_v1 = "def col_of(frame):\n    return 0\n"
    helper_obs = "def observe(frame):\n    return col_of(frame)\n"
    fake = types.SimpleNamespace(_sim_check_cache=None, _sim_nudged=set())
    fake._kept_functions = {"col_of": helper_v1, "observe": helper_obs, "step": "def step(state, action):\n    return state\n"}
    fake._retained_sources = lambda: list(fake._kept_functions.values())
    with patch.dict(os.environ, {"ARC3_PERSISTENT_FUNCTIONS": "1"}):
        first_line = ToolAgent._sim_check_lines(fake, sim_hist, sim_hist[-1].frame)
        fake._kept_functions["col_of"] = "def col_of(frame):\n    return frame.step\n"
        second_line = ToolAgent._sim_check_lines(fake, sim_hist, sim_hist[-1].frame)
    assert "on all 4" in first_line[0] and "on 0 of 4" in second_line[0], (first_line, second_line)
    nudged: set = set()
    assert sim_check.missing_lines(["observe", "step"], actions_on_level=3, level=2, nudged=nudged) == []
    first = sim_check.missing_lines(["observe", "step"], actions_on_level=9, level=2, nudged=nudged)
    assert first and first[0].startswith("Sim check: no `observe(frame)` and `step(state, action)` are retained"), first
    assert sim_check.missing_lines(["observe", "step"], actions_on_level=12, level=2, nudged=nudged) == []
    assert "`step` is retained but `observe` is not" in sim_check.missing_lines(
        ["observe"], actions_on_level=9, level=3, nudged=nudged)[0]
    grid_out = run_sandboxed_python(code="print(current_frame.grid[30][42], history[-1].grid[30][30], history[0].frame.grid[30][30])",
                                    timeout_seconds=5, initial_state=sim_state,
                                    action_handler=lambda a, stale_after=None: None, kept_functions=[])
    assert grid_out.get("stdout", "").strip().split() == ["9", "0", "9"], grid_out  # .grid: colour indices, forwarded on history entries
    checks.append("simulator contract: observe/step replayed per field, blind observe, copies for step, search (verified / found_unverified, clock), batch-guard answers incl. errors, cache over all retained sources, missing-pair line")

    # 15. Death context (r14): the fatal animation's scroll, tracks and contact; the moment-of-death image under a scroll.
    from inference.agent import death_context
    from inference.agent.tool_agent import ToolAgent, GAME_OVER_USER_PROMPT

    def dense_world():
        g = [[10] * 64 for _ in range(64)]  # S water
        for r in range(64):
            for c in range(64):
                if r < 5 or r > 58 or c < 5 or c > 58 or (r % 7 == 0 and c % 5 == 0):
                    g[r][c] = 5  # B walls with texture, so a shift is detectable
        for r in range(13, 17):
            for c in range(20, 23):
                g[r][c] = 15  # p 4x3 spike
        return g

    def with_player(g, top):
        g = [list(r) for r in g]
        for r in range(top, top + 4):
            for c in range(20, 22):
                g[r][c] = 9  # b 4x2
        return g

    def shifted_down(g, dy):
        out = [[5] * 64 for _ in range(64)]
        for r in range(64):
            if 0 <= r + dy < 64:
                out[r + dy] = list(g[r])
        return out

    base = dense_world()
    chain = [with_player(base, 37 - 2 * k) for k in range(10)] + [base]   # rises 2 rows per frame, vanishes under the spike
    rep = death_context.report(chain, prefer_colors=["b"], level_offset=(48, 0))
    assert not rep["scrolled"], rep["lines"]
    assert rep["lines"][0].startswith("b 4x2 moved (37,20)->(35,20)") and rep["lines"][0].endswith("then disappeared."), rep["lines"]
    assert "Where the b 4x2 disappeared (19,20)" in rep["lines"][1] and "above" in rep["lines"][1] and "p 4x3 at (13,20)" in rep["lines"][1], rep["lines"][1]
    assert rep["lines"][2].startswith("In the level's starting view") and "(-29,20)" in rep["lines"][2], rep["lines"][2]
    # follow-cam: the view pans 2 rows per frame with the player, then the death frame is the panned world
    pan = [shifted_down(f, 2 * k) for k, f in enumerate(chain)]
    rep2 = death_context.report(pan, prefer_colors=["b"])
    assert rep2["scrolled"] and rep2["shift"] == (20, 0), rep2["shift"]
    assert rep2["lines"][0].startswith("The view scrolled 20 rows down") and "while the b 4x2 was still moving" in rep2["lines"][0], rep2["lines"][0]
    assert "moved (37,20)->(35,20)" in rep2["lines"][1] and "[death frame: (39,20)]" in rep2["lines"][2] and "p 4x3 at (13,20) [death frame: (33,20)]" in rep2["lines"][2], rep2["lines"]
    assert rep2["moment_frame"] is not None and rep2["moment_label"].startswith("Moment of death image: frame 9"), rep2["moment_label"]
    assert death_context.report([])["lines"] == [] and death_context.report([base, base, base])["lines"] == [
        "No object moved or disappeared during the fatal animation."]
    two = death_context.report([with_player(base, 19), base], prefer_colors=["b"])["lines"]
    assert len(two) == 1 and two[0].startswith("b 4x2 is gone in the final frame, last seen at (19,20)") and "without having moved" in two[0] and "p 4x3 at (13,20)" in two[0], two
    assert death_context.recent_movers([(with_player(base, 37), with_player(base, 35))]) == ["b"]
    # agent level: history ending in the automatic RESET; the chain held by the agent; describe_animation dropped under a scroll
    class _Holder:
        _last_animation_chain = pan
        _last_action_gameplay_changed = None
        _pending_gameover_images = []
        _death_context = ToolAgent._death_context
        _game_over_lines = ToolAgent._game_over_lines
        _prepare_gameover_images = ToolAgent._prepare_gameover_images
        _append_fatal_animation_image = lambda self, summary: None
    hist15 = [HistoryEntry("", Frame(tuple(map(tuple, chain[0])), 0, 2)),
              HistoryEntry("RIGHT", Frame(tuple(map(tuple, chain[0])), 1, 2), {"executed": True, "action_num": 1, "view_offset": [48, 0]}),
              HistoryEntry("MOUSE(row=33, col=21)", Frame(tuple(map(tuple, pan[-1])), 2, 2), {"executed": True, "action_num": 2, "game_over": True}),
              HistoryEntry("RESET", Frame(tuple(map(tuple, chain[0])), 3, 2), {"executed": True, "action_num": 3, "automatic": True})]
    summary15 = {"game_over": True, "animation": {"frames": 10, "transient_pixels": 1700, "transient_bbox": [0, 0, 63, 63]},
                 "animation_action": "MOUSE(row=33, col=21)", "fatal_action": "MOUSE(row=33, col=21)"}
    holder = _Holder()
    with patch.dict(os.environ, {"ARC3_GAMEOVER_DIFF_IMAGE": "1", "MULTIMODAL_CONTEXT": "current_grid"}):
        go = holder._game_over_lines(hist15, summary15)
    labels = [label for label, _ in holder._pending_gameover_images]
    assert len(labels) == 2 and labels[0].startswith("Death frame image") and labels[1].startswith("Moment of death image: frame 9"), labels
    assert go[0] == GAME_OVER_USER_PROMPT and go[1].startswith("The view scrolled 20 rows down"), go[:2]
    assert not any("changed and changed back" in l for l in go), "the transient-cell sentence is dropped under a scroll"
    assert any("In the level's starting view" in l for l in go), go
    with patch.dict(os.environ, {"ARC3_DEATH_CONTEXT": "0"}):
        assert holder._death_context(hist15) is None
    checks.append("death context: tracks, contact, scroll with death-frame coordinates, moment image, kill switch")
    return {"status": "PASS", "passed": len(checks), "checks": checks}


if __name__ == "__main__":
    print(json.dumps(self_test(), indent=2))
