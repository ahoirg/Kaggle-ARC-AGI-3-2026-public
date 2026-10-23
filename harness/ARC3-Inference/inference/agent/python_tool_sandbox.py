"""Lightweight isolated runner for analyzer Python tool calls."""
from __future__ import annotations

import inspect
import json
import logging
import os
import queue
import signal
import subprocess
import sys
import tempfile
import threading
import textwrap
import time
from typing import Any, Callable

from inference.utils import frame_diff as _frame_diff_module
from inference.utils import motion as _motion_module
from inference.utils import segmentation as _segmentation
from inference.utils import retained_functions as _retained_functions
from inference.agent import sim_contract as _sim_contract
from inference.utils.grid_utils import ARC_COLOR_CHARS


_SANDBOX_BOOTSTRAP = textwrap.dedent(
    r"""
    import ast
    import builtins
    import contextlib
    import io
    import json
    import os
    import sys
    import traceback

    try:
        import resource
    except ImportError:  # pragma: no cover
        resource = None

    COLOR_CHARS = ""

    __SEGMENTATION_SOURCE__

    __FRAME_DIFF_SOURCE__

    __MOTION_SOURCE__

    __RETAINED_FUNCTIONS_SOURCE__

    __SIM_CONTRACT_SOURCE__

    HOST_STDOUT = sys.stdout

    SAFE_MODULES = {
        "bisect",
        "collections",
        "copy",
        # pure-Python sequence comparison, no I/O. Measured across 32
        # transcripts it was imported 9 times and USED none of them - the model
        # reaches for it on "compare two things" and then writes the comparison
        # by hand. Allowing it therefore buys robustness, not capability: a
        # stray import line no longer costs the whole snippet.
        "difflib",
        "fractions",
        "functools",
        "heapq",
        "itertools",
        "json",
        "math",
        "operator",
        "random",
        "re",
        "statistics",
        "string",
    }
    SAFE_BUILTINS = {
        # Exception types. Only four were present, so anything narrower than
        # `except Exception` failed at the except clause and took the snippet
        # with it. These are inert classes and expose nothing; SystemExit and
        # KeyboardInterrupt stay out because the sandbox uses them for its own
        # timeout and shutdown handling.
        "ArithmeticError",
        "AssertionError",
        "AttributeError",
        "EOFError",
        "IndexError",
        "KeyError",
        "LookupError",
        "NameError",
        "NotImplementedError",
        "OverflowError",
        "RecursionError",
        "StopIteration",
        "UnboundLocalError",
        "ZeroDivisionError",
        "ImportError",
        "__build_class__",
        "abs",
        "all",
        "any",
        "ascii",
        "bin",
        "bool",
        "bytearray",
        "bytes",
        "callable",
        "chr",
        "complex",
        "dict",
        "dir",
        "divmod",
        "enumerate",
        "Exception",
        "filter",
        "float",
        "format",
        "frozenset",
        "getattr",
        "hasattr",
        "hash",
        "hex",
        "int",
        "isinstance",
        "issubclass",
        "iter",
        "len",
        "list",
        "map",
        "max",
        "min",
        "next",
        "object",
        "oct",
        "ord",
        "pow",
        "print",
        "range",
        "repr",
        "reversed",
        "round",
        "set",
        "setattr",
        "slice",
        "staticmethod",
        "classmethod",
        "property",
        "super",
        "id",
        "sorted",
        "str",
        "sum",
        "tuple",
        "TypeError",
        "type",
        "ValueError",
        "RuntimeError",
        "zip",
    }


    def _send(payload):
        HOST_STDOUT.write(json.dumps(payload, ensure_ascii=False) + "\n")
        HOST_STDOUT.flush()


    def _recv():
        line = sys.stdin.readline()
        if not line:
            raise EOFError("sandbox input closed")
        return json.loads(line)


    class FrameView:
        def __init__(self, *, ascii, step, level, shape, grid):
            self.ascii = ascii
            self.step = step
            self.level = level
            self.shape = tuple(shape)
            self._grid = grid
            self._segmentation = None

        @property
        def segmentation(self):
            if self._segmentation is None:
                self._segmentation = segment_layer(self._grid, COLOR_CHARS)
            return self._segmentation

        @property
        def frame(self):
            # `history[i].frame.frame` and `current_frame.frame` both appeared in snippets
            return self

        @property
        def grid(self):
            # colour indices, row 0 first; a copy, so a snippet cannot alter the record
            return [list(row) for row in self._grid]

        def __str__(self):
            rows, cols = self.shape
            return f"AsciiFrameView(level={self.level}, step={self.step}, shape={rows}x{cols})"

        __repr__ = __str__


    class HistoryEntryView:
        def __init__(self, *, action, frame, result=None):
            self.action = action
            self.frame = frame
            self.result = dict(result) if isinstance(result, dict) else {}

        def __getattr__(self, name):
            # the model treats a history entry like its frame (`.ascii`, `.level`, `.step`,
            # `.segmentation`: 18 AttributeErrors in r4-r7); delegate instead of failing
            if name.startswith("_"):
                raise AttributeError(name)
            frame = self.__dict__.get("frame")
            if frame is not None and hasattr(frame, name):
                return getattr(frame, name)
            raise AttributeError(f"'HistoryEntryView' has no attribute {name!r} (it has .action, .frame, .result; "
                                 "frame attributes are forwarded)")

        def __str__(self):
            return f"AsciiHistoryEntryView(action={self.action!r}, frame={self.frame})"

        __repr__ = __str__


    class TransitionView:
        def __init__(self, *, action, before_frame, after_frame, result):
            self.action = action
            self.before_frame = before_frame
            self.after_frame = after_frame
            self.frame = after_frame
            self.result = dict(result) if isinstance(result, dict) else {}

        def __str__(self):
            return (
                "ActionTransitionView("
                f"action={self.action!r}, "
                f"before_frame={self.before_frame}, "
                f"after_frame={self.after_frame})"
            )

        __repr__ = __str__


    def _frame_from_payload(payload):
        if not isinstance(payload, dict):
            return None
        return FrameView(
            ascii=str(payload.get("ascii", "")),
            step=int(payload.get("step", 0)),
            level=int(payload.get("level", 0)),
            shape=payload.get("shape", [0, 0]),
            grid=payload.get("grid", []),
        )


    def _history_from_payload(payload):
        items = []
        for entry in payload or []:
            if not isinstance(entry, dict):
                continue
            items.append(
                HistoryEntryView(
                    action=str(entry.get("action", "")),
                    frame=_frame_from_payload(entry.get("frame")),
                    result=entry.get("result"),
                )
            )
        return items


    def _transitions_from_history(history):
        transitions = []
        for index, entry in enumerate(history):
            action = str(getattr(entry, "action", "") or "").strip()
            if not action:
                continue
            before_frame = history[index - 1].frame if index > 0 else None
            transitions.append(
                TransitionView(
                    action=action,
                    before_frame=before_frame,
                    after_frame=entry.frame,
                    result=entry.result,
                )
            )
        return transitions


    def _json_safe(value):
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, dict):
            return {str(key): _json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [_json_safe(item) for item in value]
        return str(value)


    def _sanitize_exception(exc):
        extracted = traceback.extract_tb(exc.__traceback__)
        user_frames = [frame for frame in extracted if frame.filename == "<python_tool>"]
        lines = ["Traceback (most recent call last):"]
        for frame in user_frames or extracted[-1:]:
            lines.append(f'  File "<python_tool>", line {frame.lineno}, in {frame.name}')
        lines.append(f"{exc.__class__.__name__}: {exc}")
        return "\n".join(lines)


    class _TimeStub:
        # `import time` was the most frequent blocked import (20 in r4-r7); clocks are harmless, the
        # rest of the module (sleep) is not. Only the three clocks exist here.
        import time as _t
        monotonic = staticmethod(_t.monotonic)
        perf_counter = staticmethod(_t.perf_counter)
        time = staticmethod(_t.time)
        __name__ = "time"

        def __getattr__(self, name):
            raise AttributeError(f"time.{name} is not available in the sandbox (only monotonic, perf_counter, time).")

    _TIME_STUB = _TimeStub()

    def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
        root = str(name or "").split(".", 1)[0]
        if root == "time":
            return _TIME_STUB
        if root not in SAFE_MODULES:
            raise ImportError(f"Module '{name}' is not allowed in the sandbox. Importable: "
                              + ", ".join(sorted(SAFE_MODULES)) + ", and time (clocks only).")
        return builtins.__import__(name, globals, locals, fromlist, level)


    def _set_limits(timeout_seconds):
        if resource is None:
            return
        cpu_limit = max(1, int(timeout_seconds)) + 1
        for limit, value in (
            (getattr(resource, "RLIMIT_CPU", None), cpu_limit),
            (getattr(resource, "RLIMIT_FSIZE", None), 1_000_000),
            (getattr(resource, "RLIMIT_NOFILE", None), 32),
        ):
            if limit is None:
                continue
            try:
                resource.setrlimit(limit, (value, value))
            except (OSError, ValueError):
                pass


    def _normalize_actions(actions):
        if isinstance(actions, str):
            items = [actions]
        elif isinstance(actions, dict):
            items = [actions]
        elif isinstance(actions, (list, tuple)):
            items = list(actions)
        else:
            raise TypeError(
                "action(actions) expects a string, an action object, or a list of action strings/objects."
            )
        if not items:
            raise ValueError("action(actions) requires at least one action.")

        normalized = []
        for index, item in enumerate(items, start=1):
            if isinstance(item, str):
                action_name = item.strip()
                if not action_name:
                    raise ValueError(f"Action {index} is empty.")
                normalized.append({"action": action_name})
                continue
            if isinstance(item, dict):
                action_name = str(item.get("action", "")).strip()
                if not action_name:
                    raise ValueError(f"Action {index} is missing an `action` field.")
                entry = {"action": action_name}
                if action_name.upper() == "MOUSE" and ("x" in item or "y" in item):
                    raise ValueError(
                        f"Action {index} uses legacy MOUSE x/y fields; use row and col."
                    )
                if "row" in item:
                    entry["row"] = item.get("row")
                if "col" in item:
                    entry["col"] = item.get("col")
                normalized.append(entry)
                continue
            raise TypeError(f"Action {index} must be a string or a dict.")
        return normalized


    def main():
        initial = _recv()
        global COLOR_CHARS
        COLOR_CHARS = str(initial.get("color_chars") or "")
        timeout_seconds = max(1, int(initial.get("timeout_seconds", 30)))
        sandbox_cwd = str(initial.get("sandbox_cwd", "")).strip()
        if sandbox_cwd:
            os.chdir(sandbox_cwd)
        _set_limits(timeout_seconds)

        action_results = []
        execution_error_fix = bool(initial.get("action_execution_error_fix", False))
        action_execution_errors = []
        action_attempt = 0

        class ActionExecutionError(BaseException):
            pass  # Tool boundary failure, not an observed game no-op.

        def _execution_failure(status, detail):
            failure = {"attempt_in_tool": action_attempt, "status": status,
                       "executed": False if status == "invalid_request" else None,
                       "no_op": None, "detail": detail}
            action_execution_errors.append(failure)
            raise ActionExecutionError(detail)

        stdout = io.StringIO()
        runtime_globals = {
            "__builtins__": {
                name: getattr(builtins, name)
                for name in SAFE_BUILTINS
            },
            "result": None,
            "__name__": "python_tool",
        }
        runtime_globals["__builtins__"]["__import__"] = _safe_import
        # pre-imported, so a snippet that uses them without an import line does not die on a
        # NameError (json 8, collections/Counter 5 of the 103 NameErrors in r4-r7)
        import collections as _collections, itertools as _itertools, math as _math, re as _re
        runtime_globals.update({
            "json": json, "math": _math, "re": _re, "itertools": _itertools, "collections": _collections,
            "Counter": _collections.Counter, "defaultdict": _collections.defaultdict, "deque": _collections.deque,
        })
        _animation_views = []

        def frame_diff(before=None, after=None):
            # Object-level diff between two frames. Defaults: previous_frame -> current_frame.
            # Returns only what changed; unchanged objects are omitted.
            if before is None:
                before = runtime_globals.get("previous_frame")
            if after is None:
                after = runtime_globals.get("current_frame")
            if before is None or after is None:
                return {"error": "need two frames (no previous_frame available?)"}
            if before.shape != after.shape:
                return {"changed_cell_count": -1, "note": "frame shapes differ; full redraw",
                        "before_shape": before.shape, "after_shape": after.shape}
            return compute_frame_diff(
                before._grid,
                after._grid,
                before.segmentation.get("nodes", []),
                after.segmentation.get("nodes", []),
            )

        def motion(before=None, after=None):
            # r10: object-level motion phrases between two frames (defaults: previous_frame -> current_frame).
            # The same phrases the turn message shows; a list of strings, largest objects first.
            if before is None:
                before = runtime_globals.get("previous_frame")
            if after is None:
                after = runtime_globals.get("current_frame")
            if before is None or after is None:
                return ["need two frames (no previous_frame available?)"]
            return motion_lines(before._grid, after._grid, max_items=12)

        # r15 simulator contract (sim_contract): the model's observe(frame) + step(state, action),
        # checked against the record, searched over, and used by the batch guard below.
        _sim_border = int(initial.get("sim_border", 4) or 4)
        _sim_guard_enabled = bool(initial.get("sim_guard"))
        # the batch check spends the snippet's own time; it stops answering with 8 s left, and search()
        # leaves it 12 s, so a search followed by its plan in one call keeps the check
        _snippet_end = time.monotonic() + max(1, int(initial.get("timeout_seconds", 30)))
        _sim_guard_deadline = _snippet_end - 8

        def _sim_pair():
            observe_fn = runtime_globals.get(OBSERVE)
            step_fn = runtime_globals.get(STEP)
            if not callable(observe_fn) or not callable(step_fn):
                return None
            return observe_fn, step_fn

        def _sim_frame(payload):
            grid = [list(row) for row in (payload or {}).get("grid") or []]
            ascii_rows = "\n".join(
                "".join(COLOR_CHARS[v] if 0 <= v < len(COLOR_CHARS) else "?" for v in row) for row in grid)
            return FrameView(ascii=ascii_rows, step=int((payload or {}).get("step", 0)),
                             level=int((payload or {}).get("level", 0)),
                             shape=[len(grid), len(grid[0]) if grid else 0], grid=grid)

        def sim_check(limit=40):
            # step(observe(before), action) against observe(after) over this level's recorded transitions
            pair = _sim_pair()
            if pair is None:
                raise NameError("sim_check() replays your observe(frame) and step(state, action); define both first")
            frame = runtime_globals.get("current_frame")
            result = sim_replay(pair[0], pair[1], runtime_globals.get("transitions") or [], frame.level,
                                limit=limit, border=_sim_border, deadline=time.monotonic() + 10)
            result["summary"] = sim_summary(result)
            return result

        def search(goal, actions=None, start=None, max_depth=60, max_nodes=200000, time_limit=15):
            # breadth-first search over your step(state, action) from observe(current_frame) to goal(state)
            pair = _sim_pair()
            if pair is None:
                raise NameError("search() plans over your observe(frame) and step(state, action); define both first")
            if not callable(goal):
                raise TypeError("search(goal, ...): goal must be a function state -> bool")
            if actions is None:
                actions = [name for name in runtime_globals.get("valid_actions") or []
                           if name not in ("MOUSE", "UNDO", "RESET")]
                if not actions:
                    raise ValueError("search(): no key action is valid here; pass actions=[{'action': 'MOUSE', "
                                     "'row': r, 'col': c}, ...] or a function state -> list of actions")
            frame = runtime_globals.get("current_frame")
            # the record first: a plan is only as good as the pair it came from
            check = sim_replay(pair[0], pair[1], runtime_globals.get("transitions") or [], frame.level,
                               border=_sim_border, deadline=min(time.monotonic() + 5, _snippet_end - 12))
            verified = bool(check.get("checked")) and not check.get("mismatches") and not check.get("error") \
                and not check.get("timed_out")
            begin = pair[0](frame) if start is None else start
            budget = min(float(time_limit), _snippet_end - 12 - time.monotonic())
            out = sim_search(pair[1], begin, goal, actions, max_depth=max_depth, max_nodes=max_nodes,
                             time_limit=max(0.2, budget))
            if out.get("status") == "found" and not verified:
                out["status"] = "found_unverified"
            out["verified"] = verified
            out["blind"] = [b for b in (check.get("blind") or []) if b is not None]
            out["sim_check"] = sim_summary(check)
            return out

        def _sim_answer(query, requested):
            try:
                pair = _sim_pair()
                index = int(query.get("index", 0))
                if pair is None or not 1 <= index <= len(requested):
                    return {"ok": None}
                if time.monotonic() > _sim_guard_deadline:
                    return {"ok": None, "skip": "time"}
                return sim_guard_verdict(pair[0], pair[1], _sim_frame(query.get("before")), requested[index - 1],
                                         _sim_frame(query.get("after")))
            except Exception as exc:
                return {"ok": None, "error": (type(exc).__name__ + ": " + str(exc))[:160]}

        def _refresh_state(state_payload):
            current_frame = _frame_from_payload(state_payload.get("current_frame"))
            history = _history_from_payload(state_payload.get("history"))
            last_action_call_result = state_payload.get("last_action_call_result")
            action_result = (
                dict(last_action_call_result) if isinstance(last_action_call_result, dict) else {}
            )
            # action_echo is printed by action(), then removed from its return value.
            # Keep the exposed call result identical across subsequent inspections.
            action_result.pop("action_echo", None)
            transitions = _transitions_from_history(history)
            last_transition = transitions[-1] if transitions else None

            runtime_globals["current_frame"] = current_frame
            runtime_globals["latest_frame"] = current_frame
            runtime_globals["history"] = history
            runtime_globals["transitions"] = transitions
            runtime_globals["last_transition"] = last_transition
            runtime_globals["previous_frame"] = (
                last_transition.before_frame if last_transition is not None else None
            )
            runtime_globals["last_action_frame"] = (
                last_transition.after_frame if last_transition is not None else None
            )
            runtime_globals["last_action"] = last_transition.action if last_transition is not None else None
            runtime_globals["frame_diff"] = frame_diff
            runtime_globals["motion"] = motion
            runtime_globals["sim_check"] = sim_check
            runtime_globals["search"] = search
            runtime_globals["valid_actions"] = [str(item) for item in state_payload.get("valid_actions", [])]
            runtime_globals["death_ledger"] = state_payload.get("death_ledger") or {"attempts": []}
            runtime_globals["last_action_call_result"] = action_result

        # Both refusals derive from BaseException, not Exception, so a
        # defensive `except Exception:` around action(...) - which models write
        # routinely - cannot swallow them. A refusal means the action did not
        # run at all, so the snippet's assumed state is already wrong and every
        # line after it rests on a false premise. One run treated 1500
        # consecutive refusals as proof that every cell on the board was inert.
        # StaleStateActionError below is deliberately catchable: it reports a
        # belief that MAY be stale rather than a verdict on the action.
        class KnownNoOpActionError(BaseException):
            pass  # raised INSTEAD of executing an action already proven inert here

        class KnownDeathActionError(BaseException):
            pass  # raised INSTEAD of executing an action that ended a prior attempt

        # Raised INSTEAD of executing an action this snippet has already run
        # from this exact board. Buggy loop code can spend a whole life bar
        # getting nowhere - a cursor that wraps turns RIGHT into a short cycle,
        # a mistaken `while pos < j: LEFT` walks away from its target forever -
        # and the terminal latch only stops that once the game actually ends,
        # measured at 20 to 30 wasted actions. BaseException like the two above:
        # the shape that produces these loops is `try: action(...) except
        # Exception:` inside a while, which would otherwise swallow the verdict
        # and keep going. No override slot: the snippet is not reading the
        # refusal, its code is running, and writing a new snippet IS the
        # response - the map is rebuilt per execution.
        class RepeatedActionInStateError(BaseException):
            pass

        # Raised when a PREVIOUS action in this snippet changed nothing, and the
        # snippet then tries to act again. Deliberately a plain Exception, not
        # BaseException: unlike the two refusals above this is a warning about
        # the snippet's beliefs rather than a verdict on the action, and a model
        # that has re-read current_frame may legitimately want to proceed.
        class StaleStateActionError(Exception):
            pass

        runtime_globals["KnownNoOpActionError"] = KnownNoOpActionError
        runtime_globals["KnownDeathActionError"] = KnownDeathActionError
        runtime_globals["StaleStateActionError"] = StaleStateActionError
        runtime_globals["RepeatedActionInStateError"] = RepeatedActionInStateError
        # armed when an action changes nothing; scoped to THIS snippet because
        # the closure is rebuilt per execution, so no reset path can be missed
        _stale_state: list[str] = []

        # Raised when a snippet tries to act after the attempt or level ended.
        # Derives from BaseException, not Exception, so a defensive
        # `try: action(...) except Exception:` around the call - which models
        # write routinely - cannot swallow it. Continuing to execute a path
        # computed against a board that no longer exists is never correct.
        class TerminalStateActionError(BaseException):
            pass

        runtime_globals["TerminalStateActionError"] = TerminalStateActionError
        # latched when an action reports a terminal result; scoped to THIS
        # snippet because these closures are rebuilt for every execution, so
        # no reset path can be missed and the next snippet always starts clear
        _terminal_state = []

        _TERMINAL_DETAIL = {
            "game_over": (
                "the previous action reported game_over. The level has been reset to its "
                "start, so the board you planned against is gone and this action was not "
                "executed. The next turn begins from the new state."
            ),
            "level_completed": (
                "the previous action reported level_completed. A new level has been loaded, "
                "so the board you planned against is gone and this action was not executed. "
                "The next turn begins from the new state."
            ),
            "run_complete": (
                "the previous action reported run_complete. The run is finished and this "
                "action was not executed."
            ),
        }

        def action(actions):
            nonlocal action_attempt
            action_attempt += 1
            if _terminal_state:
                raise TerminalStateActionError(_TERMINAL_DETAIL[_terminal_state[0]])
            try:
                normalized_actions = _normalize_actions(actions)
            except (TypeError, ValueError) as exc:
                if not execution_error_fix:
                    raise
                _execution_failure("invalid_request",
                    "ACTION NOT EXECUTED: invalid request before dispatch. "
                    + str(exc) + " No new game observation was produced. Repair the call; this is not a no-op.")
            try:
                _send({
                    "type": "action",
                    "actions": normalized_actions,
                    # the solver decides precedence: a known death or a known no-op
                    # is a sharper message than "your state may be stale", and it
                    # carries a concrete override
                    "stale_after": _stale_state[0] if _stale_state else None,
                    # the solver checks each executed action of the batch against step(observe(before))
                    "sim_guard": bool(_sim_guard_enabled and len(normalized_actions) > 1 and _sim_pair()),
                })
                reply = _recv()
                while reply.get("type") == "sim_query":
                    _send({"type": "sim_answer", **_json_safe(_sim_answer(reply, normalized_actions))})
                    reply = _recv()
                if reply.get("type") == "action_error":
                    raise RuntimeError(str(reply.get("error", "action failed")))
                if reply.get("type") != "action_result":
                    raise RuntimeError("Invalid action response from sandbox host.")
            except Exception:
                if not execution_error_fix:
                    raise
                _execution_failure("unknown",
                    "ACTION OUTCOME UNKNOWN: dispatch or response failed. This is not a no-op; "
                    "inspect current state and recorded transitions before retrying. Earlier actions may have executed.")
            action_result = reply.get("action_result") or {}
            if (execution_error_fix and action_result.get("error") and action_result.get("executed") is False
                    and not action_result.get("executed_actions") and not action_result.get("stop_reason")):
                _execution_failure("invalid_request",
                    "ACTION NOT EXECUTED: " + str(action_result.get("error")) + " Valid actions: "
                    + ", ".join(str(a) for a in action_result.get("valid_actions") or []) + "."
                    " No game action was spent; this is not a game observation.")
            action_results.append(action_result)
            _refresh_state(reply.get("state") or {})
            # Animation globals describe the previous action; reload them now.
            for reload_animation in _animation_views:
                reload_animation()
            if action_result.get("stop_reason") == "reset_not_first":
                raise RuntimeError(str(action_result.get("stop_detail")))
            if action_result.get("stop_reason") == "stale_state":
                raise StaleStateActionError(
                    str(action_result.get("stop_detail") or "stale state")
                )
            if action_result.get("stop_reason") == "known_death":
                raise KnownDeathActionError(
                    str(action_result.get("stop_detail") or "known fatal action was not executed.")
                )
            if action_result.get("stop_reason") == "repeated_action_in_state":
                raise RepeatedActionInStateError(
                    str(action_result.get("stop_detail") or "")
                    or "this action was already run from this board in this snippet"
                )
            if action_result.get("stop_reason") == "known_noop":
                # nothing executed and no action was spent; halt the snippet so
                # the model re-plans instead of continuing on a false premise
                raise KnownNoOpActionError(
                    str(action_result.get("stop_detail") or "known no-op action was not executed.")
                )
            for _flag in ("run_complete", "done", "level_completed", "game_over"):
                if action_result.get(_flag):
                    # `done` means the run was won; report it as run_complete
                    _terminal_state.append(
                        "run_complete" if _flag == "done" else _flag
                    )
                    break
            _echo = action_result.pop("action_echo", None)
            if _echo:
                # into the snippet's own stdout, so a loop can see what its
                # call did without waiting for the next turn's opener
                print(_echo)
            gameplay_changed = action_result.get("gameplay_changed")
            if gameplay_changed is True:
                # Interior change clears the inspection latch; it does not prove a plan correct.
                del _stale_state[:]
            # Arm only on a true no-op (nothing visible changed, not even transiently).
            # Border-only and animation-only results are inconclusive, not stale.
            if action_result.get("no_op") is True and action_result.get("executed"):
                _executed = action_result.get("executed_actions")
                _name = (
                    _executed[-1]
                    if isinstance(_executed, list) and _executed
                    else action_result.get("action_display")
                    or action_result.get("action_name")
                    or "?"
                )
                # Arm rather than raise. The final interior was unchanged;
                # edge or animation effects may exist. The snippet may be looking
                # for - it printed the board before, and wants to print it
                # after. Killing it here throws away the second half of the
                # measurement AND the batch stop's own trace, which is computed
                # and then discarded. So let the snippet finish observing, and
                # stop it only if it tries to ACT on the stale belief.
                if not _stale_state:
                    _stale_state.append(str(_name))
            return action_result

        def _animation_fetch(kind):
            _send({"type": "animation", "request": {"kind": kind}})
            reply = _recv()
            if reply.get("type") == "animation_error":
                raise RuntimeError(str(reply.get("error", "animation failed")))
            if reply.get("type") != "animation_result":
                raise RuntimeError("Invalid animation response from sandbox host.")
            payload = reply.get("animation")
            if kind != "frames":
                return payload or {}
            return [
                FrameView(
                    ascii=item.get("ascii", ""),
                    step=item.get("step", 0),
                    level=item.get("level", 0),
                    shape=item.get("shape", (0, 0)),
                    grid=item.get("grid", ()),
                )
                for item in (payload or [])
            ]

        runtime_globals["action"] = action
        # Animation globals are plain list/dict values. They are fetched from the
        # host only when the snippet (or a retained function) mentions them, and
        # re-fetched after every action() call so they always describe the most
        # recent executed action.
        _animation_needed = bool(initial.get("animation_enabled")) and (
            "last_animation" in str(initial.get("code", ""))
            or any("last_animation" in str(src) for src in (initial.get("kept_functions") or []))
        )

        def _load_animation_globals():
            frames, timeline, tracks = [], {}, []
            if _animation_needed:
                try:
                    frames = list(_animation_fetch("frames") or [])
                except Exception:
                    frames = []
                try:
                    fetched = _animation_fetch("timeline")
                    timeline = dict(fetched) if isinstance(fetched, dict) else {}
                except Exception as exc:
                    timeline = {"error": "animation timeline unavailable: " + str(exc)}
                try:
                    # r10: object tracks through the animation (what moved where, what flashed)
                    tracks = list(animation_tracks([f._grid for f in frames], max_tracks=8)) if len(frames) >= 3 else []
                except Exception:
                    tracks = []
            runtime_globals["last_animation_frames"] = frames
            runtime_globals["last_animation_timeline"] = timeline
            runtime_globals["last_animation_tracks"] = tracks

        _animation_views.append(_load_animation_globals)

        _refresh_state(initial.get("state") or {})
        for reload_animation in _animation_views:
            reload_animation()

        _persistence_enabled = bool(initial.get("persistence_enabled"))
        _retention_options = {
            "retain_imports": bool(initial.get("retain_imports")),
            "repair_hints": bool(initial.get("repair_hints")),
        }
        _kept_pool = {}
        _retention_rejected = []
        _available_names = set(runtime_globals) | set(runtime_globals["__builtins__"])
        if _persistence_enabled:
            _before_restore = dict(runtime_globals)
            try:
                _kept_pool, _retention_rejected = restore_functions(
                    initial.get("kept_functions") or [], runtime_globals, _available_names,
                    **_retention_options)
            except Exception:
                # Optional cache maintenance must never prevent this snippet running.
                runtime_globals.clear()
                runtime_globals.update(_before_restore)
                _kept_pool = {}
                _retention_rejected = [{"name": "all", "reason": "restoration failed; redefine needed functions"}]

        def _collect_retention_payload():
            if not _persistence_enabled:
                return {}
            try:
                kept, rejected = collect_functions(
                    _code_text, _kept_pool, runtime_globals, _available_names,
                    **_retention_options)
                return {
                    "keepable_functions": [{"name": n, "source": src} for n, src in kept.items()],
                    "retention_rejected": _retention_rejected + rejected,
                }
            except Exception:
                # Bookkeeping failures may drop helpers, but must never replace
                # the snippet's original error/output or replay its actions.
                return {"keepable_functions": [], "retention_rejected": [
                    {"name": "all", "reason": "retention failed; redefine needed functions"}]}

        try:
            _code_text = str(initial.get("code", ""))
            _tree = ast.parse(_code_text, "<python_tool>")
            _last_expr = None
            if _tree.body and isinstance(_tree.body[-1], ast.Expr):
                _last_expr = ast.Expression(_tree.body[-1].value)
                ast.copy_location(_last_expr, _tree.body[-1])
                _tree.body = _tree.body[:-1]
            with contextlib.redirect_stdout(stdout):
                if _tree.body:
                    exec(compile(_tree, "<python_tool>", "exec"), runtime_globals, runtime_globals)
                if _last_expr is not None:
                    _value = eval(compile(_last_expr, "<python_tool>", "eval"), runtime_globals, runtime_globals)
                    if _value is not None:
                        print(repr(_value))
                        if runtime_globals.get("result") is None:
                            runtime_globals["result"] = _value
            _send(
                {
                    "type": "final",
                    **_collect_retention_payload(),
                    "stdout": stdout.getvalue(),
                    "result": _json_safe(runtime_globals.get("result")),
                    "action_results": _json_safe(action_results),
                    **({"action_execution_errors": action_execution_errors} if execution_error_fix else {}),
                }
            )
        except (
            ActionExecutionError,
            KnownNoOpActionError,
            RepeatedActionInStateError,
            KnownDeathActionError,
            TerminalStateActionError,
            Exception,
        ) as exc:
            # The three refusals derive from BaseException so a model's own
            # `except Exception:` cannot swallow them - but THIS handler is the
            # same construct, so without naming them explicitly they escaped it
            # too, killed the child process, and the host reported a bare
            # "Sandbox process exited unexpectedly". The refusal messages - the
            # action name, that nothing was spent, how to override - were built
            # and then never delivered. They stop the snippet either way; the
            # point is that the model is told why.
            _send(
                {
                    "type": "error",
                    **_collect_retention_payload(),
                    "error": _sanitize_exception(exc),
                    "stdout": stdout.getvalue(),
                    "action_results": _json_safe(action_results),
                    **({"action_execution_errors": action_execution_errors} if execution_error_fix else {}),
                }
            )


    if __name__ == "__main__":
        main()
    """
).replace("__RETAINED_FUNCTIONS_SOURCE__\n", inspect.getsource(_retained_functions)).replace(
    "__SEGMENTATION_SOURCE__\n", inspect.getsource(_segmentation)
).replace(
    "__FRAME_DIFF_SOURCE__\n", inspect.getsource(_frame_diff_module)
).replace(
    "__MOTION_SOURCE__\n", inspect.getsource(_motion_module)
).replace(
    "__SIM_CONTRACT_SOURCE__\n", inspect.getsource(_sim_contract)
)


log = logging.getLogger(__name__)


def _sanitize_host_error_text(text: str) -> str:
    """What the MODEL is told when the sandbox process dies.

    Deliberately fixed: the child's traceback carries host paths and internals
    that have no business in the model's context. But the text was previously
    read and thrown away entirely, so a crash left no trace anywhere - the one
    thing that knows what went wrong said nothing. It now goes to the harness
    log instead, where it is useful and nobody is prompted by it.
    """
    detail = str(text or "").strip()
    if detail:
        log.warning(
            "sandbox process died; stderr follows:\n%s",
            detail[-4000:],
        )
    return "Sandbox process exited unexpectedly."


def _sandbox_env() -> dict[str, str]:
    return {
        "PYTHONUNBUFFERED": "1",
        "PYTHONIOENCODING": "utf-8",
        "PYTHONDONTWRITEBYTECODE": "1",
        "HOME": "/tmp",
        "TMPDIR": "/tmp",
        "PATH": os.environ.get("PATH", ""),
    }


def _send_json_line(handle: Any, payload: dict[str, Any]) -> None:
    handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    handle.flush()


def _kill_process_group(process: subprocess.Popen[str]) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except OSError:
        try:
            process.kill()
        except OSError:
            pass


def _wait_for_process_exit(process: subprocess.Popen[str], *, timeout: float = 1.0) -> None:
    try:
        process.wait(timeout=timeout)
        return
    except subprocess.TimeoutExpired:
        _kill_process_group(process)
    except OSError:
        return

    try:
        process.wait(timeout=timeout)
    except (subprocess.TimeoutExpired, OSError):
        pass


def run_sandboxed_python(
    *,
    code: str,
    timeout_seconds: int,
    initial_state: dict[str, Any],
    action_handler: Callable[..., dict[str, Any]],
    animation_handler: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    kept_functions: list[str] | None = None,
    retain_imports: bool = False,
    repair_hints: bool = False,
    sim_guard: bool = False,
    sim_border: int = 4,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="rgb_python_tool_") as sandbox_dir:
        host_action_results: list[dict[str, Any]] = []
        try:
            process = subprocess.Popen(
                [sys.executable, "-I", "-S", "-c", _SANDBOX_BOOTSTRAP],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                cwd=sandbox_dir,
                env=_sandbox_env(),
                start_new_session=True,
            )
        except OSError:
            return {
                "error": "Sandbox process could not start.",
                "stdout": "",
                "action_results": [],
            }
        assert process.stdin is not None
        assert process.stdout is not None
        assert process.stderr is not None

        stdout_queue: queue.Queue[str | None] = queue.Queue()

        def _stdout_reader() -> None:
            for raw_line in process.stdout:
                stdout_queue.put(raw_line)
            stdout_queue.put(None)

        threading.Thread(target=_stdout_reader, daemon=True).start()

        _send_json_line(
            process.stdin,
            {
                "code": code,
                "action_execution_error_fix": os.environ.get("ARC3_ACTION_EXECUTION_ERROR_FIX", "1").strip().lower() in {"1", "true", "yes", "on"},
                "kept_functions": list(kept_functions or []),
                "persistence_enabled": kept_functions is not None,
                "retain_imports": retain_imports,
                "repair_hints": repair_hints,
                "animation_enabled": animation_handler is not None,
                "timeout_seconds": timeout_seconds,
                "sandbox_cwd": sandbox_dir,
                "state": initial_state,
                "color_chars": ARC_COLOR_CHARS,
                "sim_guard": bool(sim_guard),
                "sim_border": int(sim_border),
            },
        )

        deadline = time.monotonic() + max(1, int(timeout_seconds))

        def _sim_query(index: int, before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any] | None:
            # r15 batch guard: ask the child whether the model's step(observe(before), action) predicted
            # what happened. Called by the solver between the actions of one batch, while the child
            # waits for the batch result; None when no answer arrives.
            _send_json_line(process.stdin, {"type": "sim_query", "index": int(index), "before": before,
                                            "after": after})
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            try:
                answer_line = stdout_queue.get(timeout=remaining)
            except queue.Empty:
                return None
            if answer_line is None:
                stdout_queue.put(None)  # the main loop must still see the child exit
                return None
            try:
                answer = json.loads(answer_line)
            except json.JSONDecodeError:
                return None
            return answer if answer.get("type") == "sim_answer" else None

        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                _kill_process_group(process)
                _wait_for_process_exit(process)
                return {
                    "error": f"Tool timed out after {timeout_seconds}s",
                    "stdout": "",
                    "action_results": list(host_action_results),
                }

            try:
                line = stdout_queue.get(timeout=remaining)
            except queue.Empty:
                continue
            if line is None:
                stderr = process.stderr.read()
                _wait_for_process_exit(process)
                return {
                    "error": _sanitize_host_error_text(stderr),
                    "stdout": "",
                    "action_results": list(host_action_results),
                }

            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                stderr = process.stderr.read()
                _kill_process_group(process)
                _wait_for_process_exit(process)
                return {
                    "error": "Sandbox process returned an invalid response.",
                    "stdout": "",
                    "action_results": list(host_action_results),
                }

            msg_type = str(message.get("type", "")).strip()
            if msg_type == "animation":
                if animation_handler is None:
                    _send_json_line(
                        process.stdin,
                        {"type": "animation_error",
                         "error": "Animation frames are not available in this session."},
                    )
                    continue
                try:
                    animation_payload = animation_handler(dict(message.get("request") or {}))
                except Exception:  # noqa: BLE001
                    _send_json_line(
                        process.stdin,
                        {"type": "animation_error",
                         "error": "animation failed in sandbox host."},
                    )
                    continue
                _send_json_line(
                    process.stdin,
                    {"type": "animation_result", "animation": animation_payload},
                )
                continue
            if msg_type == "action":
                try:
                    # the message carries more than the action list: stale_after
                    # tells the solver a previous action in this snippet changed
                    # nothing, which it weighs after its own guards
                    action_result_payload = action_handler(
                        list(message.get("actions") or []),
                        stale_after=message.get("stale_after"),
                        **({"sim_query": _sim_query} if message.get("sim_guard") else {}),
                    )
                except Exception:  # noqa: BLE001
                    _send_json_line(
                        process.stdin,
                        {
                            "type": "action_error",
                            "error": "action failed in sandbox host.",
                        },
                    )
                    continue
                raw_action_result = action_result_payload.get("action_result") or {}
                if isinstance(raw_action_result, dict):
                    host_action_results.append(dict(raw_action_result))
                _send_json_line(
                    process.stdin,
                    {
                        "type": "action_result",
                        "action_result": raw_action_result,
                        "state": action_result_payload.get("state") or {},
                    },
                )
                continue

            if msg_type in {"final", "error"}:
                _wait_for_process_exit(process)
                return {
                    "stdout": str(message.get("stdout", "") or ""),
                    "result": message.get("result"),
                    "error": str(message.get("error", "") or ""),
                    "action_results": list(message.get("action_results") or host_action_results),
                    **({"action_execution_errors": message["action_execution_errors"]}
                       if "action_execution_errors" in message else {}),
                    "keepable_functions": list(message.get("keepable_functions") or []),
                    **({"retention_rejected": message.get("retention_rejected", [])}
                       if kept_functions is not None else {}),
                }

            _wait_for_process_exit(process)
            return {
                "error": "Sandbox process returned an unknown message type.",
                "stdout": "",
                "action_results": list(host_action_results),
            }
