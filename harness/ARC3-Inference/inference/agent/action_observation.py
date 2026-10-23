"""Frame observations only: no game rules, memory tools or model calls."""
import os


def enabled():
    return os.environ.get("ARC3_ACTION_EVIDENCE", "1") == "1"


def classify(before, after, border=4, animation=None):
    def shape(grid):
        if not isinstance(grid, (list, tuple)) or not grid or not isinstance(grid[0], (list, tuple)) or not grid[0]:
            return None
        cols = len(grid[0])
        return (len(grid), cols) if all(isinstance(row, (list, tuple)) and len(row) == cols for row in grid) else None
    a, b = shape(before), shape(after)
    transient = int((animation or {}).get("transient_pixels") or 0)
    result = {"change_scope": "unavailable", "interior_changed": None,
              "border_changed": None, "changed_cells": None,
              "interior_changed_cells": None, "border_changed_cells": None,
              "transient_pixels": transient, "no_op": None}
    if a is None or b is None:
        return result
    if a != b:
        return dict(result, change_scope="shape_changed", interior_changed=True, no_op=False)
    rows, cols = a
    margin = max(0, int(border))
    if rows <= 2 * margin or cols <= 2 * margin:
        margin = 0
    inner = edge = 0
    for r in range(rows):
        for c in range(cols):
            if before[r][c] != after[r][c]:
                if margin <= r < rows - margin and margin <= c < cols - margin:
                    inner += 1
                else:
                    edge += 1
    scope = "interior_changed" if inner else "border_only" if edge else "frame_unchanged"
    # no_op is a visual flag, never proof that hidden game state is unchanged.
    noop = False if inner else None if edge or transient else True
    return dict(result, change_scope=scope, interior_changed=bool(inner),
                border_changed=bool(edge), changed_cells=inner+edge,
                interior_changed_cells=inner, border_changed_cells=edge, no_op=noop)


def verdict(item):
    if item.get("game_over"):
        return "GAME OVER (executed)"
    if item.get("run_complete") or item.get("done"):
        return "RUN COMPLETE"
    if item.get("level_completed"):
        return "LEVEL COMPLETED"
    scope = item.get("change_scope")
    if scope == "border_only":
        return "BORDER CHANGE (executed; interior unchanged; meaning unknown)"
    if scope == "frame_unchanged":
        if int(item.get("transient_pixels") or 0):
            return "ANIMATION ONLY (executed; final frame unchanged)"
        return "NO VISIBLE CHANGE (executed; hidden effects unknown)"
    if scope in ("interior_changed", "shape_changed"):
        shift = item.get("view_shift")
        if shift and len(shift) == 2:
            from inference.utils.view_shift import describe
            return f"VISIBLE CHANGE (executed; view scrolled {describe(shift[0], shift[1], short=True)})"
        return "VISIBLE CHANGE (executed; progress not inferred)"
    return "executed (observation unavailable)"


def stop_reason(item):
    if item.get("change_scope") == "border_only":
        return "border_only_change"
    if item.get("change_scope") == "frame_unchanged":
        return "animation_only" if int(item.get("transient_pixels") or 0) else "no_visible_change"
    return None


def clock_status(agent, fallback=900):
    """Honor game/global deadlines again on each HTTP retry."""
    import time
    session = getattr(getattr(agent, "_step_env_callback", None), "__self__", None)
    limits = [float(fallback or getattr(agent, "_timeout", 900))]
    stopped = False
    if session is not None and hasattr(session, "should_stop"):
        stopped = bool(session.should_stop())
        lifetime = getattr(session.solver, "max_runtime_s_per_game", None)
        if lifetime is not None:
            limits.append(float(lifetime) - (time.monotonic() - session.started_at))
        soft = session.solver.soft_time_remaining_seconds()
        if soft is not None:
            limits.append(float(soft))
    return stopped, max(0.0, min(limits))
