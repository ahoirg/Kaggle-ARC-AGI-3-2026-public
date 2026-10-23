"""Death context for the game-over opener (r14): what the fatal action's animation shows.

bp35 (r12-r13 pilots): 13 deaths by floating into a spike, none attributed by the model; the opener's death
block reported "1766 cells changed and changed back" (the camera pan after the death), gave no object tracks
on death turns, and the death-frame image was in post-pan coordinates while the last alive board was not.

Observation only. On a death turn this module adds up to four lines after GAME_OVER_USER_PROMPT:
  1. scroll note: the validated end-to-end view shift between the board before the fatal action and the death
     frame, when the scroll began relative to the victim's disappearance, and the coordinate rule for the
     death-frame image and `last_animation_frames`;
  2. tracks (<= 3): where the moving objects went, in the pre-action board's coordinates, which one disappeared;
  3. contact: what the disappeared object's cells turned into and the nearest objects on each side, with the
     death-frame coordinates of the main contact in brackets when the view scrolled;
  4. the death position in the level's starting view (the current board) when the level's scroll offset is known.
It also supplies the frame in which the victim was last present, for the "moment of death" image that replaces
the fatal-diff image when the view scrolled.

Measurement notes: per-frame un-scrolling is used only when the per-pair shifts reproduce the validated
end-to-end shift (or a net-zero bounce); otherwise anchors from the first frame are interpolated and the note
says so. A frame pair that replaces most of the screen (a cut) ends the record. UI bars are dropped by shape
or by being screen-fixed near an edge; objects over 400 cells are not tracked; parts moving with a body (the
sprite's outline, an eye) are merged into it; a track whose object changes shape is linked; a static object in
the edge band that vanishes is treated as a counter, not a victim. `ARC3_DEATH_CONTEXT=0` disables it.
"""
from __future__ import annotations

import os
from typing import Any

from inference.utils import motion, view_shift
from inference.utils.frame_diff import compute_frame_diff
from inference.utils.grid_utils import ARC_COLOR_CHARS as CC

FILL = 99          # out-of-view cells after un-scrolling
MAX_OBJECT = 400   # larger components are backgrounds, walls, water
MAX_FRAMES = 60


def enabled() -> bool:
    return os.environ.get("ARC3_DEATH_CONTEXT", "1").strip().lower() in {"1", "true", "yes", "on"}


def _letter(color):
    if isinstance(color, str):
        return color
    try:
        v = int(color)
    except (TypeError, ValueError):
        return "?"
    return CC[v] if 0 <= v < len(CC) else "?"


def _clean(chain):
    """Lists of lists of ints, equal shapes, or None."""
    out = []
    shape = None
    for g in chain or []:
        if g is None:
            return None
        try:
            rows = [[int(v) for v in row] for row in g]
        except (TypeError, ValueError):
            return None
        if not rows or any(len(r) != len(rows[0]) for r in rows):
            return None
        if shape is None:
            shape = (len(rows), len(rows[0]))
        elif shape != (len(rows), len(rows[0])):
            return None
        out.append(rows)
    return out if len(out) >= 2 else None


def _nodes(grid):
    return [n for n in motion._nodes(grid) if n["color"] != FILL]


def _shift(before, after, frames):
    """End-to-end view shift between the pre-action board and the death frame, validated."""
    try:
        s = view_shift.detect_action(before, after, frames)
    except Exception:
        s = None
    return (int(s["dy"]), int(s["dx"])) if s else (0, 0)


def _per_pair(chain):
    cum = [(0, 0)]
    dy = dx = 0
    for a, b in zip(chain, chain[1:]):
        try:
            s = view_shift.detect(a, b)
        except Exception:
            s = None
        if s:
            dy += int(s["dy"]); dx += int(s["dx"])
        cum.append((dy, dx))
    return cum


def _unscroll(chain, cum):
    H, W = len(chain[0]), len(chain[0][0])
    out = []
    for g, (dy, dx) in zip(chain, cum):
        ng = [[FILL] * W for _ in range(H)]
        for r in range(H):
            rr = r - dy
            if 0 <= rr < H:
                row = g[r]; nrow = ng[rr]
                for c in range(W):
                    cc = c - dx
                    if 0 <= cc < W:
                        nrow[cc] = row[c]
        out.append(ng)
    return out


def _bar_like(size):
    h, w = size
    return (h <= 2 and w >= 16) or (w <= 2 and h >= 16)


def _sample(chain):
    if len(chain) - 1 <= MAX_FRAMES:
        return list(chain)
    stride = (len(chain) - 1) / MAX_FRAMES
    return [chain[0]] + [chain[int(round(1 + k * stride))] for k in range(MAX_FRAMES - 1)] + [chain[-1]]


def analyse(chain, *, border=4, prefer_colors=()):
    """Structured result: shift, frames used (world coordinates), tracks (merged), vanished objects."""
    chain = _clean(chain)
    if chain is None:
        return None
    frames = _sample(chain)
    shift = _shift(frames[0], frames[-1], frames[1:-1])
    cum = _per_pair(frames)
    note = ""
    interpolated = False
    # a cut: most of the screen replaced between two frames (after removing any detected shift)
    cut = None
    total = len(frames[0]) * len(frames[0][0])
    for i in range(1, len(frames)):
        d = (cum[i][0] - cum[i - 1][0], cum[i][1] - cum[i - 1][1])
        if d == (0, 0):
            changed = sum(1 for ra, rb in zip(frames[i - 1], frames[i]) for a, b in zip(ra, rb) if a != b)
        else:
            try:
                changed = int(view_shift.residual(frames[i - 1], frames[i], d[0], d[1]))
            except Exception:
                changed = total
        if changed > 0.4 * total:
            cut = i
            break
    if cut is not None:
        if cut < 2:
            return {"shift": (0, 0), "note": "the board was replaced in the first frame of the fatal animation; nothing can be tracked",
                    "world": frames[:1], "frames": frames, "tracks": [], "vanished": [], "rank": lambda c: 0, "replaced": True}
        frames = frames[:cut]
        cum = cum[:cut]
        shift = _shift(frames[0], frames[-1], frames[1:-1]) if len(frames) > 1 else (0, 0)
        note = f"the board was replaced at frame {cut} of the animation; the record below ends there"
    if shift != (0, 0) or any(c != (0, 0) for c in cum):
        if cum[-1] == shift:
            world = _unscroll(frames, cum)          # per-frame compensation reproduces the end-to-end shift (or a bounce)
        else:
            # second attempt: offsets measured from the first frame, gaps interpolated between anchors
            anchors = {0: (0, 0), len(frames) - 1: shift}
            for i in range(1, len(frames) - 1):
                try:
                    s_ = view_shift.detect(frames[0], frames[i])
                except Exception:
                    s_ = None
                if s_:
                    anchors[i] = (int(s_["dy"]), int(s_["dx"]))
            keys = sorted(anchors)
            cum2 = []
            for i in range(len(frames)):
                lo = max(k for k in keys if k <= i); hi = min(k for k in keys if k >= i)
                if lo == hi:
                    cum2.append(anchors[lo])
                else:
                    f = (i - lo) / (hi - lo)
                    cum2.append((int(round(anchors[lo][0] + f * (anchors[hi][0] - anchors[lo][0]))),
                                 int(round(anchors[lo][1] + f * (anchors[hi][1] - anchors[lo][1])))))
            cum = cum2
            world = _unscroll(frames, cum)
            interpolated = True
            if len(anchors) <= 2:
                note = (note + "; " if note else "") + "the scroll could not be followed frame by frame, so it was spread evenly over the animation"
    else:
        world = frames
    unscrolled = world is not frames and len(world) == len(frames)
    rows, cols = len(world[0]), len(world[0][0])
    nodes = [_nodes(f) for f in world]
    tracks = {}
    for i in range(1, len(world)):
        try:
            diff = compute_frame_diff(world[i - 1], world[i], nodes[i - 1], nodes[i], max_group_match=motion._MAX_GROUP_MATCH)
        except Exception:
            continue
        for e in diff.get("moved") or []:
            if int(e.get("pixels") or 0) > MAX_OBJECT:
                continue
            if abs(e["to"][0] - e["from"][0]) + abs(e["to"][1] - e["from"][1]) > 8:
                continue  # a jump this large in one frame is a spurious pairing, not motion
            node = next((nd for nd in nodes[i - 1] if nd["color"] == e["color"] and tuple(nd["bbox"][:2]) == tuple(e["from"])), None)
            if node is None:
                continue
            size = motion._size_of(node["bbox"])
            if _bar_like(size) or motion._in_band(node["bbox"], rows, cols, border):
                continue
            if unscrolled and cum[i] != cum[i - 1]:
                # screen-fixed while the view moved, and hugging an edge: a UI element, not a world object
                raw_from = (e["from"][0] + cum[i - 1][0], e["from"][1] + cum[i - 1][1])
                raw_to = (e["to"][0] + cum[i][0], e["to"][1] + cum[i][1])
                near_edge = min(raw_from[0], rows - raw_from[0] - size[0], raw_from[1], cols - raw_from[1] - size[1]) <= 8
                if raw_from == raw_to and near_edge:
                    continue
            key = (e["color"], e["hash"])
            t = tracks.setdefault(key, {"color": e["color"], "size": size, "cells": int(e.get("pixels") or 0),
                                        "path": [tuple(e["from"])], "steps": [], "pos": {}, "parts": [], "band": False})
            if t["path"][-1] != tuple(e["from"]):
                t["path"].append(tuple(e["from"]))
            t["path"].append(tuple(e["to"])); t["steps"].append(i)
            t["pos"][i - 1] = tuple(e["from"]); t["pos"][i] = tuple(e["to"])
    final = {(nd["color"], nd["hash"], tuple(nd["bbox"][:2])) for nd in nodes[-1]}
    out = []
    for key, t in tracks.items():
        t["key"] = key
        if interpolated and abs(t["path"][-1][0] - t["path"][0][0]) + abs(t["path"][-1][1] - t["path"][0][1]) <= 2:
            continue  # jitter from the spread-out scroll, not motion
        end = t["path"][-1]
        left_view = any(world[-1][r][c] == FILL for r in range(end[0], min(rows, end[0] + t["size"][0]))
                        for c in range(end[1], min(cols, end[1] + t["size"][1])))
        t["left_view"] = left_view
        t["disappeared"] = (key[0], key[1], end) not in final and not left_view
        t["last_frame"] = t["steps"][-1]
        out.append(t)
    out.sort(key=lambda t: (-(t["cells"] * len(t["steps"])), -len(t["steps"])))
    # merge parts: a smaller track whose positions stay within 1 cell of the host's bbox on the shared steps
    merged = []
    for t in out:
        host = None
        for m in merged:
            shared = [s for s in t["steps"] if s in m["pos"] and s - 1 in m["pos"]]
            if not shared or len(shared) < max(1, len(t["steps"]) // 2):
                continue
            ok = True
            for s in shared:
                hr, hc = m["pos"][s]; tr, tc = t["pos"][s]
                if not (hr - 1 <= tr <= hr + m["size"][0] and hc - 1 <= tc <= hc + m["size"][1]):
                    ok = False; break
            if ok:
                host = m; break
        if host is not None:
            host["parts"].append(t)
        else:
            merged.append(t)
    # link shape changes: a track that ends where another of the same colour starts one frame later, nearby
    linked = []
    merged.sort(key=lambda t: t["steps"][0])
    for t in merged:
        prev = next((m for m in linked if m["color"] == t["color"] and m["disappeared"] and t["steps"]
                     and m["last_frame"] <= t["steps"][0] - 1 <= m["last_frame"] + 1
                     and abs(m["path"][-1][0] - t["path"][0][0]) <= 2 and abs(m["path"][-1][1] - t["path"][0][1]) <= 2), None)
        if prev is not None:
            prev["path"].extend(t["path"]); prev["steps"].extend(t["steps"]); prev["pos"].update(t["pos"])
            prev["disappeared"] = t["disappeared"]; prev["last_frame"] = t["last_frame"]; prev["size"] = t["size"]
            prev["parts"].extend(t["parts"]); prev["reshaped"] = True
        else:
            linked.append(t)
    # objects gone in the final frame without a track: follow them to their last seen position
    vanished = []
    try:
        d0 = compute_frame_diff(world[0], world[-1], nodes[0], nodes[-1], max_group_match=motion._MAX_GROUP_MATCH)
        for e in d0.get("disappeared") or []:
            px = int(e.get("pixels") or 0)
            if not (2 <= px <= MAX_OBJECT) or _bar_like(motion._size_of(e["bbox"])):
                continue
            key = (e["color"], e["hash"]); at = tuple(e["at"])
            if any(t["key"] == key for t in linked):
                continue
            last, pos = 0, at
            for i in range(1, len(world)):
                # the same object, or what is left of it while it is being covered: same colour, not larger,
                # within 3 cells; never the final frame, where it is gone by definition
                near = [nd for nd in nodes[i] if nd["color"] == e["color"] and nd["pixels"] <= px
                        and abs(nd["bbox"][0] - pos[0]) <= 3 and abs(nd["bbox"][1] - pos[1]) <= 3
                        and not (i == len(world) - 1)]
                if not near:
                    break
                nd = min(near, key=lambda nd: abs(nd["bbox"][0] - pos[0]) + abs(nd["bbox"][1] - pos[1]))
                last, pos = i, tuple(nd["bbox"][:2])
            # scrolled out of view rather than gone?
            if world[-1] is not frames[-1] or any(world[-1][r][c] == FILL for r in range(pos[0], min(rows, pos[0] + e["bbox"][2] - e["bbox"][0] + 1))
                                                  for c in range(pos[1], min(cols, pos[1] + e["bbox"][3] - e["bbox"][1] + 1))):
                if any(world[-1][r][c] == FILL for r in range(pos[0], min(rows, pos[0] + e["bbox"][2] - e["bbox"][0] + 1))
                       for c in range(pos[1], min(cols, pos[1] + e["bbox"][3] - e["bbox"][1] + 1))):
                    continue
            in_band = motion._in_band(e["bbox"], rows, cols, border)
            if in_band and pos == at:
                continue  # a static object in the edge band that vanished is a counter or an indicator, not a victim
            bb = e["bbox"]
            part_cells = set()
            for o in d0.get("disappeared") or []:
                ob = o.get("bbox") or []
                if o is e or len(ob) != 4 or int(o.get("pixels") or 0) > 40:
                    continue
                if ob[2] >= bb[0] - 1 and ob[0] <= bb[2] + 1 and ob[3] >= bb[1] - 1 and ob[1] <= bb[3] + 1:
                    part_cells.update((r, c) for r in range(ob[0], ob[2] + 1) for c in range(ob[1], ob[3] + 1))
            vanished.append({"color": e["color"], "size": motion._size_of(e["bbox"]), "cells": px, "at": at, "pos": pos,
                             "last_frame": last, "moved": pos != at, "band": in_band, "part_cells": part_cells})
    except Exception:
        pass
    pref = [_letter(c) for c in prefer_colors]
    rank = lambda color: pref.index(_letter(color)) if _letter(color) in pref else len(pref)
    vanished.sort(key=lambda v: (rank(v["color"]), -v["cells"]))
    linked.sort(key=lambda t: (-(t["cells"] * len(t["steps"])), -len(t["steps"])))
    return {"shift": shift, "note": note, "world": world, "frames": frames, "tracks": linked, "vanished": vanished, "rank": rank, "cum": cum}


def contact(frame, color, topleft, size, *, exclude_cells=frozenset(), reach=4):
    """Nearest objects on each side of the bbox within `reach` cells, scanning past large regions."""
    H, W = len(frame), len(frame[0])
    r0, c0 = topleft; r1, c1 = r0 + size[0] - 1, c0 + size[1] - 1
    own = _letter(color)
    nodes = _nodes(frame)

    def comp_at(r, c):
        v = frame[r][c]
        return next((nd for nd in nodes if _letter(nd["color"]) == _letter(v) and nd["bbox"][0] <= r <= nd["bbox"][2]
                     and nd["bbox"][1] <= c <= nd["bbox"][3]), None)

    out = []
    for side, cells_fn in (
        ("above", lambda g: [(r0 - 1 - g, c) for c in range(c0, c1 + 1)]),
        ("below", lambda g: [(r1 + 1 + g, c) for c in range(c0, c1 + 1)]),
        ("left", lambda g: [(r, c0 - 1 - g) for r in range(r0, r1 + 1)]),
        ("right", lambda g: [(r, c1 + 1 + g) for r in range(r0, r1 + 1)]),
    ):
        through = []
        found = False
        for gap in range(reach):
            small = {}
            for r, c in cells_fn(gap):
                if not (0 <= r < H and 0 <= c < W) or (r, c) in exclude_cells:
                    continue
                v = frame[r][c]
                if v == FILL or (gap == 0 and _letter(v) == own):
                    continue
                comp = comp_at(r, c)
                if comp is not None and comp["pixels"] > MAX_OBJECT:
                    if _letter(v) not in through:
                        through.append(_letter(v))
                    continue
                small.setdefault(_letter(v), []).append((r, c))
            if small:
                window = set()
                for g2 in range(gap, gap + 5):
                    for r, c in cells_fn(g2):
                        for rr, cc in ((r, c), (r, c - 1), (r, c + 1), (r - 1, c), (r + 1, c)):
                            if 0 <= rr < H and 0 <= cc < W and (rr, cc) not in exclude_cells and frame[rr][cc] != FILL:
                                comp = comp_at(rr, cc)
                                if comp is not None and comp["pixels"] <= MAX_OBJECT and not (
                                        _letter(comp["color"]) == own and abs(comp["bbox"][0] - r0) <= 1 and abs(comp["bbox"][1] - c0) <= 1):
                                    window.add((_letter(comp["color"]), tuple(comp["bbox"]), comp["pixels"]))
                objs = sorted(window, key=lambda w: (-w[2], w[1], w[0]))[:3]
                out.append({"side": side, "gap": gap, "through": list(through), "objects": objs, "large": False})
                found = True
                break
        if not found and through:
            out.append({"side": side, "gap": 0, "through": through, "objects": [], "large": True})
    order = {"above": 0, "below": 1, "left": 2, "right": 3}
    out.sort(key=lambda it: (it["large"], it["gap"], order[it["side"]]))
    return out


def _obj(color, size, cells=None):
    if size:
        return f"{_letter(color)} {size[0]}x{size[1]}"
    return f"{_letter(color)} object"


def _turned(frame, topleft, size):
    if frame is None:
        return None
    H, W = len(frame), len(frame[0])
    counts = {}
    for r in range(topleft[0], topleft[0] + size[0]):
        for c in range(topleft[1], topleft[1] + size[1]):
            if 0 <= r < H and 0 <= c < W and frame[r][c] != FILL:
                counts[_letter(frame[r][c])] = counts.get(_letter(frame[r][c]), 0) + 1
    if not counts:
        return None
    return "/".join(k for k, _ in sorted(counts.items(), key=lambda kv: -kv[1])[:2])


def _contact_text(items):
    texts = []
    for it in items[:4]:
        if it["large"]:
            texts.append(f"{it['side']}: only {'/'.join(it['through'])} (large region{'s' if len(it['through']) > 1 else ''})")
            continue
        what = ", ".join(f"{col} {bb[2] - bb[0] + 1}x{bb[3] - bb[1] + 1} at ({bb[0]},{bb[1]})" for col, bb, _ in it["objects"]) or "small cells"
        gap = "touching" if it["gap"] == 0 else f"{it['gap']} cell{'s' if it['gap'] > 1 else ''} away"
        past = f" past {'/'.join(it['through'])}" if it["through"] else ""
        texts.append(f"{it['side']} ({gap}{past}): {what}")
    return "; ".join(texts) if texts else "nothing but out-of-view cells"


def _in_death_frame(pos, shift):
    """Pre-action coordinates -> death-frame screen coordinates (content moved by `shift`)."""
    return (pos[0] + shift[0], pos[1] + shift[1])


def _level_start_text(pos, offset, rows, cols):
    """Where a pre-action position sits in the level's starting view: content moved `offset` since the start."""
    r, c = pos[0] - offset[0], pos[1] - offset[1]
    parts = []
    if r < 0:
        parts.append(f"{-r} rows above the top edge")
    elif r >= rows:
        parts.append(f"{r - rows + 1} rows below the bottom edge")
    if c < 0:
        parts.append(f"{-c} columns left of the left edge")
    elif c >= cols:
        parts.append(f"{c - cols + 1} columns right of the right edge")
    where = f"({r},{c})" + (f", i.e. {' and '.join(parts)} of that view" if parts else "")
    return where


def report(chain, *, border=4, max_tracks=3, prefer_colors=(), level_offset=None):
    """The death-turn observation.

    Returns {"lines": [...], "scrolled": bool, "shift": (dy, dx), "moment_frame": grid | None,
    "moment_label": str}. `level_offset` is the level's cumulative view offset (dy, dx) before the fatal
    action when it is known, for the level-start coordinates of the death position.
    """
    empty = {"lines": [], "scrolled": False, "shift": (0, 0), "moment_frame": None, "moment_label": ""}
    info = analyse(chain, border=border, prefer_colors=prefer_colors)
    if info is None:
        return empty
    out = []
    if info.get("replaced"):
        return dict(empty, lines=[f"The fatal animation: {info['note']}."])
    shift = tuple(info["shift"])
    scrolled = shift != (0, 0)
    world = info["world"]
    rows, cols = len(world[0]), len(world[0][0])
    tracks = info["tracks"][:max_tracks]
    gone = sorted([t for t in info["tracks"] if t["disappeared"]], key=lambda t: info["rank"](t["color"]))
    victim = None       # (name, last position, size, last frame index, parts)
    if gone:
        t = gone[0]
        victim = (_obj(t["color"], t["size"]), t["path"][-1], t["size"], t["last_frame"], t["parts"], t["color"], set())
    elif info["vanished"]:
        v = info["vanished"][0]
        victim = (_obj(v["color"], v["size"]), v["pos"], v["size"], v["last_frame"], [], v["color"], v.get("part_cells") or set())
    # 1. scroll note, with its timing relative to the disappearance
    if scrolled:
        cum = info.get("cum") or []
        first = next((i for i, c in enumerate(cum) if c != (0, 0)), None)
        timing = ""
        if victim is not None and first is not None:
            k = victim[3]
            timing = (f", beginning at frame {first}, after the {victim[0]} had disappeared (frame {k})" if first > k
                      else f", beginning at frame {first}, while the {victim[0]} was still moving")
        extra = f"; {info['note']}" if info["note"] else ""
        dy, dx = shift
        out.append(f"The view scrolled {view_shift.describe(dy, dx)} during the fatal animation{timing}. "
                   f"The positions below are in the coordinates of the board before the fatal action (`history[-3].frame`); "
                   f"the death frame image, `history[-2].frame` and `last_animation_frames` are shifted by {dy} rows, {dx} columns "
                   f"(death-frame coordinates in brackets; frame k here is `last_animation_timeline` step k, i.e. `last_animation_frames[k-1]`){extra}.")
    elif info["note"]:
        out.append(f"The fatal animation: {info['note']}.")
    # 2. tracks
    for t in tracks:
        p = t["path"]
        shown = p if len(p) <= 6 else p[:3] + [None] + p[-2:]
        text = "->".join("..." if q is None else f"({q[0]},{q[1]})" for q in shown)
        parts = f" (with {', '.join(_obj(q['color'], q['size']) for q in t['parts'][:2])})" if t["parts"] else ""
        shape = " (changing shape)" if t.get("reshaped") else ""
        end = ", then disappeared" if t["disappeared"] else (", then left the view" if t.get("left_view") else "")
        out.append(f"{_obj(t['color'], t['size'])}{parts}{shape} moved {text} over {len(t['steps'])} step(s); ends "
                   f"{motion._delta(p[0], p[-1])} of its start{end}.")
    # 3. contact
    moment_frame = None
    moment_label = ""
    if victim is not None:
        name, pos, size, k, parts, color, excl = victim
        frame = world[k]
        excl = set(excl)
        for q in parts:
            qp = q["pos"].get(k) or q["path"][-1]
            excl.update((r, c) for r in range(qp[0], qp[0] + q["size"][0]) for c in range(qp[1], qp[1] + q["size"][1]))
        items = contact(frame, color, pos, size, exclude_cells=excl)
        nxt = _turned(world[k + 1], pos, size) if k + 1 < len(world) else None
        fin = _turned(world[-1], pos, size)
        turned = (f"its cells turned {nxt}" + (f", and in the final frame {fin}" if fin and fin != nxt else "")) if nxt else ""
        parts_text = f" (with {', '.join(_obj(q['color'], q['size']) for q in parts[:2])})" if parts else ""
        bracket = f" [death frame: ({_in_death_frame(pos, shift)[0]},{_in_death_frame(pos, shift)[1]})]" if scrolled else ""
        if gone:
            head = f"Where the {name}{parts_text} disappeared ({pos[0]},{pos[1]}){bracket}: "
        else:
            v = info["vanished"][0]
            where = (f"after sliding from ({v['at'][0]},{v['at'][1]})" if v["moved"] else "without having moved")
            band = " (in the outer edge band)" if v["band"] else ""
            head = (f"{name}{band} is gone in the final frame, last seen at ({pos[0]},{pos[1]}){bracket} in frame {k} "
                    f"of {len(world) - 1} {where}: ")
        text = _contact_text(items)
        if scrolled and items and not items[0]["large"] and items[0]["objects"]:
            col, bb, _ = items[0]["objects"][0]
            dr, dc = _in_death_frame((bb[0], bb[1]), shift)
            text = text.replace(f"{col} {bb[2] - bb[0] + 1}x{bb[3] - bb[1] + 1} at ({bb[0]},{bb[1]})",
                                f"{col} {bb[2] - bb[0] + 1}x{bb[3] - bb[1] + 1} at ({bb[0]},{bb[1]}) [death frame: ({dr},{dc})]", 1)
        out.append(head + (turned + "; " if turned else "") + "nearest on each side: " + text + ".")
        # 4. the death position in the level's starting view
        if level_offset is not None and tuple(level_offset) != (0, 0):
            out.append(f"In the level's starting view (the board after the reset) that position is "
                       f"{_level_start_text(pos, level_offset, rows, cols)}.")
        moment_frame = [list(r) for r in frame]
        moment_label = (f"Moment of death image: frame {k} of the fatal animation, the last frame in which the {name} is "
                        f"present; same coordinates as the lines above" + (
                            " (cells that had scrolled out of view are white)" if any(FILL in r for r in frame) else "") + ":")
        for r in moment_frame:
            for i, v in enumerate(r):
                if v == FILL:
                    r[i] = 0
    elif not tracks:
        out.append("No object moved or disappeared during the fatal animation.")
    return {"lines": out, "scrolled": scrolled, "shift": shift, "moment_frame": moment_frame, "moment_label": moment_label}


def lines(chain, *, border=4, max_tracks=3, prefer_colors=()):
    return report(chain, border=border, max_tracks=max_tracks, prefer_colors=prefer_colors)["lines"]


def history_pairs(entries, level, *, limit=12):
    """(before, after) boards of the model's own actions on this level, excluding automatic resets,
    RESET, not-executed and scrolled steps and the fatal step itself."""
    from inference.agent.exploration import level_entries, _name

    lvl = level_entries(list(entries or []), level)
    pairs = []
    for i in range(1, len(lvl)):
        r = getattr(lvl[i], "result", None) or {}
        display = str(getattr(lvl[i], "action", "") or "")
        if (not display or r.get("automatic") or _name(display) == "RESET"
                or not r.get("executed", True) or r.get("view_shift") or r.get("game_over")):
            continue
        before = getattr(getattr(lvl[i - 1], "frame", None), "grid", None)
        after = getattr(getattr(lvl[i], "frame", None), "grid", None)
        if before is not None and after is not None:
            pairs.append((before, after))
    return pairs[-limit:]


def recent_movers(pairs, *, limit=12):
    """Colours of objects that moved in the last (before, after) boards, most frequent first: avatar candidates."""
    colors = {}
    for before, after in list(pairs)[-limit:]:
        try:
            na, nb = motion._nodes(before), motion._nodes(after)
            diff = compute_frame_diff(before, after, na, nb, max_group_match=motion._MAX_GROUP_MATCH)
        except Exception:
            continue
        for e in diff.get("moved") or []:
            if int(e.get("pixels") or 0) <= MAX_OBJECT:
                colors[_letter(e["color"])] = colors.get(_letter(e["color"]), 0) + 1
    return [c for c, _ in sorted(colors.items(), key=lambda kv: -kv[1])]

