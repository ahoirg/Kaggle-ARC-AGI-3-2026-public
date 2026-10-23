"""Object-level motion summaries for the turn opener and the sandbox (r10, lever 1).

Why: the model estimated motion by eye or by printing frames. sb26 spent about 230k reasoning
characters in one level reading 51-frame playbacks cell by cell; ar25 read a +4 column move as +3
and lost 20 minutes to the wrong residue class; vc33 swaps looked like recolours; a sk48 indicator
lit and went out inside one batch and never reached a turn message. `frame_diff` already matches
objects between two boards; this module turns its output into short phrases, applies it to every
action of a batch and to every frame of an animation (tracks: where an object went, what flashed).

The same source runs in the sandbox child, where `compute_frame_diff` and `segment_layer` are
module globals; the import below is for the host.
"""
try:  # host
    from inference.utils.frame_diff import compute_frame_diff  # noqa: F401
except Exception:  # pragma: no cover - sandbox child: the name is already in the namespace
    pass

MOTION_COLOR_CHARS = "WwgGcBMPRbSYOrNp"
_MAX_GROUP_MATCH = 12          # identical-shape groups larger than this are not paired (cost)
_MAX_ANIMATION_DIFFS = 60      # longer chains are sampled evenly
_MAX_OBJECT = 400              # larger components are backgrounds or frames (same bound as exploration.py)


def _nodes(grid, color_chars=MOTION_COLOR_CHARS):
    """Connected components with the fields `compute_frame_diff` reads (color letter, pixels, hash,
    shape_hash, boundary = bbox corners). About 8x faster than `segment_layer`, which also traces
    contours and rotations; an animation of 50 frames must be summarized in about a second."""
    try:
        rows = len(grid)
        cols = len(grid[0]) if rows else 0
        seen = bytearray(rows * cols)
        nodes = []
        for r0 in range(rows):
            row0 = grid[r0]
            for c0 in range(cols):
                if seen[r0 * cols + c0]:
                    continue
                colour = row0[c0]
                stack = [(r0, c0)]
                seen[r0 * cols + c0] = 1
                cells = []
                while stack:
                    r, c = stack.pop()
                    cells.append((r, c))
                    if r > 0 and not seen[(r - 1) * cols + c] and grid[r - 1][c] == colour:
                        seen[(r - 1) * cols + c] = 1; stack.append((r - 1, c))
                    if r + 1 < rows and not seen[(r + 1) * cols + c] and grid[r + 1][c] == colour:
                        seen[(r + 1) * cols + c] = 1; stack.append((r + 1, c))
                    if c > 0 and not seen[r * cols + c - 1] and grid[r][c - 1] == colour:
                        seen[r * cols + c - 1] = 1; stack.append((r, c - 1))
                    if c + 1 < cols and not seen[r * cols + c + 1] and grid[r][c + 1] == colour:
                        seen[r * cols + c + 1] = 1; stack.append((r, c + 1))
                rr = [cell[0] for cell in cells]; cc = [cell[1] for cell in cells]
                top, left, bottom, right = min(rr), min(cc), max(rr), max(cc)
                if len(cells) <= _MAX_OBJECT:
                    shape = tuple(sorted((r - top, c - left) for r, c in cells))
                    shape_hash = str(hash(shape))
                else:
                    shape_hash = f"big{len(cells)}"
                letter = color_chars[colour] if isinstance(colour, int) and 0 <= colour < len(color_chars) else str(colour)
                nodes.append({"id": len(nodes), "color": letter, "pixels": len(cells),
                              "hash": f"{letter}:{shape_hash}", "shape_hash": shape_hash,
                              "bbox": [top, left, bottom, right],
                              "boundary": [[top, left], [top, right], [bottom, left], [bottom, right]],
                              "rotation": None, "rotational_symmetry": 1})
        return nodes
    except Exception:
        return []


def _size_of(bbox):
    if not bbox or len(bbox) < 4:
        return None
    return (int(bbox[2]) - int(bbox[0]) + 1, int(bbox[3]) - int(bbox[1]) + 1)


def _in_band(bbox, rows, cols, border):
    """True when the whole object lies in the outer band (counters, step bars)."""
    if not bbox or border <= 0:
        return False
    r0, c0, r1, c1 = bbox
    return r1 < border or c1 < border or r0 >= rows - border or c0 >= cols - border


def _delta(a, b):
    dr, dc = int(b[0]) - int(a[0]), int(b[1]) - int(a[1])
    parts = []
    if dr:
        parts.append(f"{abs(dr)} row{'s' if abs(dr) != 1 else ''} {'down' if dr > 0 else 'up'}")
    if dc:
        parts.append(f"{abs(dc)} column{'s' if abs(dc) != 1 else ''} {'right' if dc > 0 else 'left'}")
    return ", ".join(parts) or "in place"


def _obj(color, size, cells=None):
    if size:
        return f"{color} {size[0]}x{size[1]}"
    return f"{color} {cells or '?'}-cell object"


def motion_items(grid_a, grid_b, *, border=4, color_chars=MOTION_COLOR_CHARS):
    """Phrases describing what changed between two boards, largest objects first.

    Returns a dict: `items` (list of (in_band, cells, text)), `changed_cells`.
    """
    if grid_a is None or grid_b is None or len(grid_a) != len(grid_b):
        return {"items": [], "changed_cells": -1}
    rows, cols = len(grid_a), len(grid_a[0]) if grid_a else 0
    nodes_a, nodes_b = _nodes(grid_a, color_chars), _nodes(grid_b, color_chars)
    try:
        diff = compute_frame_diff(grid_a, grid_b, nodes_a, nodes_b, max_group_match=_MAX_GROUP_MATCH)
    except Exception:
        return {"items": [], "changed_cells": -1}
    by_pos_a = {(n.get("color"), tuple(n.get("bbox", [])[:2])): n for n in nodes_a}
    by_pos_b = {(n.get("color"), tuple(n.get("bbox", [])[:2])): n for n in nodes_b}
    items = []

    def add(bbox, cells, text, rank):
        if int(cells or 0) > _MAX_OBJECT:
            return  # backgrounds and frames, not objects
        items.append((_in_band(bbox, rows, cols, border), rank, int(cells or 0), text))

    for e in diff.get("moved") or []:
        node = by_pos_a.get((e.get("color"), tuple(e.get("from") or [])))
        bbox = node.get("bbox") if node else None
        text = f"{_obj(e.get('color'), _size_of(bbox), e.get('pixels'))} moved ({e['from'][0]},{e['from'][1]})->({e['to'][0]},{e['to'][1]}): {_delta(e['from'], e['to'])}"
        if e.get("rotated_by"):
            text += f", rotated {e['rotated_by']} deg"
        add(bbox, e.get("pixels"), text, 0)
    for e in diff.get("rotated") or []:
        node = by_pos_b.get((e.get("color"), tuple(e.get("at") or [])))
        bbox = node.get("bbox") if node else None
        add(bbox, e.get("pixels"), f"{_obj(e.get('color'), _size_of(bbox), e.get('pixels'))} at ({e['at'][0]},{e['at'][1]}) rotated {e.get('rotated_by')} deg", 0)
    for e in diff.get("changed_color") or []:
        at = e.get("at") or [0, 0]
        before, after = e.get("before_color"), e.get("after_color")
        node = by_pos_b.get((after, tuple(at)))
        bbox = node.get("bbox") if node else None
        add(bbox, e.get("pixels"), f"{_obj(before, _size_of(bbox), e.get('pixels'))} at ({at[0]},{at[1]}) changed colour {before}>{after}", 0)
    for e in diff.get("resized") or []:
        at = e.get("at_after") or e.get("at_before") or [0, 0]
        node = by_pos_b.get((e.get("color"), tuple(at)))
        bbox = node.get("bbox") if node else None
        add(bbox, max(int(e.get("before_pixels") or 0), int(e.get("after_pixels") or 0)),
            f"{e.get('color')} object at ({at[0]},{at[1]}) resized {e.get('before_pixels')}->{e.get('after_pixels')} cells", 2)
    for e in diff.get("appeared") or []:
        bbox = e.get("bbox")
        add(bbox, e.get("pixels"), f"{_obj(e.get('color'), _size_of(bbox), e.get('pixels'))} appeared at ({e['at'][0]},{e['at'][1]})", 1)
    for e in diff.get("disappeared") or []:
        bbox = e.get("bbox")
        add(bbox, e.get("pixels"), f"{_obj(e.get('color'), _size_of(bbox), e.get('pixels'))} disappeared from ({e['at'][0]},{e['at'][1]})", 1)
    if sum(1 for item in items if item[2] > 1) >= 2:
        items = [item for item in items if item[2] > 1]  # single cells are fragments of larger changes
    items.sort(key=lambda item: (item[0], item[1], -item[2]))
    return {"items": items, "changed_cells": int(diff.get("changed_cell_count") or 0)}


def motion_lines(grid_a, grid_b, *, border=4, max_items=6, color_chars=MOTION_COLOR_CHARS):
    """One phrase per changed object, board objects first, edge-band objects tagged and last."""
    result = motion_items(grid_a, grid_b, border=border, color_chars=color_chars)
    items = result["items"]
    lines = [text + (" (edge band)" if in_band else "") for in_band, _, _, text in items[:max_items]]
    if len(items) > max_items:
        lines.append(f"... and {len(items) - max_items} more changed objects")
    return lines


def _signature(node):
    return (node.get("color"), node.get("hash"), tuple(node.get("bbox", [])[:2]))


def batch_motion(grids, actions, *, border=4, max_actions=12, color_chars=MOTION_COLOR_CHARS):
    """Per-action phrases for a batch. `grids[i]` is the board before `actions[i]`; `grids[-1]` the
    final board (len(grids) == len(actions) + 1). Returns {"per_action": [(index, action, lines)]}."""
    per_action = []
    if not grids or len(grids) != len(actions) + 1:
        return {"per_action": per_action}
    for index in range(min(len(actions), max_actions)):
        lines = motion_lines(grids[index], grids[index + 1], border=border, max_items=3, color_chars=color_chars)
        per_action.append((index + 1, str(actions[index]), lines))
    return {"per_action": per_action}


def animation_tracks(chain, *, border=4, max_tracks=4, color_chars=MOTION_COLOR_CHARS):
    """Object tracks through an animation: where each moving object went, what flashed.

    `chain[0]` is the pre-action board. Returns a list of phrases (≤ max_tracks + 1).
    """
    if not chain or len(chain) < 3:
        return []
    frames = list(chain)
    if len(frames) - 1 > _MAX_ANIMATION_DIFFS:
        stride = (len(frames) - 1) / _MAX_ANIMATION_DIFFS
        frames = [chain[0]] + [chain[int(round(1 + k * stride))] for k in range(_MAX_ANIMATION_DIFFS - 1)] + [chain[-1]]
    rows, cols = len(frames[0]), len(frames[0][0]) if frames[0] else 0
    tracks = {}      # key (color, hash) -> {"path": [topleft...], "steps": [i...], "cells": n}
    flashes = {}     # key (color, hash, topleft) -> count of appearances
    for index in range(1, len(frames)):
        a, b = frames[index - 1], frames[index]
        nodes_a, nodes_b = _nodes(a, color_chars), _nodes(b, color_chars)
        try:
            diff = compute_frame_diff(a, b, nodes_a, nodes_b, max_group_match=_MAX_GROUP_MATCH)
        except Exception:
            continue
        for e in diff.get("moved") or []:
            key = (e.get("color"), e.get("hash"))
            node = next((nd for nd in nodes_a if nd.get("color") == e.get("color") and tuple(nd.get("bbox", [])[:2]) == tuple(e["from"])), None)
            if node is not None and _in_band(node.get("bbox"), rows, cols, border):
                continue
            tr = tracks.setdefault(key, {"path": [tuple(e["from"])], "steps": [], "cells": e.get("pixels"), "size": _size_of(node.get("bbox")) if node else None})
            if tr["path"][-1] != tuple(e["from"]):
                tr["path"].append(tuple(e["from"]))
            tr["path"].append(tuple(e["to"])); tr["steps"].append(index)
        for e in diff.get("appeared") or []:
            if _in_band(e.get("bbox"), rows, cols, border) or int(e.get("pixels") or 0) > 400:
                continue
            fk = (e.get("color"), e.get("hash"), tuple(e["at"]))
            fl = flashes.setdefault(fk, {"count": 0, "first": index, "last": index, "size": _size_of(e.get("bbox")), "cells": e.get("pixels"), "final": False})
            fl["count"] += 1; fl["last"] = index
    final_nodes_list = _nodes(frames[-1], color_chars)
    final_nodes = {_signature(nd) for nd in final_nodes_list}
    lines = []
    ranked = sorted(tracks.items(), key=lambda kv: (-(int(kv[1]["cells"] or 0) * len(kv[1]["steps"])), -len(kv[1]["steps"])))
    if sum(1 for _, tr in ranked if int(tr["cells"] or 0) > 1) >= 2:
        ranked = [item for item in ranked if int(item[1]["cells"] or 0) > 1]  # single cells are trails, not objects
    for key, tr in ranked[:max_tracks]:
        path = tr["path"]
        distinct = sorted(set(path))
        if len(distinct) == 2 and len(tr["steps"]) >= 3:
            a_, b_ = distinct
            lines.append(f"{_obj(key[0], tr['size'], tr['cells'])} alternated between ({a_[0]},{a_[1]}) and ({b_[0]},{b_[1]}) "
                         f"{len(tr['steps'])} times (steps {tr['steps'][0]}-{tr['steps'][-1]}): two objects of the same shape flashing in turn, or one object jumping")
            continue
        shown = path if len(path) <= 6 else path[:3] + [None] + path[-2:]
        text_path = "->".join("..." if p is None else f"({p[0]},{p[1]})" for p in shown)
        end = ""
        if (key[0], key[1], path[-1]) not in final_nodes:
            there = [nd for nd in final_nodes_list if tuple(nd.get("bbox", [])[:2]) == path[-1] and _size_of(nd.get("bbox")) == tr["size"]]
            end = f", then turned {there[0].get('color')}" if there else ", then disappeared"
        lines.append(f"{_obj(key[0], tr['size'], tr['cells'])} moved {text_path} over {len(tr['steps'])} step(s); ends {_delta(path[0], path[-1])} of its start{end}")
    for fk, fl in sorted(flashes.items(), key=lambda kv: -kv[1]["count"])[: max(0, max_tracks - len(lines))]:
        color, _, at = fk
        stays = (color, fk[1], at) in final_nodes
        if fl["count"] >= 2:
            lines.append(f"{_obj(color, fl['size'], fl['cells'])} at ({at[0]},{at[1]}) appeared {fl['count']} times (steps {fl['first']}-{fl['last']}){'' if stays else ', gone in the final frame'}")
        elif not stays:
            lines.append(f"{_obj(color, fl['size'], fl['cells'])} at ({at[0]},{at[1]}) appeared at step {fl['first']} and is gone in the final frame")
    return lines
