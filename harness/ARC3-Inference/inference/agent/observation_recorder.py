"""Passive, per-game observation archive. Never injected into model context."""
import gzip
import hashlib
import json
import os
import re
import threading
import warnings
from pathlib import Path

REGISTRY = {}
_LOCK = threading.Lock()


def _digest(grid):
    return hashlib.sha256(json.dumps(grid, separators=(",", ":")).encode()).hexdigest()[:16]


class ObservationRecorder:
    def __init__(self, owner):
        self.owner = owner
        self.root = Path(os.environ["ARC3_OBSERVATION_DIR"]) / re.sub(r"[^a-zA-Z0-9_.-]", "_", owner)
        self.sequence = self.actions = self.disk_errors = 0
        with _LOCK:
            REGISTRY[owner] = self

    def write(self, record):
        self.sequence += 1
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            with gzip.open(self.root / "evidence.jsonl.gz", "at", encoding="utf-8") as handle:
                handle.write(json.dumps({"id": f"e{self.sequence:06d}", **record}, separators=(",", ":"), allow_nan=False) + "\n")
        except (OSError, ValueError, TypeError) as error:
            self.disk_errors += 1
            if self.disk_errors == 1:
                warnings.warn(f"Observation recording failed for {self.owner}: {type(error).__name__}")

    def transition(self, before, after, payload, pre_level):
        if self.sequence == 0:
            self.write({"kind": "initial", "level": pre_level, "step": max(0, int(payload.get("action_num") or 1)-1),
                        "after": before, "hash": _digest(before)})
        automatic = bool(payload.get("automatic"))
        kind = (("automatic_reset" if automatic else "reset") if payload.get("action_name") == "RESET" else
                "fatal" if payload.get("game_over") else
                "level_boundary" if payload.get("level_completed") or payload.get("done") else "transition")
        changes = [(r,c,old,new) for r,(a,b) in enumerate(zip(before or [],after or []))
                   for c,(old,new) in enumerate(zip(a,b)) if old != new]
        self.write({"kind": kind, "level": pre_level, "after_level": payload.get("level"),
                    "step": payload.get("action_num"), "action": payload.get("action_display"),
                    "automatic": automatic, "before": before, "after": after,
                    "before_hash": _digest(before), "after_hash": _digest(after),
                    "changes_count": len(changes), "changes": changes[:24], "changes_truncated": len(changes)>24,
                    "animation": payload.get("animation_chain") or [],
                    "result": {k:v for k,v in payload.items() if k != "animation_chain"}})
        self.actions += int(not automatic)


def record_transition(session, before, after, payload, pre_level):
    if not os.environ.get("ARC3_OBSERVATION_DIR"):
        return
    recorder = getattr(session, "_observation_recorder", None)
    if recorder is None:
        owner = f"g{session.game_index:02d}-p{session.pass_index}-{session.transcript_path.stem}"
        recorder = session._observation_recorder = ObservationRecorder(owner)
    recorder.transition(before, after, payload, pre_level)


def run_summary():
    with _LOCK:
        rows = [{"owner": r.owner, "evidence_records": r.sequence, "model_actions": r.actions,
                 "disk_errors": r.disk_errors} for r in REGISTRY.values()]
    return {"games": rows, "totals": {key: sum(r[key] for r in rows)
            for key in ("evidence_records", "model_actions", "disk_errors")}}
