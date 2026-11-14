"""State-change events for dassie-50m (JSON lines).

Emits on transitions only (entered, recognised, lost_recognition_but_tracked,
recognition_regained, released, approaching, receding, left_frame), so a small
language model's context isn't flooded with per-frame data.
"""
from __future__ import annotations

import json
import sys
import time
from typing import TextIO

from .tracker import Track


class EventBus:
    def __init__(self, identity: str, sink: str | None = "-"):
        self.identity = identity.lower()
        self.out: TextIO | None
        if sink is None:
            self.out = None
        elif sink == "-":
            self.out = sys.stdout
        else:
            self.out = open(sink, "a", buffering=1)
        self._prev: dict[int, dict] = {}

    def _emit(self, t: Track, event: str, conf: float) -> None:
        if self.out is None:
            return
        rec = {
            "t": round(time.time(), 2), "track_id": t.id,
            "identity": self.identity if t.locked else "unknown", "state": t.state,
            "similarity": None if t.sim is None else round(t.sim, 3),
            "confidence": round(conf, 3),
            "distance_m": None if t.distance_m is None else round(t.distance_m, 2),
            "face_px": int(t.face_px), "liveness": t.live_state, "event": event,
        }
        self.out.write(json.dumps(rec) + "\n")

    def update(self, tracks: list[Track], removed: list[Track], conf_fn) -> None:
        for t in tracks:
            if t.hits < 3:  # wait for a stable track before announcing it
                continue
            c = conf_fn(t)
            prev = self._prev.get(t.id)
            if prev is None:
                self._emit(t, "entered", c)
                if t.locked:
                    self._emit(t, "recognised", c)
            else:
                if t.locked and not prev["locked"]:
                    self._emit(t, "recognised", c)
                if prev["locked"] and not t.locked:
                    self._emit(t, "released", c)
                if t.locked and t.recognition_lost and not prev["lost"]:
                    self._emit(t, "lost_recognition_but_tracked", c)
                if t.locked and prev["lost"] and not t.recognition_lost:
                    self._emit(t, "recognition_regained", c)
                if t.live_state == "spoof" and prev["live"] != "spoof":
                    self._emit(t, "spoof_detected", c)
                if t.motion != prev["motion"] and t.motion in ("approaching", "receding"):
                    self._emit(t, t.motion, c)
            self._prev[t.id] = {"locked": t.locked, "lost": t.recognition_lost, "motion": t.motion,
                                "live": t.live_state}
        for t in removed:
            if self._prev.pop(t.id, None) is not None:
                self._emit(t, "left_frame", conf_fn(t))

    def close(self) -> None:
        if self.out not in (None, sys.stdout):
            self.out.close()
