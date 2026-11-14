"""Kalman + IoU tracker with a per-track identity lock (hysteresis state machine).

Separates two things the UI shows:
  * per-frame recognition: does this frame's embedding clear the match threshold?
  * track identity: has this track been confirmed as the enrolled person?
A locked track stays locked while the tracker holds the box, even when the face is too
small for per-frame recognition to work.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from itertools import count

import numpy as np

from .config import Thresholds, TrackerConfig
from .detect import Detection


def iou(a: np.ndarray, b: np.ndarray) -> float:
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw = max(0.0, min(ax2, bx2) - max(a[0], b[0]))
    ih = max(0.0, min(ay2, by2) - max(a[1], b[1]))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return float(inter / union) if union > 0 else 0.0


class BoxKalman:
    """Constant-velocity Kalman filter on (cx, cy, w, h)."""

    def __init__(self, box: np.ndarray):
        cx, cy, w, h = box[0] + box[2] / 2, box[1] + box[3] / 2, box[2], box[3]
        self.x = np.array([cx, cy, w, h, 0, 0, 0, 0], np.float64)
        self.P = np.diag([10, 10, 10, 10, 100, 100, 100, 100]).astype(np.float64)
        self.F = np.eye(8)
        self.F[:4, 4:] = np.eye(4)
        self.H = np.eye(4, 8)
        self.Q = np.diag([1, 1, 1, 1, 0.5, 0.5, 0.5, 0.5])
        self.R = np.diag([4, 4, 9, 9]).astype(np.float64)

    def predict(self) -> np.ndarray:
        self.x = self.F @ self.x
        self.x[2:4] = np.maximum(self.x[2:4], 2)
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.box()

    def update(self, box: np.ndarray) -> None:
        z = np.array([box[0] + box[2] / 2, box[1] + box[3] / 2, box[2], box[3]])
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(8) - K @ self.H) @ self.P

    def box(self) -> np.ndarray:
        cx, cy, w, h = self.x[:4]
        return np.array([cx - w / 2, cy - h / 2, w, h], np.float32)


@dataclass
class Track:
    id: int
    kf: BoxKalman
    det: Detection | None
    cfg: TrackerConfig
    hits: int = 1
    missed: int = 0
    age: int = 0
    # recognition
    sim: float | None = None           # this frame's similarity (None if not embedded)
    sim_ema: float | None = None
    fused_sim: float | None = None     # similarity of the averaged recent embeddings
    embeds: deque = field(default_factory=deque)
    last_emb: np.ndarray | None = None
    last_width: float = 0.0
    locked: bool = False
    hi_count: int = 0
    lo_count: int = 0
    below_count: int = 0
    recognised_now: bool = False       # per-frame sim >= match threshold
    recognition_lost: bool = False     # locked but per-frame recognition has failed for a while
    # distance
    distance_m: float | None = None
    dist_hist: deque = field(default_factory=lambda: deque(maxlen=30))
    motion: str = "still"

    @property
    def box(self) -> np.ndarray:
        return self.kf.box()

    @property
    def face_px(self) -> float:
        return self.det.width_px if self.det is not None else self.last_width

    @property
    def state(self) -> str:
        if self.locked:
            return "locked" if not self.recognition_lost else "tracked"
        return "candidate" if self.hi_count > 0 else "unknown"

    def observe(self, emb: np.ndarray, sim: float, fused_fn, th: Thresholds) -> None:
        """Feed this frame's embedding + per-frame similarity into the lock state machine."""
        c = self.cfg
        self.sim = sim
        self.sim_ema = sim if self.sim_ema is None else c.ema_alpha * sim + (1 - c.ema_alpha) * self.sim_ema
        self.recognised_now = sim >= th.match
        self.below_count = 0 if self.recognised_now else self.below_count + 1

        judgeable = self.face_px >= c.min_judge_px
        if judgeable:
            self.embeds.append(emb)
            while len(self.embeds) > c.embed_window:
                self.embeds.popleft()
        if self.embeds:
            mean = np.mean(np.stack(self.embeds), axis=0)
            self.fused_sim = fused_fn(mean / max(np.linalg.norm(mean), 1e-12))

        if not judgeable or self.fused_sim is None:
            # Too small to judge: keep the current identity, don't move the counters.
            self.recognition_lost = self.locked and self.below_count >= c.lost_recog_frames
            return

        if not self.locked:
            self.hi_count = self.hi_count + 1 if self.fused_sim >= th.lock_high else 0
            if self.hi_count >= c.lock_frames:
                self.locked, self.lo_count = True, 0
        else:
            self.lo_count = self.lo_count + 1 if self.fused_sim < th.lock_low else 0
            if self.lo_count >= c.release_frames:
                self.locked, self.hi_count = False, 0
                self.embeds.clear()
        self.recognition_lost = self.locked and self.below_count >= c.lost_recog_frames

    def update_distance(self, d: float, t: float, alpha: float = 0.35) -> None:
        self.distance_m = d if self.distance_m is None else alpha * d + (1 - alpha) * self.distance_m
        self.dist_hist.append((t, self.distance_m))
        if len(self.dist_hist) >= 8:
            (t0, d0), (t1, d1) = self.dist_hist[0], self.dist_hist[-1]
            if t1 - t0 > 0.3:
                v = (d1 - d0) / (t1 - t0)  # m/s, positive = moving away
                if v < -0.25:
                    self.motion = "approaching"
                elif v > 0.25:
                    self.motion = "receding"
                elif abs(v) < 0.1:
                    self.motion = "still"


class Tracker:
    def __init__(self, cfg: TrackerConfig | None = None):
        self.cfg = cfg or TrackerConfig()
        self.tracks: list[Track] = []
        self._ids = count(1)
        self.removed: list[Track] = []  # tracks dropped on the last step (for events)

    def step(self, dets: list[Detection]) -> list[Track]:
        self.removed = []
        for t in self.tracks:
            if t.det is not None:
                t.last_width = t.det.width_px
            if t.missed > 0:  # coasting: damp velocity so a lost box doesn't fly off or collapse
                t.kf.x[4:] *= 0.5
            t.kf.predict()
            t.age += 1
            t.det = None

        # Greedy IoU association (optimal enough for a handful of faces).
        pairs = sorted(((iou(t.box, d.box), ti, di) for ti, t in enumerate(self.tracks)
                        for di, d in enumerate(dets)), reverse=True)
        used_t, used_d = set(), set()
        for score, ti, di in pairs:
            if score < self.cfg.iou_match:
                break
            if ti in used_t or di in used_d:
                continue
            t = self.tracks[ti]
            t.kf.update(dets[di].box)
            t.det, t.hits, t.missed = dets[di], t.hits + 1, 0
            used_t.add(ti)
            used_d.add(di)

        for ti, t in enumerate(self.tracks):
            if ti not in used_t:
                t.missed += 1
        for di, d in enumerate(dets):
            if di not in used_d:
                self.tracks.append(Track(next(self._ids), BoxKalman(d.box), d, self.cfg,
                                         last_width=d.width_px))

        keep = []
        for t in self.tracks:
            (keep if t.missed <= self.cfg.max_missed else self.removed).append(t)
        self.tracks = keep
        return self.tracks
