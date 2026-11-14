"""Detect -> track -> align/embed -> match -> fuse -> distance, shared by run and eval."""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .align_embed import Embedder
from .config import Calibration, Thresholds, TrackerConfig
from .detect import Detector
from .distance import DistanceEstimator
from .gallery import Gallery, nearest_band, pose_label
from .overlay import confidence
from .tracker import Track, Tracker


@dataclass
class SessionStats:
    max_recognised_m: float = 0.0  # furthest per-frame recognition on a locked track
    max_tracked_m: float = 0.0     # furthest distance a locked track was held
    frames: int = 0


class Pipeline:
    def __init__(self, gallery: Gallery | None, embedder: Embedder, calib: Calibration,
                 frame_width: int, tracker_cfg: TrackerConfig | None = None,
                 match_mode: str = "max", adaptive: bool = False, adaptive_cap: int = 40):
        self.detector = Detector()
        self.embedder = embedder
        self.gallery = gallery
        self.th: Thresholds = calib.thresholds[embedder.name]
        self.dist = DistanceEstimator(calib.distance, frame_width)
        self.tracker = Tracker(tracker_cfg)
        self.match_mode = match_mode
        self.adaptive = adaptive
        self.adaptive_cap = adaptive_cap
        self.stats = SessionStats()

    def _match(self, e: np.ndarray) -> float:
        return self.gallery.match(e, self.match_mode) if self.gallery is not None else 0.0

    def process(self, frame: np.ndarray, t: float | None = None) -> list[Track]:
        t = time.time() if t is None else t
        tracks = self.tracker.step(self.detector(frame))
        self.stats.frames += 1
        for tr in tracks:
            if tr.det is None:
                continue
            emb, _crop = self.embedder(frame, tr.det)
            tr.last_emb = emb
            tr.observe(emb, self._match(emb), self._match, self.th)
            tr.update_distance(self.dist.estimate(tr.det.width_px, tr.det.eye_px, tr.det.yaw_hint()), t)

            if tr.locked and tr.distance_m is not None:
                self.stats.max_tracked_m = max(self.stats.max_tracked_m, tr.distance_m)
                if tr.recognised_now:
                    self.stats.max_recognised_m = max(self.stats.max_recognised_m, tr.distance_m)
            if self.adaptive:
                self._maybe_adapt(tr, emb)
        return tracks

    def _maybe_adapt(self, tr: Track, emb: np.ndarray) -> None:
        """Add embeddings from under-represented bands, only under strict conditions."""
        g = self.gallery
        if g is None or not tr.locked or tr.sim is None or tr.det is None:
            return
        if tr.sim < self.th.lock_high + 0.05 or (tr.fused_sim or 0) < self.th.lock_high + 0.05:
            return
        band = nearest_band(tr.distance_m)
        if g.band_counts().get(band, 0) >= self.adaptive_cap // 2:
            return
        if len(g) >= self.adaptive_cap * 6 or float((g.emb @ emb).max()) > 0.92:
            return  # gallery full, or nothing new to learn from this frame
        g.add(emb, band, tr.det.width_px, pose_label(tr.det.yaw_hint()))

    def conf(self, tr: Track) -> float:
        return confidence(tr.sim_ema, self.th)
