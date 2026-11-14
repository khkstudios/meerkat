"""Hand tracking (MediaPipe HandLandmarker, Apache 2.0) + per-hand gesture classification.

Landmark indexes (MediaPipe): 0 wrist, 4 thumb tip, 8 index tip, 12 middle tip, 16 ring tip,
20 pinky tip; finger joints MCP/PIP at 5/6, 9/10, 13/14, 17/18.
"""
from __future__ import annotations

import contextlib
import math
import os
import sys
import threading
import time
from dataclasses import dataclass, field

os.environ.setdefault("GLOG_minloglevel", "2")       # quieten MediaPipe / LiteRT logging
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from .config import MODELS_DIR  # noqa: E402

HAND_MODEL = "hand_landmarker.task"
FINGERS = {"index": (5, 6, 8), "middle": (9, 10, 12), "ring": (13, 14, 16), "pinky": (17, 18, 20)}
CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10),
               (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18),
               (18, 19), (19, 20), (0, 17)]


@contextlib.contextmanager
def _quiet_stderr():
    """MediaPipe/LiteRT log from C++ straight to fd 2; Python-level settings can't silence it."""
    sys.stderr.flush()
    saved = os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(saved, 2)
        os.close(devnull)
        os.close(saved)


# ---------------------------------------------------------------- smoothing
class OneEuro:
    """One Euro filter: smooth when still, responsive when moving (Casiez et al. 2012)."""

    def __init__(self, min_cutoff: float = 1.2, beta: float = 0.02, d_cutoff: float = 1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.x = self.dx = self.t = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x: np.ndarray, t: float) -> np.ndarray:
        x = np.asarray(x, np.float64)
        if self.x is None:
            self.x, self.dx, self.t = x, np.zeros_like(x), t
            return x
        dt = max(t - self.t, 1e-3)
        dx = (x - self.x) / dt
        self.dx = self.dx + self._alpha(self.d_cutoff, dt) * (dx - self.dx)
        cutoff = self.min_cutoff + self.beta * float(np.linalg.norm(self.dx))
        self.x = self.x + self._alpha(cutoff, dt) * (x - self.x)
        self.t = t
        return self.x


# ---------------------------------------------------------------- classification
def palm_size(lm: np.ndarray) -> float:
    return float(np.linalg.norm(lm[0] - lm[9])) + 1e-6


def finger_extended(lm: np.ndarray, finger: str) -> bool:
    mcp, pip, tip = FINGERS[finger]
    # extended: tip is clearly further from the wrist than the PIP joint
    return np.linalg.norm(lm[tip] - lm[0]) > 1.15 * np.linalg.norm(lm[pip] - lm[0])


def pinch_ratio(lm: np.ndarray) -> float:
    return float(np.linalg.norm(lm[4] - lm[8])) / palm_size(lm)


def classify_pose(lm: np.ndarray) -> str:
    """Static pose from 21x2 landmarks: open, point, fist or other (pinch is handled separately)."""
    ext = {f: finger_extended(lm, f) for f in FINGERS}
    n = sum(ext.values())
    if n == 4:
        return "open"
    if ext["index"] and n == 1:
        return "point"
    if n == 0:
        return "fist"
    return "other"


# ---------------------------------------------------------------- tracking
def palm_centre(lm: np.ndarray) -> np.ndarray:
    """Centre of the palm (wrist + finger bases): stays put when the hand opens or closes."""
    return lm[[0, 5, 9, 13, 17]].mean(axis=0)


@dataclass
class Hand:
    side: str                  # "Left" / "Right" (correct for a mirrored selfie view)
    lm: np.ndarray             # 21x2 pixel coordinates (raw)
    score: float
    pose: str = "other"
    pinching: bool = False
    pinch_started: bool = False
    pinch_ended: bool = False
    cursor: np.ndarray = field(default_factory=lambda: np.zeros(2))  # smoothed pinch / fingertip point
    fist: bool = False
    fist_started: bool = False
    fist_ended: bool = False
    palm: np.ndarray = field(default_factory=lambda: np.zeros(2))    # smoothed palm centre (grab point)
    roll: float = 0.0          # smoothed, unwrapped in-plane hand angle (wrist -> middle knuckle), rad
    missing: int = 0


class _HandState:
    def __init__(self):
        self.filter = OneEuro()
        self.palm_filter = OneEuro()
        self.roll_filter = OneEuro(min_cutoff=0.8, beta=0.3)
        self.roll_raw: float | None = None
        self.roll_unwrapped = 0.0
        self.pinching = False
        self.pinch_frames = 0
        self.release_frames = 0
        self.fist = False
        self.fist_frames = 0
        self.unfist_frames = 0
        self.pose_hist: list[str] = []
        self.missing = 0
        self.last: Hand | None = None


class HandTracker:
    """MediaPipe hands in VIDEO mode, with pinch / fist hysteresis and smoothing per hand."""

    # Pinch: thumb-index distance / palm size. Must fall below PINCH_ON for PINCH_FRAMES frames in
    # a row to start, and rise above PINCH_OFF for 2 frames to end (wide gap = strokes don't drop).
    PINCH_ON, PINCH_OFF, PINCH_FRAMES = 0.20, 0.45, 4
    FIST_FRAMES, UNFIST_FRAMES = 3, 3  # consecutive frames to close / open a fist
    MAX_MISSING = 6                    # frames a hand may vanish before it's treated as gone

    def __init__(self, num_hands: int = 2, min_conf: float = 0.5, track_conf: float = 0.3,
                 reach: float = 0.0):
        """reach: margin (fraction of the frame) mapped off-canvas, so the middle (1 - 2*reach)
        of the camera view covers the whole canvas. Reaching the canvas edge then doesn't need the
        hand at the camera's edge, where it's cut off and seen edge-on.
        track_conf: low = keep following a hand frame-to-frame at odd angles instead of dropping it
        and re-running palm detection (slower to recover, and fails on fast motion)."""
        import mediapipe as mp
        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python.core.base_options import BaseOptions

        path = MODELS_DIR / HAND_MODEL
        if not path.exists():
            raise FileNotFoundError(f"{path} missing. Run: python scripts/fetch_models.py")
        self._mp = mp
        opts = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(path)),
            running_mode=vision.RunningMode.VIDEO, num_hands=num_hands,
            min_hand_detection_confidence=min_conf, min_hand_presence_confidence=min_conf,
            min_tracking_confidence=track_conf)
        with _quiet_stderr():
            self.lm = vision.HandLandmarker.create_from_options(opts)
            self._warm = False
        self.states: dict[str, _HandState] = {}
        self._last_ts = -1
        self.reach = reach
        self.frame_size: tuple[int, int] | None = None

    def to_canvas(self, p: np.ndarray) -> np.ndarray:
        """Map a camera-pixel point through the reach box to canvas pixels."""
        if not getattr(self, "reach", 0) or self.frame_size is None:
            return p
        w, h = self.frame_size
        m = np.array([w, h], float) * self.reach
        return (p - m) / (1 - 2 * self.reach)

    def reach_rect(self) -> tuple[int, int, int, int] | None:
        if not getattr(self, "reach", 0) or self.frame_size is None:
            return None
        w, h = self.frame_size
        return (int(w * self.reach), int(h * self.reach), int(w * (1 - self.reach)), int(h * (1 - self.reach)))

    def __call__(self, frame_bgr: np.ndarray, t: float) -> list[Hand]:
        h, w = frame_bgr.shape[:2]
        self.frame_size = (w, h)
        ts = max(int(t * 1000), self._last_ts + 1)  # VIDEO mode needs strictly increasing ms
        self._last_ts = ts
        img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB,
                             data=cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        if not self._warm:  # first inference prints a one-off NORM_RECT warning
            with _quiet_stderr():
                res = self.lm.detect_for_video(img, ts)
            self._warm = True
        else:
            res = self.lm.detect_for_video(img, ts)
        raw = []
        for lms, hd in zip(res.hand_landmarks, res.handedness):
            pts = np.array([[p.x * w, p.y * h] for p in lms], np.float64)
            raw.append((hd[0].category_name, float(hd[0].score), pts))
        return self.update(raw, t)

    def update(self, raw: list[tuple[str, float, np.ndarray]], t: float) -> list[Hand]:
        """Model-independent part (unit-testable): raw (side, score, 21x2) -> stateful Hands."""
        seen = set()
        # if MediaPipe reports two hands with the same side, disambiguate the second one
        out = []
        for side, score, pts in raw:
            key = side if side not in seen else side + "2"
            seen.add(key)
            st = self.states.setdefault(key, _HandState())
            st.missing = 0
            raw_pose = classify_pose(pts)
            r = pinch_ratio(pts)

            # fist (grab): all four fingers curled, held for a few frames
            was_fist = st.fist
            if not st.fist:
                st.fist_frames = st.fist_frames + 1 if raw_pose == "fist" else 0
                if st.fist_frames >= self.FIST_FRAMES:
                    st.fist, st.unfist_frames = True, 0
            else:
                st.unfist_frames = st.unfist_frames + 1 if raw_pose != "fist" else 0
                if st.unfist_frames >= self.UNFIST_FRAMES:
                    st.fist, st.fist_frames = False, 0

            # pinch (draw): never while the hand is a fist, where the thumb rests on the index
            was = st.pinching
            curled = raw_pose == "fist" or st.fist
            if not st.pinching:
                st.pinch_frames = st.pinch_frames + 1 if (r < self.PINCH_ON and not curled) else 0
                if st.pinch_frames >= self.PINCH_FRAMES:
                    st.pinching, st.release_frames = True, 0
            else:
                st.release_frames = st.release_frames + 1 if (r > self.PINCH_OFF or curled) else 0
                if st.release_frames >= 2:
                    st.pinching, st.pinch_frames = False, 0

            # interaction point: between thumb and index tips when pinching, index tip otherwise
            target = (pts[4] + pts[8]) / 2 if (st.pinching or r < self.PINCH_OFF) else pts[8]
            st.pose_hist = (st.pose_hist + [raw_pose])[-5:]
            pose = max(set(st.pose_hist), key=st.pose_hist.count)  # majority vote, kills flicker
            hand = Hand(side=key, lm=pts, score=score, pose=pose, pinching=st.pinching,
                        pinch_started=st.pinching and not was, pinch_ended=was and not st.pinching,
                        cursor=st.filter(self.to_canvas(target), t), fist=st.fist,
                        fist_started=st.fist and not was_fist, fist_ended=was_fist and not st.fist,
                        palm=st.palm_filter(self.to_canvas(palm_centre(pts)), t))
            hand.roll = self._roll(st, pts, t)
            st.last = hand
            out.append(hand)
        for key, st in list(self.states.items()):
            if key in seen:
                continue
            st.missing += 1
            if st.missing > self.MAX_MISSING:
                if (st.pinching or st.fist) and st.last is not None:
                    # hand vanished mid-gesture: report a release so drags/strokes finish cleanly
                    end = Hand(side=key, lm=st.last.lm, score=0.0, pose="gone",
                               pinch_ended=st.pinching, fist_ended=st.fist, cursor=st.last.cursor,
                               palm=st.last.palm, roll=st.last.roll, missing=st.missing)
                    out.append(end)
                del self.states[key]
        return out

    @staticmethod
    def _roll(st: _HandState, pts: np.ndarray, t: float) -> float:
        v = pts[9] - pts[0]
        a = math.atan2(v[1], v[0])
        if st.roll_raw is not None:  # unwrap so turning past +/-180 deg doesn't jump
            st.roll_unwrapped += (a - st.roll_raw + math.pi) % (2 * math.pi) - math.pi
        st.roll_raw = a
        return float(st.roll_filter(np.array([st.roll_unwrapped]), t)[0])

    def close(self) -> None:
        with _quiet_stderr():
            self.lm.close()


def draw_hand(img: np.ndarray, hand: Hand, colour=(255, 220, 120)) -> None:
    for a, b in CONNECTIONS:
        cv2.line(img, tuple(hand.lm[a].astype(int)), tuple(hand.lm[b].astype(int)), colour, 2, cv2.LINE_AA)
    for p in hand.lm:
        cv2.circle(img, tuple(p.astype(int)), 3, (255, 255, 255), -1, cv2.LINE_AA)


class HandWorker:
    """Runs the hand tracker on its own thread, on the newest camera frame, while the main thread
    draws. Results are queued in order so edge events (pinch/fist start and end) are never lost."""

    def __init__(self, tracker: HandTracker, cap):
        import queue
        self.tracker, self.cap = tracker, cap
        self.results: "queue.Queue[tuple[list[Hand], float]]" = queue.Queue()
        self.running = True
        self.hz = 0.0
        self._thread = threading.Thread(target=self._loop, name="hands", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        seq, last = 0, time.time()
        while self.running:
            frame, new_seq, t = self.cap.wait_new(seq)
            if frame is None or new_seq == seq:
                if not self.cap.running:
                    break
                continue
            seq = new_seq
            hs = self.tracker(frame, t)
            self.results.put((hs, t))
            now = time.time()
            self.hz = 0.9 * self.hz + 0.1 / max(now - last, 1e-6)
            last = now

    def drain(self) -> list[tuple[list[Hand], float]]:
        out = []
        while not self.results.empty():
            out.append(self.results.get_nowait())
        return out

    def stop(self) -> None:
        self.running = False
        self._thread.join(timeout=1.0)
