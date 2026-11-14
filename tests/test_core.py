"""Model-free unit tests: python -m unittest discover tests"""
import unittest

import numpy as np

from meerkat.config import DEFAULT_THRESHOLDS, DistanceCalib, TrackerConfig
from meerkat.detect import Detection
from meerkat.distance import DistanceEstimator, focal_from_hfov, solve_k
from meerkat.overlay import confidence
from meerkat.tracker import Tracker, iou

TH = DEFAULT_THRESHOLDS["sface"]


def det(x, y, w, score=0.95):
    box = np.array([x, y, w, w * 1.2], np.float32)
    lm = np.array([[x + .3 * w, y + .4 * w], [x + .7 * w, y + .4 * w], [x + .5 * w, y + .6 * w],
                   [x + .35 * w, y + .85 * w], [x + .65 * w, y + .85 * w]], np.float32)
    return Detection(box, lm, score, np.zeros(15, np.float32))


def feed(tracker, d, sim):
    tr = tracker.step([d] if d is not None else [])
    for t in tr:
        if t.det is not None:
            e = np.ones(4, np.float32) / 2
            t.observe(e, sim, lambda _: sim, TH)
    return tr


class TestTracker(unittest.TestCase):
    def test_iou(self):
        a = np.array([0, 0, 10, 10.0])
        self.assertAlmostEqual(iou(a, a), 1.0)
        self.assertEqual(iou(a, np.array([20, 20, 5, 5.0])), 0.0)

    def test_id_stable_while_moving(self):
        tk = Tracker()
        ids = {feed(tk, det(100 + 3 * i, 100, 150), 0.0)[0].id for i in range(30)}
        self.assertEqual(ids, {1})

    def test_lock_hold_small_and_release(self):
        cfg = TrackerConfig()
        tk = Tracker(cfg)
        for _ in range(cfg.lock_frames):
            t = feed(tk, det(500, 300, 200), 0.7)[0]
        self.assertTrue(t.locked)
        # walking away: similarity collapses once the face is too small to judge -> stays locked
        for i in range(40):
            w = max(200 - 6 * i, 30)
            sim = 0.7 if w >= cfg.min_judge_px else 0.05
            t = feed(tk, det(500 + (200 - w) / 2, 300, w), sim)[0]
        self.assertTrue(t.locked)
        self.assertEqual(t.state, "tracked")  # locked, but per-frame recognition lost
        # a large face that clearly isn't the person releases the lock after M frames
        tk2 = Tracker(cfg)
        for _ in range(cfg.lock_frames):
            feed(tk2, det(500, 300, 200), 0.7)
        for _ in range(cfg.release_frames + cfg.embed_window):
            t2 = feed(tk2, det(500, 300, 200), 0.0)[0]
        self.assertFalse(t2.locked)

    def test_no_lock_below_high(self):
        tk = Tracker()
        for _ in range(50):
            t = feed(tk, det(500, 300, 200), TH.lock_high - 0.01)[0]
        self.assertFalse(t.locked)

    def test_track_dropped_after_max_missed(self):
        cfg = TrackerConfig(max_missed=3)
        tk = Tracker(cfg)
        feed(tk, det(100, 100, 100), 0.0)
        removed = []
        for _ in range(5):
            feed(tk, None, 0.0)
            removed += tk.removed
        self.assertEqual(len(tk.tracks), 0)
        self.assertEqual(len(removed), 1)
        self.assertEqual(removed[0].face_px, 100)


class TestDistance(unittest.TestCase):
    def test_focal(self):
        self.assertAlmostEqual(focal_from_hfov(1280, 65), 1004.6, delta=1)

    def test_uncalibrated_matches_table(self):
        d = DistanceEstimator(DistanceCalib(), 1280)
        self.assertAlmostEqual(d.estimate(144, 1e9, yaw=1.0), 1.0, delta=0.02)

    def test_calibrated_and_resolution_scaling(self):
        k_box, k_eye = solve_k([200] * 5, [90] * 5, 1.0)
        c = DistanceCalib(k_box=k_box, k_eye=k_eye, calibrated_at_width=1920)
        self.assertAlmostEqual(DistanceEstimator(c, 1920).estimate(100, 45), 2.0, places=5)
        self.assertAlmostEqual(DistanceEstimator(c, 1280).estimate(100 * 2 / 3, 45 * 2 / 3), 2.0, places=5)


class TestConfidence(unittest.TestCase):
    def test_ramp(self):
        self.assertEqual(confidence(None, TH), 0.0)
        self.assertEqual(confidence(-1, TH), 0.0)
        self.assertEqual(confidence(1, TH), 1.0)
        self.assertAlmostEqual(confidence(TH.match, TH), 0.5, delta=0.05)


if __name__ == "__main__":
    unittest.main()
