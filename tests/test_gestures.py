"""Model-free tests for shape recognition, hand pose / pinch logic and canvas interactions."""
import math
import unittest

import numpy as np

from meerkat.gestures import CLEAR_HOLD, Canvas, PALETTE
from meerkat.hands import Hand, HandTracker, classify_pose, pinch_ratio
from meerkat.shapes import Item, recognise

RNG = np.random.default_rng(1)


def noisy(pts, sd=3.0):
    pts = np.asarray(pts, float)
    return pts + RNG.normal(0, sd, pts.shape)


def poly_stroke(corners, per_side=25, overshoot=0.0):
    c = np.asarray(corners, float)
    out = []
    for a, b in zip(c, np.roll(c, -1, axis=0)):
        for t in np.linspace(0, 1, per_side, endpoint=False):
            out.append(a + (b - a) * t)
    out.append(c[0] + (c[1] - c[0]) * overshoot)
    return np.array(out)


class TestShapes(unittest.TestCase):
    def test_circle(self):
        t = np.linspace(0, 2 * np.pi * 1.03, 80)
        kind, pts = recognise(noisy(np.stack([400 + 120 * np.cos(t), 300 + 120 * np.sin(t)], 1)))
        self.assertEqual(kind, "circle")
        r = np.linalg.norm(pts - pts.mean(0), axis=1)
        self.assertAlmostEqual(r.mean(), 120, delta=10)

    def test_ellipse(self):
        t = np.linspace(0, 2 * np.pi, 80)
        kind, _ = recognise(noisy(np.stack([400 + 200 * np.cos(t), 300 + 90 * np.sin(t)], 1)))
        self.assertEqual(kind, "ellipse")

    def test_rect_snaps_axis_aligned(self):
        kind, pts = recognise(noisy(poly_stroke([(100, 100), (400, 105), (395, 300), (102, 296)])))
        self.assertEqual(kind, "rect")
        xs, ys = sorted(set(np.round(pts[:, 0]))), sorted(set(np.round(pts[:, 1])))
        self.assertEqual((len(xs), len(ys)), (2, 2))  # snapped: only 2 distinct x and y values

    def test_triangle(self):
        kind, pts = recognise(noisy(poly_stroke([(300, 100), (450, 350), (150, 350)])))
        self.assertEqual(kind, "triangle")
        self.assertEqual(len(pts), 3)

    def test_line(self):
        kind, pts = recognise(noisy(np.linspace((100, 100), (500, 260), 40), 2))
        self.assertEqual(kind, "line")
        self.assertEqual(len(pts), 2)

    def test_scribble_is_free(self):
        x = np.linspace(100, 500, 120)
        kind, _ = recognise(np.stack([x, 300 + 60 * np.sin(x / 15)], 1))
        self.assertEqual(kind, "free")

    def test_tap_is_dot(self):
        self.assertEqual(recognise(np.array([[10, 10], [12, 11]]))[0], "dot")

    def test_item_hit_and_transform(self):
        sq = Item("rect", np.array([[0, 0], [100, 0], [100, 100], [0, 100]], float), (0, 0, 0))
        self.assertEqual(sq.distance(np.array([50, 50])), 0.0)
        self.assertAlmostEqual(sq.distance(np.array([120, 50])), 20, delta=0.5)
        base = sq.pts.copy()
        sq.transformed(base, np.array([50, 50.0]), 2.0, math.pi / 2, np.array([10, 0.0]))
        self.assertTrue(np.allclose(sq.centre, [60, 50]))
        self.assertAlmostEqual(np.ptp(sq.pts[:, 0]), 200, delta=1e-6)


# ---------------------------------------------------------------- synthetic hands
def hand_pts(extended=("index", "middle", "ring", "pinky"), pinch=False, at=(0.0, 0.0)):
    """Upright hand, wrist at origin, palm size ~80 px. Returns 21x2."""
    p = np.zeros((21, 2))
    p[1:5] = [(-25, -15), (-45, -35), (-60, -50), (-70, -62)]  # thumb
    for (mcp, pip, dip, tip), x in zip([(5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20)],
                                       (-30, -10, 10, 30)):
        name = {5: "index", 9: "middle", 13: "ring", 17: "pinky"}[mcp]
        p[mcp] = (x, -80)
        if name in extended:
            p[pip], p[dip], p[tip] = (x, -110), (x, -135), (x, -155)
        else:
            p[pip], p[dip], p[tip] = (x, -100), (x, -85), (x, -70)
    if pinch:
        p[4] = p[8] + (4, 3)
    return p + np.asarray(at)


def mk(side, cursor, pinching=False, started=False, ended=False, pose="other"):
    """A pinching (or idle) hand at `cursor`."""
    return Hand(side=side, lm=np.zeros((21, 2)), score=1.0, pose=pose, pinching=pinching,
                pinch_started=started, pinch_ended=ended, cursor=np.asarray(cursor, float),
                palm=np.asarray(cursor, float) + (0, 60))


def fist(side, palm, closed=True, started=False, ended=False, roll=0.0):
    """A fist (or just-opened hand) with its palm at `palm` and wrist angle `roll`."""
    return Hand(side=side, lm=np.zeros((21, 2)), score=1.0, pose="fist" if closed else "open",
                fist=closed, fist_started=started, fist_ended=ended,
                cursor=np.asarray(palm, float) - (0, 60), palm=np.asarray(palm, float), roll=roll)


class TestHandLogic(unittest.TestCase):
    def test_poses(self):
        self.assertEqual(classify_pose(hand_pts()), "open")
        self.assertEqual(classify_pose(hand_pts(("index",))), "point")
        self.assertEqual(classify_pose(hand_pts(())), "fist")

    def test_pinch_ratio(self):
        self.assertGreater(pinch_ratio(hand_pts(("index",))), 0.6)
        self.assertLess(pinch_ratio(hand_pts(("index",), pinch=True)), 0.15)

    @staticmethod
    def tracker():
        ht = HandTracker.__new__(HandTracker)  # skip model loading
        ht.states, ht.reach, ht.frame_size = {}, 0.0, None
        return ht

    def run_seq(self, ht, frames, t0=0.0):
        hs = []
        for i, pts in enumerate(frames):
            hs += ht.update([("Right", 0.9, pts)], t0 + i / 30)
        return hs

    def test_pinch_needs_sustained_close_fingers(self):
        ht = self.tracker()
        P, O = hand_pts(("index",), pinch=True), hand_pts(("index",))
        hs = self.run_seq(ht, [P, P, P, O, P, P, P, O, O])  # never 4 frames in a row
        self.assertFalse(any(h.pinch_started for h in hs))
        near = hand_pts(("index",))
        near[4] = near[8] + (20, 0)  # 0.25 x palm: would have counted before, not now
        self.assertFalse(any(h.pinching for h in self.run_seq(self.tracker(), [near] * 8)))

    def test_pinch_hysteresis_and_lost_hand(self):
        ht = self.tracker()
        P, O = hand_pts(("index",), pinch=True), hand_pts(("index",))
        hs = self.run_seq(ht, [O, P, P, P, P, P, O, P, P, O, O, O])
        self.assertEqual(sum(h.pinch_started for h in hs), 1)
        self.assertEqual(sum(h.pinch_ended for h in hs), 1)  # one-frame glitch doesn't break it
        self.run_seq(ht, [P] * 5, t0=1.0)
        gone = []
        for i in range(HandTracker.MAX_MISSING + 1):
            gone += ht.update([], 1.1 + i / 30)
        self.assertEqual(len(gone), 1)
        self.assertTrue(gone[0].pinch_ended)


    def test_reach_box_maps_inner_area_to_whole_canvas(self):
        ht = self.tracker()
        ht.reach, ht.frame_size = 0.1, (1280, 720)
        self.assertTrue(np.allclose(ht.to_canvas(np.array([128.0, 72.0])), (0, 0)))        # box corner
        self.assertTrue(np.allclose(ht.to_canvas(np.array([1152.0, 648.0])), (1280, 720)))  # opposite corner
        self.assertTrue(np.allclose(ht.to_canvas(np.array([640.0, 360.0])), (640, 360)))   # centre unchanged
        self.assertEqual(ht.reach_rect(), (128, 72, 1152, 648))
        # cursor and palm come out in canvas space; landmarks stay in camera space for drawing
        pts = hand_pts(("index",), at=(1100, 400))
        h = ht.update([("Right", 0.9, pts)], 0.0)[0]
        self.assertGreater(h.palm[0], pts[[0, 5, 9, 13, 17], 0].mean())
        self.assertTrue(np.allclose(h.lm, pts))

    def test_roll_unwraps_past_180(self):
        ht = self.tracker()
        base = hand_pts()
        rolls = []
        for i, ang in enumerate(np.linspace(0, 2 * np.pi, 60)):  # rotate the hand a full turn
            c, s_ = np.cos(ang), np.sin(ang)
            pts = base @ np.array([[c, -s_], [s_, c]]).T + (500, 500)
            rolls.append(ht.update([("Right", 0.9, pts)], i / 30)[0].roll)
        self.assertTrue(np.all(np.diff(rolls) > -1e-6))  # monotonic: no jump at +/-180 deg
        self.assertGreater(rolls[-1] - rolls[0], 4.0)      # filter lags a little behind 2*pi

    def test_fist_detected_and_blocks_pinch(self):
        ht = self.tracker()
        F = hand_pts(())
        F[4] = F[8] + (3, 0)  # thumb resting on the curled index: looks like a pinch
        hs = self.run_seq(ht, [F] * 6)
        self.assertEqual(sum(h.fist_started for h in hs), 1)
        self.assertFalse(any(h.pinching for h in hs))
        hs = self.run_seq(ht, [hand_pts()] * 4, t0=1)
        self.assertEqual(sum(h.fist_ended for h in hs), 1)


class TestCanvas(unittest.TestCase):
    def grab(self, cv, side, path, t0=0.0):
        """Fist at path[0], move along path, open the hand at the end."""
        cv.update([fist(side, path[0], started=True)], t0)
        for i, p in enumerate(path[1:], 1):
            cv.update([fist(side, p)], t0 + i / 30)
        cv.update([fist(side, path[-1], closed=False, ended=True)], t0 + (len(path) + 1) / 30)

    def drag(self, cv, side, path, t0=0.0):
        cv.update([mk(side, path[0], True, started=True)], t0)
        for i, p in enumerate(path[1:], 1):
            cv.update([mk(side, p, True)], t0 + i / 30)
        cv.update([mk(side, path[-1], False, ended=True)], t0 + (len(path) + 1) / 30)

    def circle_path(self, cx=600, cy=360, r=100):
        t = np.linspace(0, 2 * np.pi * 1.02, 60)
        return list(np.stack([cx + r * np.cos(t), cy + r * np.sin(t)], 1))

    def test_draw_move_delete_undo(self):
        cv = Canvas(1280, 720)
        self.drag(cv, "Right", self.circle_path())
        self.assertEqual([i.kind for i in cv.items], ["circle"])
        c0 = cv.items[0].centre.copy()
        self.grab(cv, "Right", list(np.linspace((600, 360), (800, 400), 20)), t0=5)
        self.assertTrue(np.allclose(cv.items[0].centre - c0, (200, 40), atol=1))
        x0, y0, x1, y1 = cv.bin
        self.grab(cv, "Right", list(np.linspace((800, 400), ((x0 + x1) / 2, (y0 + y1) / 2), 20)), t0=10)
        self.assertEqual(cv.items, [])
        cv.undo()
        self.assertEqual(len(cv.items), 1)

    def test_grab_with_several_items(self):
        # regression: grabbing any item but the first crashed (dataclass __eq__ on numpy arrays)
        cv = Canvas(1280, 720)
        for i, cx in enumerate((300, 600, 900)):
            self.drag(cv, "Right", self.circle_path(cx, 360, 80), t0=i * 5)
        self.assertEqual(len(cv.items), 3)
        target = cv.items[1]
        self.grab(cv, "Right", list(np.linspace((600, 360), (600, 560), 15)), t0=20)
        self.assertIs(cv.items[-1], target)  # brought to front
        self.assertAlmostEqual(target.centre[1], 560, delta=2)
        x0, y0, x1, y1 = cv.bin
        self.grab(cv, "Right", list(np.linspace((600, 560), ((x0 + x1) / 2, (y0 + y1) / 2), 15)), t0=30)
        self.assertEqual(len(cv.items), 2)
        self.assertNotIn(target, cv.items)

    def test_two_hand_scale(self):
        cv = Canvas(1280, 720)
        self.drag(cv, "Right", self.circle_path(r=100))
        size0 = np.ptp(cv.items[0].pts[:, 0])
        cv.update([fist("Left", (560, 360), started=True)], 1.0)
        cv.update([fist("Left", (560, 360)), fist("Right", (640, 360), started=True)], 1.03)
        cv.update([fist("Left", (520, 360)), fist("Right", (680, 360))], 1.06)  # spread 2x
        self.assertAlmostEqual(np.ptp(cv.items[0].pts[:, 0]) / size0, 2.0, delta=0.05)
        cv.update([fist("Left", (520, 360), closed=False, ended=True), fist("Right", (680, 360))], 1.1)
        self.assertIsNone(cv.two)
        cv.update([fist("Right", (700, 360))], 1.13)  # remaining fist carries on dragging
        self.assertAlmostEqual(cv.items[0].centre[0], 620, delta=2)

    def test_pinch_on_item_draws_instead_of_grabbing(self):
        cv = Canvas(1280, 720)
        self.drag(cv, "Right", self.circle_path(r=100))
        c0 = cv.items[0].centre.copy()
        self.drag(cv, "Right", list(np.linspace((600, 360), (900, 360), 20)), t0=5)
        self.assertTrue(np.allclose(cv.items[0].centre, c0))   # circle didn't move
        self.assertEqual([i.kind for i in cv.items], ["circle", "line"])

    def test_one_hand_twist_rotates_item(self):
        cv = Canvas(1280, 720)
        sq = [np.array(p, float) for p in np.concatenate([
            np.linspace((500, 260), (700, 260), 12), np.linspace((700, 260), (700, 460), 12),
            np.linspace((700, 460), (500, 460), 12), np.linspace((500, 460), (500, 262), 12)])]
        self.drag(cv, "Right", sq)
        it = cv.items[0]
        self.assertEqual(it.kind, "rect")
        e0 = it.pts[1] - it.pts[0]
        cv.update([fist("Right", (600, 360), started=True, roll=0.0)], 10)
        for i, r in enumerate(np.linspace(0, 0.4, 10), 1):  # wrist turns 0.4 rad
            cv.update([fist("Right", (600, 360), roll=r)], 10 + i / 30)
        e1 = it.pts[1] - it.pts[0]
        turned = np.arctan2(e1[1], e1[0]) - np.arctan2(e0[1], e0[0])
        self.assertAlmostEqual(turned, 0.4 * 1.5, delta=0.05)   # TWIST_GAIN
        self.assertTrue(np.allclose(it.centre, (600, 360), atol=2))

    def swipe(self, cv, start, step, n, t0, release=True, fps=30):
        """Fist at `start`, move `step` px per frame for n frames, optionally open the hand."""
        p = np.array(start, float)
        cv.update([fist("Right", p, started=True)], t0)
        t = t0
        for _ in range(n):
            p = p + step
            t += 1 / fps
            cv.update([fist("Right", p)], t)
        if release:
            cv.update([fist("Right", p, closed=False, ended=True)], t + 1 / fps)
        return t

    def test_fast_swipe_off_screen_throws_away(self):
        cv = Canvas(1280, 720)
        self.drag(cv, "Right", self.circle_path(600, 360, 80))
        t = self.swipe(cv, (600, 360), (60, 0), 8, t0=5)  # 1800 px/s towards the right edge
        self.assertEqual(cv.items, [])
        self.assertEqual(len(cv.flying), 1)
        cv.update([], t + 1.0)
        self.assertEqual(cv.flying, [])  # animation finished
        cv.undo()
        self.assertEqual(len(cv.items), 1)
        self.assertTrue(np.allclose(cv.items[0].centre, (600, 360), atol=2))  # back where it was

    def test_slow_drag_to_edge_is_just_a_move(self):
        cv = Canvas(1280, 720)
        self.drag(cv, "Right", self.circle_path(600, 360, 80))
        self.swipe(cv, (600, 360), (12, 0), 40, t0=5)  # 360 px/s, ends near the edge
        self.assertEqual(len(cv.items), 1)

    def test_fast_swipe_towards_centre_is_not_a_throw(self):
        cv = Canvas(1280, 720)
        self.drag(cv, "Right", self.circle_path(200, 360, 80))
        self.swipe(cv, (200, 360), (60, 0), 6, t0=5)  # fast, but ends mid-screen far from an edge
        self.assertEqual(len(cv.items), 1)

    def test_fast_move_that_stops_is_not_a_throw(self):
        cv = Canvas(1280, 720)
        self.drag(cv, "Right", self.circle_path(600, 360, 80))
        t = self.swipe(cv, (600, 360), (60, 0), 8, t0=5, release=False)  # fast towards the edge...
        p = np.array([1080.0, 360.0])
        for i in range(1, 5):                                             # ...then stops
            cv.update([fist("Right", p)], t + i / 30)
        cv.update([fist("Right", p, closed=False, ended=True)], t + 5 / 30)
        self.assertEqual(len(cv.items), 1)

    def test_hand_lost_mid_swipe_throws(self):
        cv = Canvas(1280, 720)
        self.drag(cv, "Right", self.circle_path(600, 360, 80))
        t = self.swipe(cv, (600, 360), (35, 5), 10, t0=5, release=False)  # ~1060 px/s, then blurs out
        lost = Hand(side="Right", lm=np.zeros((21, 2)), score=0.0, pose="gone", fist_ended=True,
                    palm=np.array([950.0, 410.0]), cursor=np.array([950.0, 350.0]))
        cv.update([lost], t + 0.2)  # tracker reports the release ~6 frames later
        self.assertEqual(cv.items, [])

    def test_toolbar_buttons(self):
        cv = Canvas(1280, 720)
        self.drag(cv, "Right", self.circle_path())
        def tap(name, t):
            x0, y0, x1, y1 = cv.buttons[name]
            c = ((x0 + x1) / 2, (y0 + y1) / 2)
            cv.update([mk("Right", c, True, started=True)], t)
            cv.update([mk("Right", c, False, ended=True)], t + 0.05)
        tap("hand", 5)
        self.assertFalse(cv.skeleton)
        tap("save", 6)
        self.assertTrue(cv.save_requested)
        tap("undo", 7)
        self.assertEqual(cv.items, [])
        self.assertEqual(len(cv.undo_stack), 0)

    def test_exit_needs_a_hold(self):
        cv = Canvas(1280, 720)
        x0, y0, x1, y1 = cv.buttons["exit"]
        c = ((x0 + x1) / 2, (y0 + y1) / 2)
        cv.update([mk("Right", c, True, started=True)], 0)
        cv.update([mk("Right", c, True)], 0.5)
        cv.update([mk("Right", c, False, ended=True)], 0.55)  # released early: no quit
        self.assertFalse(cv.quit)
        cv.update([mk("Right", c, True, started=True)], 1)
        cv.update([mk("Right", (600, 400), True)], 1.5)       # slid off: cancelled
        cv.update([mk("Right", (600, 400), True)], 2.5)
        self.assertFalse(cv.quit)
        cv.update([mk("Right", (600, 400), False, ended=True)], 2.6)
        cv.update([mk("Right", c, True, started=True)], 3)
        for t in np.arange(3.05, 4.2, 0.05):
            cv.update([mk("Right", c, True)], t)
        self.assertTrue(cv.quit)

    def test_fist_on_empty_space_does_nothing(self):
        cv = Canvas(1280, 720)
        self.grab(cv, "Right", list(np.linspace((300, 300), (500, 300), 10)))
        self.assertEqual(cv.items, [])

    def test_swatch_and_clear(self):
        cv = Canvas(1280, 720)
        (x, y), _, colour = cv.swatches[2]
        cv.update([mk("Right", (x, y), True, started=True)], 0)
        cv.update([mk("Right", (x, y), False, ended=True)], 0.05)
        self.assertEqual(cv.colour, PALETTE[2])
        self.assertEqual(cv.items, [])  # tapping a swatch must not draw anything
        self.drag(cv, "Right", self.circle_path(), t0=1)
        self.assertEqual(cv.items[0].colour, PALETTE[2])
        t = 5.0
        while t < 5 + CLEAR_HOLD + 0.2:
            cv.update([mk("Left", (300, 300), pose="open"), mk("Right", (900, 300), pose="open")], t)
            t += 0.05
        self.assertEqual(cv.items, [])

    def test_disabled_ignores_gestures(self):
        cv = Canvas(1280, 720)
        path = self.circle_path()
        cv.update([mk("Right", path[0], True, started=True)], 0, enabled=False)
        for p in path[1:]:
            cv.update([mk("Right", p, True)], 0.1, enabled=False)
        cv.update([mk("Right", path[-1], False, ended=True)], 0.2, enabled=False)
        self.assertEqual(cv.items, [])


if __name__ == "__main__":
    unittest.main()
