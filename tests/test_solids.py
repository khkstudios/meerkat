"""3D solids: construction from 2D shapes, rendering sanity, and canvas interactions in 3D mode."""
import unittest

import numpy as np

from meerkat.gestures import Canvas
from meerkat.shapes import circle_points
from meerkat.solids import Solid
from tests import test_gestures as tg
from tests.test_gestures import fist, mk

SQUARE = np.array([[400, 200], [600, 200], [600, 400], [400, 400]], float)


class TestSolids(unittest.TestCase):
    def test_mapping(self):
        cases = {"circle": (circle_points(300, 300, 100), "sphere"),
                 "rect": (SQUARE, "cube"),
                 "triangle": (np.array([[300, 100], [400, 300], [200, 300]], float), "pyramid"),
                 "line": (np.array([[100, 100], [400, 200]], float), "cylinder")}
        for kind, (pts, want) in cases.items():
            s = Solid.from_shape(kind, pts, (255, 0, 0))
            self.assertEqual(s.kind, want, kind)
            self.assertTrue(np.allclose(s.R @ s.R.T, np.eye(3), atol=1e-9))
        box = Solid.from_shape("rect", np.array([[0, 0], [300, 0], [300, 100], [0, 100]], float), (0, 0, 0))
        self.assertEqual(box.kind, "box")

    def test_size_matches_drawing(self):
        s = Solid.from_shape("circle", circle_points(300, 300, 100), (0, 0, 0))
        self.assertAlmostEqual(np.ptp(s.pts[:, 0]), 200, delta=8)
        self.assertTrue(np.allclose(s.centre, (300, 300), atol=2))

    def test_cube_shows_three_faces_and_draws(self):
        s = Solid.from_shape("rect", SQUARE, (0, 200, 0))
        img = np.zeros((600, 800, 3), np.uint8)
        s.draw(img)
        lit = img.reshape(-1, 3).any(1)
        self.assertGreater(lit.sum(), 30000)                     # something substantial drawn
        self.assertEqual(len({tuple(c) for c in img.reshape(-1, 3)[lit] if c[1] > 60}) > 2, True)

    def test_hit_rotate_spin(self):
        s = Solid.from_shape("rect", SQUARE, (0, 0, 0))
        self.assertEqual(s.distance(np.array([500, 300])), 0.0)
        self.assertGreater(s.distance(np.array([900, 300])), 100)
        R0 = s.R.copy()
        for _ in range(200):
            s.rotate_by_drag(7, -3)
        self.assertFalse(np.allclose(s.R, R0))
        self.assertTrue(np.allclose(s.R @ s.R.T, np.eye(3), atol=1e-9))  # no drift
        s.spin = np.array([0.1, 0.0])
        for _ in range(400):
            s.step_spin()
        self.assertLess(np.abs(s.spin).max(), 1e-4)  # momentum dies away


class TestCanvas3D(unittest.TestCase):
    def setUp(self):
        self.cv = Canvas(1280, 720)
        self.cv.mode3d = True
        self.h = tg.TestCanvas()

    def test_toggle_button(self):
        cv = Canvas(1280, 720)
        x0, y0, x1, y1 = cv.mode_button
        cv.update([mk("Right", ((x0 + x1) / 2, (y0 + y1) / 2), True, started=True)], 0)
        cv.update([mk("Right", ((x0 + x1) / 2, (y0 + y1) / 2), False, ended=True)], 0.1)
        self.assertTrue(cv.mode3d)
        self.assertEqual(cv.items, [])

    def test_draw_rotate_move_scale_delete(self):
        cv = self.cv
        self.h.drag(cv, "Right", self.h.circle_path(600, 360, 100))
        self.assertIsInstance(cv.items[0], Solid)
        self.assertEqual(cv.items[0].kind, "sphere")
        s = cv.items[0]
        # pinch-drag on the solid rotates it (no new stroke)
        R0 = s.R.copy()
        self.h.drag(cv, "Right", list(np.linspace((600, 360), (700, 330), 15)), t0=5)
        self.assertEqual(len(cv.items), 1)
        self.assertFalse(np.allclose(s.R, R0))
        self.assertGreater(np.abs(s.spin).max(), 0)  # released with momentum
        # fist moves it
        self.h.grab(cv, "Right", list(np.linspace((600, 360), (400, 360), 15)), t0=10)
        self.assertTrue(np.allclose(s.centre, (400, 360), atol=2))
        # two fists scale it
        cv.update([fist("Left", (370, 360), started=True)], 20)
        cv.update([fist("Left", (370, 360)), fist("Right", (430, 360), started=True)], 20.03)
        cv.update([fist("Left", (340, 360)), fist("Right", (460, 360))], 20.06)
        self.assertAlmostEqual(s.scale, 2.0, delta=0.05)
        cv.update([fist("Left", (340, 360), closed=False, ended=True),
                   fist("Right", (460, 360), closed=False, ended=True)], 20.1)
        # drop on the bin
        x0, y0, x1, y1 = cv.bin
        self.h.grab(cv, "Right", list(np.linspace(s.centre, ((x0 + x1) / 2, (y0 + y1) / 2), 15)), t0=30)
        self.assertEqual(cv.items, [])
        cv.undo()
        self.assertEqual(cv.items[0].kind, "sphere")

    def test_one_hand_full_rotation(self):
        """Tilt/turn with a pinch-drag and twist with a fist: all three axes with one hand."""
        cv = self.cv
        sq = [np.array(p, float) for p in np.concatenate([
            np.linspace((500, 260), (700, 260), 12), np.linspace((700, 260), (700, 460), 12),
            np.linspace((700, 460), (500, 460), 12), np.linspace((500, 460), (500, 262), 12)])]
        self.h.drag(cv, "Right", sq)
        cube = cv.items[0]
        self.assertEqual(cube.kind, "cube")
        cube.spin[:] = 0
        R0 = cube.R.copy()
        cv.update([fist("Right", (600, 360), started=True, roll=0.0)], 10)
        for i, r in enumerate(np.linspace(0, 0.5, 10), 1):
            cv.update([fist("Right", (600, 360), roll=r)], 10 + i / 30)
        # twist is a rotation about the screen's z axis: R_new = Rz(theta) @ R0
        Rz = cube.R @ R0.T
        self.assertAlmostEqual(Rz[2, 2], 1.0, places=6)
        self.assertAlmostEqual(np.arctan2(Rz[1, 0], Rz[0, 0]), 0.5 * 1.5, delta=0.05)

    def test_freehand_stays_2d_in_3d_mode(self):
        x = np.linspace(100, 500, 60)
        self.h.drag(self.cv, "Right", list(np.stack([x, 300 + 60 * np.sin(x / 15)], 1)))
        self.assertEqual(self.cv.items[0].kind, "free")
        self.assertNotIsInstance(self.cv.items[0], Solid)


if __name__ == "__main__":
    unittest.main()
