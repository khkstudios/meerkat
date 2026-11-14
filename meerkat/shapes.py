"""Stroke -> shape recognition, and canvas items (everything is a point list, so move /
scale / rotate are the same operation for every kind)."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import count

import cv2
import numpy as np

_ids = count(1)


def path_length(pts: np.ndarray) -> float:
    return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1))) if len(pts) > 1 else 0.0


def resample(pts: np.ndarray, n: int = 64) -> np.ndarray:
    """Evenly spaced points along the stroke (removes speed-dependent point density)."""
    pts = np.asarray(pts, np.float64)
    if len(pts) < 2:
        return pts
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    d = np.concatenate([[0], np.cumsum(seg)])
    if d[-1] == 0:
        return pts[:1]
    s = np.linspace(0, d[-1], n)
    return np.stack([np.interp(s, d, pts[:, 0]), np.interp(s, d, pts[:, 1])], axis=1)


def recognise(stroke: np.ndarray) -> tuple[str, np.ndarray]:
    """Returns (kind, clean_points). kind: line | circle | ellipse | rect | triangle | free."""
    pts = resample(np.asarray(stroke, np.float64), 96)
    L = path_length(pts)
    if L < 25:
        return "dot", pts[:1]
    gap = float(np.linalg.norm(pts[0] - pts[-1]))
    closed = gap < max(30.0, 0.22 * L)

    if not closed:
        if gap / L > 0.88:
            return "line", np.array([pts[0], pts[-1]])
        return "free", pts

    hull = cv2.convexHull(pts.astype(np.float32)).reshape(-1, 2).astype(np.float64)
    P = cv2.arcLength(hull.astype(np.float32), True)
    A = cv2.contourArea(hull.astype(np.float32))
    if A < 200:
        return "free", pts
    approx = cv2.approxPolyDP(hull.astype(np.float32), 0.06 * P, True).reshape(-1, 2).astype(np.float64)
    (cx, cy), (rw, rh), ang = cv2.minAreaRect(pts.astype(np.float32))
    fill = A / max(rw * rh, 1e-6)
    circularity = 4 * math.pi * A / (P * P)

    if len(approx) == 3:
        return "triangle", approx
    if len(approx) == 4 and fill > 0.82:
        box = cv2.boxPoints(((cx, cy), (rw, rh), ang)).astype(np.float64)
        a = ang % 90
        if min(a, 90 - a) < 12:  # nearly axis-aligned: snap straight
            x0, y0 = box.min(0)
            x1, y1 = box.max(0)
            box = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
        return "rect", box
    if len(pts) >= 5:
        (ex, ey), (ea, eb), eang = cv2.fitEllipse(pts.astype(np.float32))
        # how well does the stroke follow the fitted ellipse? (mean normalised radial error)
        th = math.radians(eang)
        d = pts - (ex, ey)
        u = d[:, 0] * math.cos(th) + d[:, 1] * math.sin(th)
        v = -d[:, 0] * math.sin(th) + d[:, 1] * math.cos(th)
        err = float(np.mean(np.abs(np.hypot(u / max(ea / 2, 1e-6), v / max(eb / 2, 1e-6)) - 1)))
        if err < 0.10 or circularity > 0.85:
            if min(ea, eb) / max(ea, eb) > 0.82:
                return "circle", circle_points(ex, ey, (ea + eb) / 4)
            poly = cv2.ellipse2Poly((int(ex), int(ey)), (int(ea / 2), int(eb / 2)), int(eang), 0, 360, 6)
            return "ellipse", poly.astype(np.float64)
    return "free", pts


def circle_points(cx: float, cy: float, r: float, n: int = 72) -> np.ndarray:
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([cx + r * np.cos(t), cy + r * np.sin(t)], axis=1)


CLOSED = {"circle", "ellipse", "rect", "triangle"}


@dataclass(eq=False)  # identity equality: items hold numpy arrays, and two shapes are never 'equal'
class Item:
    kind: str
    pts: np.ndarray
    colour: tuple[int, int, int]
    thickness: int = 5
    id: int = field(default_factory=lambda: next(_ids))

    @property
    def closed(self) -> bool:
        return self.kind in CLOSED

    @property
    def centre(self) -> np.ndarray:
        return self.pts.mean(axis=0)

    def distance(self, p: np.ndarray) -> float:
        """0 inside a closed shape, else distance in px to the outline."""
        poly = self.pts.astype(np.float32)
        if self.closed and len(poly) >= 3:
            d = cv2.pointPolygonTest(poly.reshape(-1, 1, 2), (float(p[0]), float(p[1])), True)
            return max(0.0, -d)
        if len(self.pts) == 1:
            return float(np.linalg.norm(self.pts[0] - p))
        a, b = self.pts[:-1], self.pts[1:]
        ab = b - a
        tt = np.clip(np.einsum("ij,ij->i", p - a, ab) / np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-9), 0, 1)
        return float(np.min(np.linalg.norm(a + ab * tt[:, None] - p, axis=1)))

    def translate(self, d: np.ndarray) -> None:
        self.pts = self.pts + d

    def state(self) -> np.ndarray:
        return self.pts.copy()

    def apply_transform(self, base, pivot, scale, angle, shift) -> None:
        self.transformed(base, pivot, scale, angle, shift)

    def step_spin(self) -> None:
        pass

    def scale_by(self, f: float) -> None:
        ctr = self.centre
        self.pts = (self.pts - ctr) * f + ctr

    def rotate_inplane(self, angle: float) -> None:
        c, s = math.cos(angle), math.sin(angle)
        ctr = self.centre
        self.pts = (self.pts - ctr) @ np.array([[c, -s], [s, c]]).T + ctr

    def transformed(self, base: np.ndarray, pivot: np.ndarray, scale: float, angle: float,
                    shift: np.ndarray) -> None:
        """Set pts = rotate/scale `base` about `pivot`, then shift (two-hand manipulation)."""
        c, s = math.cos(angle), math.sin(angle)
        R = np.array([[c, -s], [s, c]]) * scale
        self.pts = (base - pivot) @ R.T + pivot + shift

    def draw(self, img: np.ndarray, highlight: bool = False) -> None:
        pts = self.pts.astype(np.int32).reshape(-1, 1, 2)
        if highlight:
            cv2.polylines(img, [pts], self.closed, (255, 255, 255), self.thickness + 6, cv2.LINE_AA)
        if len(self.pts) == 1:
            cv2.circle(img, tuple(self.pts[0].astype(int)), self.thickness, self.colour, -1, cv2.LINE_AA)
        else:
            cv2.polylines(img, [pts], self.closed, self.colour, self.thickness, cv2.LINE_AA)

    def to_json(self) -> dict:
        return {"id": self.id, "kind": self.kind, "colour": list(self.colour),
                "pts": np.round(self.pts, 1).tolist()}
