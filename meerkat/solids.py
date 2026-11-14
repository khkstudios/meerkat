"""3D solids built from recognised 2D strokes, with a tiny flat-shaded renderer (pure OpenCV).

    circle -> sphere      ellipse -> ellipsoid     square -> cube
    rect   -> box         triangle -> pyramid      line   -> cylinder

Coordinates: x right, y down (screen), z away from the viewer. Projection is orthographic, so
every solid here being convex means back-face culling alone renders it correctly.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from .shapes import _ids  # shared id counter with 2D items

LIGHT = np.array([-0.45, -0.65, -1.0])
LIGHT /= np.linalg.norm(LIGHT)
SOLID_FOR = {"circle": "sphere", "ellipse": "ellipsoid", "rect": "box", "triangle": "pyramid",
             "line": "cylinder"}


# ---------------------------------------------------------------- rotations
def rx(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def ry(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rz(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def orthonormalise(R: np.ndarray) -> np.ndarray:
    """Remove drift from many incremental rotations."""
    u, _, vt = np.linalg.svd(R)
    return u @ vt


VIEW_TILT = rx(math.radians(24)) @ ry(math.radians(34))  # shows the top and a side by default


# ---------------------------------------------------------------- meshes (unit-ish, centred)
def box_mesh(w: float, h: float, d: float):
    x, y, z = w / 2, h / 2, d / 2
    v = np.array([[-x, -y, -z], [x, -y, -z], [x, y, -z], [-x, y, -z],
                  [-x, -y, z], [x, -y, z], [x, y, z], [-x, y, z]], float)
    f = [[0, 1, 2, 3], [5, 4, 7, 6], [4, 0, 3, 7], [1, 5, 6, 2], [4, 5, 1, 0], [3, 2, 6, 7]]
    return v, f, True


def pyramid_mesh(w: float, h: float):
    b, t = h / 2, -h / 2  # square base at the bottom (y down), apex at the top
    v = np.array([[-w / 2, b, -w / 2], [w / 2, b, -w / 2], [w / 2, b, w / 2], [-w / 2, b, w / 2],
                  [0, t, 0]], float)
    f = [[0, 1, 2, 3], [0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]]
    return v, f, True


def ellipsoid_mesh(a: float, b: float, c: float, lat: int = 20, lon: int = 36):
    v = [[0, -b, 0]]
    for i in range(1, lat):
        th = math.pi * i / lat
        for j in range(lon):
            ph = 2 * math.pi * j / lon
            v.append([a * math.sin(th) * math.cos(ph), -b * math.cos(th), c * math.sin(th) * math.sin(ph)])
    v.append([0, b, 0])
    v = np.array(v, float)
    f, bottom = [], len(v) - 1
    ring = lambda i, j: 1 + (i - 1) * lon + (j % lon)  # noqa: E731
    for j in range(lon):
        f.append([0, ring(1, j + 1), ring(1, j)])
        f.append([bottom, ring(lat - 1, j), ring(lat - 1, j + 1)])
    for i in range(1, lat - 1):
        for j in range(lon):
            f.append([ring(i, j), ring(i, j + 1), ring(i + 1, j + 1), ring(i + 1, j)])
    return v, f, False


def cylinder_mesh(length: float, r: float, n: int = 24):
    """Axis along x (so a drawn line's direction maps straight onto it)."""
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ring = np.stack([np.cos(t) * r, np.sin(t) * r], 1)
    v = np.vstack([np.column_stack([np.full(n, -length / 2), ring]),
                   np.column_stack([np.full(n, length / 2), ring])])
    f = [list(range(n))[::-1], list(range(n, 2 * n))]
    f += [[j, (j + 1) % n, n + (j + 1) % n, n + j] for j in range(n)]
    return v, f, False


# ---------------------------------------------------------------- solid item
@dataclass(eq=False)
class Solid:
    kind: str
    verts: np.ndarray            # model space, in px
    faces: list
    colour: tuple[int, int, int]
    pos: np.ndarray              # screen position of the centre
    R: np.ndarray = field(default_factory=lambda: VIEW_TILT.copy())
    scale: float = 1.0
    edges: bool = True           # outline faces (flat solids) vs smooth look (curved solids)
    spin: np.ndarray = field(default_factory=lambda: np.zeros(2))  # (about screen y, about screen x) rad/frame
    id: int = field(default_factory=lambda: next(_ids))
    closed = True

    # ---- construction from a recognised 2D shape
    @classmethod
    def from_shape(cls, kind: str, pts: np.ndarray, colour) -> "Solid | None":
        pts = np.asarray(pts, float)
        centre = pts.mean(0)
        if kind == "rect":
            e1, e2 = pts[1] - pts[0], pts[3] - pts[0]
            w, h = np.linalg.norm(e1), np.linalg.norm(e2)
            ang = math.atan2(e1[1], e1[0])
            if min(w, h) / max(w, h) > 0.85:  # square -> cube
                s = (w + h) / 2
                v, f, ed = box_mesh(s, s, s)
                name = "cube"
            else:
                v, f, ed = box_mesh(w, h, min(w, h))
                name = "box"
            return cls(name, v, f, colour, centre, rz(ang) @ VIEW_TILT, edges=ed, spin=np.array([0.012, 0]))
        if kind == "triangle":
            # apex = the vertex furthest from the centroid's opposite side; pyramid points that way
            d = [np.linalg.norm(p - centre) for p in pts]
            apex = pts[int(np.argmax(d))]
            base = np.delete(pts, int(np.argmax(d)), axis=0)
            w = float(np.linalg.norm(base[0] - base[1]))
            mid = base.mean(0)
            h = float(np.linalg.norm(apex - mid))
            up = apex - mid
            ang = math.atan2(up[0], -up[1])  # 0 when the apex points straight up
            v, f, ed = pyramid_mesh(w, h)
            R = rz(ang) @ rx(math.radians(18)) @ ry(math.radians(30))
            return cls("pyramid", v, f, colour, (apex + mid) / 2, R, edges=ed, spin=np.array([0.012, 0]))
        if kind in ("circle", "ellipse"):
            if len(pts) >= 5:
                (cx, cy), (ea, eb), eang = cv2.fitEllipse(pts.astype(np.float32))
            else:
                return None
            if kind == "circle":
                r = (ea + eb) / 4
                v, f, ed = ellipsoid_mesh(r, r, r)
                return cls("sphere", v, f, colour, np.array([cx, cy]), VIEW_TILT.copy(), edges=ed,
                           spin=np.array([0.015, 0]))
            a, b = ea / 2, eb / 2
            v, f, ed = ellipsoid_mesh(a, b, min(a, b))
            return cls("ellipsoid", v, f, colour, np.array([cx, cy]), rz(math.radians(eang)) @ rx(0.35),
                       edges=ed, spin=np.array([0.012, 0]))
        if kind == "line":
            d = pts[1] - pts[0]
            L = float(np.linalg.norm(d))
            tilt = 0.55  # turn the axis into the screen a little so an end cap shows
            v, f, ed = cylinder_mesh(L / math.cos(tilt), max(10.0, 0.07 * L))  # keep drawn length
            return cls("cylinder", v, f, colour, centre, rz(math.atan2(d[1], d[0])) @ ry(tilt) @ rx(0.4),
                       edges=ed, spin=np.array([0.0, 0.0]))
        return None

    # ---- geometry
    def world(self) -> np.ndarray:
        return (self.verts * self.scale) @ self.R.T

    def projected(self) -> np.ndarray:
        return self.world()[:, :2] + self.pos

    @property
    def centre(self) -> np.ndarray:
        return self.pos

    @property
    def pts(self) -> np.ndarray:  # silhouette points (used for sizing, saving, tests)
        p = self.projected().astype(np.float32)
        return cv2.convexHull(p).reshape(-1, 2).astype(np.float64)

    def distance(self, p: np.ndarray) -> float:
        hull = self.pts.astype(np.float32).reshape(-1, 1, 2)
        return max(0.0, -cv2.pointPolygonTest(hull, (float(p[0]), float(p[1])), True))

    # ---- manipulation (same interface as 2D Item)
    def translate(self, d: np.ndarray) -> None:
        self.pos = self.pos + d

    def state(self):
        return (self.pos.copy(), self.R.copy(), self.scale)

    def apply_transform(self, base, pivot: np.ndarray, scale: float, angle: float, shift: np.ndarray) -> None:
        pos0, R0, s0 = base
        c, s = math.cos(angle), math.sin(angle)
        R2 = np.array([[c, -s], [s, c]]) * scale
        self.pos = (pos0 - pivot) @ R2.T + pivot + shift
        self.R = rz(angle) @ R0
        self.scale = float(np.clip(s0 * scale, 0.1, 10))

    def scale_by(self, f: float) -> None:
        self.scale *= f

    def rotate_inplane(self, angle: float) -> None:
        self.R = orthonormalise(rz(angle) @ self.R)

    def rotate_by_drag(self, dx: float, dy: float, k: float = 0.012) -> None:
        """Trackball: drag right turns the front face right, drag down tips it down."""
        self.R = orthonormalise(rx(dy * k) @ ry(-dx * k) @ self.R)

    def step_spin(self, decay: float = 0.97) -> None:
        if np.any(np.abs(self.spin) > 1e-4):
            self.R = orthonormalise(rx(self.spin[1]) @ ry(self.spin[0]) @ self.R)
            self.spin = self.spin * decay

    # ---- render
    def _face_groups(self) -> list[np.ndarray]:
        """Faces grouped by vertex count, as int arrays, so normals can be computed in bulk."""
        if getattr(self, "_groups", None) is None:
            by_n: dict[int, list] = {}
            for f in self.faces:
                by_n.setdefault(len(f), []).append(f)
            self._groups = [np.array(v, np.int32) for v in by_n.values()]
        return self._groups

    def draw(self, img: np.ndarray, highlight: bool = False) -> None:
        W = self.world()
        P = W[:, :2] + self.pos
        centroid = W.mean(0)
        polys, shades, depths = [], [], []
        for F in self._face_groups():
            Q = W[F]                                              # (k, n, 3)
            N = np.cross(Q[:, 1] - Q[:, 0], Q[:, 2] - Q[:, 0])
            if F.shape[1] > 3:  # a collapsed first corner (e.g. at a pole) -> use the next three
                bad = np.linalg.norm(N, axis=1) < 1e-9
                N[bad] = np.cross(Q[bad, 2] - Q[bad, 0], Q[bad, 3] - Q[bad, 0])
            norm = np.linalg.norm(N, axis=1)
            ok = norm > 1e-9
            N[ok] /= norm[ok, None]
            mid = Q.mean(1)
            flip = np.einsum("ij,ij->i", N, mid - centroid) < 0   # make normals point outwards
            N[flip] *= -1
            vis = ok & (N[:, 2] < 0)                              # facing the viewer
            light = 0.28 + 0.72 * np.clip(N[vis] @ LIGHT, 0, None)  # LIGHT points towards the lamp
            polys += [np.round(P[f] * 16).astype(np.int32) for f in F[vis]]  # 4 fractional bits
            shades += list(light)
            depths += list(mid[vis, 2])
        if highlight:
            hull = cv2.convexHull(P.astype(np.float32)).astype(np.int32)
            cv2.polylines(img, [hull], True, (255, 255, 255), 8, cv2.LINE_AA)
        col = np.array(self.colour, float)
        for i in np.argsort(depths)[::-1]:  # far first (painter's), belt-and-braces with culling
            shade = tuple(int(v) for v in np.clip(col * shades[i], 0, 255))
            cv2.fillConvexPoly(img, polys[i], shade, cv2.LINE_AA, shift=4)
            if self.edges:
                edge = tuple(int(v) for v in np.clip(col * shades[i] * 0.55, 0, 255))
                cv2.polylines(img, [polys[i]], True, edge, 2, cv2.LINE_AA, shift=4)

    def to_json(self) -> dict:
        return {"id": self.id, "kind": self.kind, "colour": list(self.colour), "solid": True,
                "pos": np.round(self.pos, 1).tolist(), "scale": round(self.scale, 3),
                "R": np.round(self.R, 4).tolist()}
