"""Enrolled-embedding gallery: storage, matching and quality gates.

Only embeddings are stored (no images). The file lives in data/ and is gitignored.
"""
from __future__ import annotations

import io
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from . import secure
from .config import DISTANCE_BANDS, gallery_path, legacy_gallery_path
from .detect import Detection


def nearest_band(distance_m: float | None) -> float:
    if distance_m is None:
        return -1.0
    return min(DISTANCE_BANDS, key=lambda b: abs(b - distance_m))


def pose_label(yaw: float) -> str:
    if yaw < -0.12:
        return "left"
    if yaw > 0.12:
        return "right"
    return "frontal"


@dataclass
class QualityResult:
    ok: bool
    reason: str = ""
    blur: float = 0.0


def quality_gate(frame: np.ndarray, det: Detection, crop: np.ndarray, min_score: float = 0.7,
                 min_blur: float = 40.0, edge_margin: int = 2) -> QualityResult:
    h, w = frame.shape[:2]
    x, y, bw, bh = det.box
    if det.score < min_score:
        return QualityResult(False, f"score {det.score:.2f}")
    if x < edge_margin or y < edge_margin or x + bw > w - edge_margin or y + bh > h - edge_margin:
        return QualityResult(False, "cut off at edge")
    blur = float(cv2.Laplacian(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
    if blur < min_blur:
        return QualityResult(False, f"blurry ({blur:.0f})", blur)
    return QualityResult(True, "", blur)


class Gallery:
    def __init__(self, name: str, backend: str, dim: int):
        self.name = name
        self.backend = backend
        self.dim = dim
        self.emb = np.zeros((0, dim), np.float32)
        self.band = np.zeros(0, np.float32)
        self.face_px = np.zeros(0, np.float32)
        self.pose: list[str] = []
        self.created = time.time()
        self.dirty = False

    def __len__(self) -> int:
        return len(self.emb)

    def add(self, e: np.ndarray, band: float, face_px: float, pose: str) -> None:
        self.emb = np.vstack([self.emb, e.reshape(1, -1).astype(np.float32)])
        self.band = np.append(self.band, np.float32(band))
        self.face_px = np.append(self.face_px, np.float32(face_px))
        self.pose.append(pose)
        self.dirty = True

    def band_counts(self) -> dict[float, int]:
        return {b: int(np.sum(self.band == b)) for b in DISTANCE_BANDS}

    def match(self, q: np.ndarray, mode: str = "max", k: int = 3) -> float:
        """Cosine similarity of q against the gallery: max, or mean of the top-k."""
        if len(self.emb) == 0:
            return 0.0
        sims = self.emb @ q.astype(np.float32)
        if mode == "topk" and len(sims) > 1:
            k = min(k, len(sims))
            return float(np.mean(np.partition(sims, -k)[-k:]))
        return float(sims.max())

    def save(self, path: Path) -> None:
        """Paths ending in .enc are encrypted (the default gallery_path); others are plaintext."""
        buf = io.BytesIO()
        np.savez(buf, emb=self.emb, band=self.band, face_px=self.face_px,
                 pose=np.array(self.pose, dtype="U8"), name=self.name, backend=self.backend,
                 created=self.created)
        data = buf.getvalue()
        secure.write_private(path, secure.encrypt(data) if path.suffix == ".enc" else data)
        self.dirty = False

    @classmethod
    def load(cls, path: Path) -> "Gallery":
        raw = path.read_bytes()
        if path.suffix == ".enc":
            raw = secure.decrypt(raw)
        z = np.load(io.BytesIO(raw), allow_pickle=False)
        g = cls(str(z["name"]), str(z["backend"]), int(z["emb"].shape[1]))
        g.emb, g.band, g.face_px = z["emb"], z["band"], z["face_px"]
        g.pose = [str(p) for p in z["pose"]]
        g.created = float(z["created"])
        return g

    @staticmethod
    def exists(backend: str) -> bool:
        return gallery_path(backend).exists() or legacy_gallery_path(backend).exists()

    @classmethod
    def open(cls, backend: str) -> "Gallery":
        """Load the encrypted gallery; a plaintext one from before encryption is migrated (encrypted,
        then the plaintext file is overwritten and deleted)."""
        enc, legacy = gallery_path(backend), legacy_gallery_path(backend)
        if enc.exists():
            return cls.load(enc)
        if legacy.exists():
            g = cls.load(legacy)
            g.save(enc)
            if cls.load(enc).emb.shape != g.emb.shape:  # verify before destroying the original
                raise RuntimeError("encrypted copy didn't verify; plaintext gallery left in place")
            secure.secure_delete(legacy)
            print(f"[gallery] encrypted {legacy.name} -> {enc.name} and securely deleted the "
                  "plaintext copy", file=sys.stderr)
            return g
        raise FileNotFoundError(f"No gallery at {enc}. Run: python -m meerkat enrol --backend {backend}")

    def summary(self) -> str:
        counts = ", ".join(f"{b:g}m:{n}" for b, n in self.band_counts().items())
        poses = {p: self.pose.count(p) for p in ("frontal", "left", "right")}
        return (f"{self.name} [{self.backend}] {len(self)} embeddings | bands {counts} | "
                f"poses {poses}")
