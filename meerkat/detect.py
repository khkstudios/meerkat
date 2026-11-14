"""YuNet face detector wrapper."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .config import yunet_path


@dataclass
class Detection:
    box: np.ndarray        # x, y, w, h (float)
    landmarks: np.ndarray  # 5x2: right eye, left eye, nose, right mouth, left mouth
    score: float
    raw: np.ndarray        # original 15-float YuNet row (FaceRecognizerSF.alignCrop wants it)

    @property
    def width_px(self) -> float:
        return float(self.box[2])

    @property
    def eye_px(self) -> float:
        return float(np.linalg.norm(self.landmarks[0] - self.landmarks[1]))

    def yaw_hint(self) -> float:
        """Rough yaw in [-1, 1]: nose offset from the eye midpoint, in eye-distances."""
        mid = (self.landmarks[0] + self.landmarks[1]) / 2
        d = max(self.eye_px, 1e-6)
        return float(np.clip((self.landmarks[2, 0] - mid[0]) / d, -1, 1))


class Detector:
    def __init__(self, model: Path | None = None, score_threshold: float = 0.6,
                 nms_threshold: float = 0.3, top_k: int = 50):
        self.model = Path(model) if model else yunet_path()
        self.det = cv2.FaceDetectorYN.create(str(self.model), "", (320, 320),
                                             score_threshold, nms_threshold, top_k)
        self._size = (320, 320)

    def __call__(self, frame: np.ndarray) -> list[Detection]:
        h, w = frame.shape[:2]
        if self._size != (w, h):
            self.det.setInputSize((w, h))
            self._size = (w, h)
        _, faces = self.det.detect(frame)
        if faces is None:
            return []
        out = []
        for row in faces:
            out.append(Detection(box=row[0:4].astype(np.float32).copy(),
                                 landmarks=row[4:14].reshape(5, 2).astype(np.float32).copy(),
                                 score=float(row[14]), raw=row.copy()))
        return out
