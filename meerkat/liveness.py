"""Face anti-spoofing (liveness): is this a real face, or a photo / screen held up to the camera?

Model: MiniFASNetV2 from facenox/face-antispoof-onnx (Apache-2.0 code; derived from Minivision's
Silent-Face-Anti-Spoofing). 128x128 RGB input, outputs logits [real, spoof]. Its weights were
trained on CelebA-Spoof, whose licence is non-commercial research only: check that before any
commercial use.

What it catches: printed photos and phone / monitor replays. What it doesn't: 3D masks. It works
on texture, so it needs faces at least ~64 px wide (roughly 2.5 m at 720p), even light and a
near-frontal pose; smaller faces are simply not judged. One frame is never trusted: a track must
average "real" over several judged frames (see Track.update_liveness).
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .config import LIVENESS_MODEL, MODELS_DIR
from .detect import Detection

INPUT = 128
EXPAND = 1.5          # the model was trained on 1.5x the face box: it needs context around the face
MIN_FACE_PX = 64      # below this the texture cues (screen pixels, print dots) are gone
MAX_YAW = 0.35        # rough yaw from landmarks; profile views drop accuracy


@dataclass
class LiveResult:
    logit_diff: float  # real - spoof; > 0 leans real

    @property
    def is_real(self) -> bool:
        return self.logit_diff >= 0.0


def square_crop(img: np.ndarray, box: np.ndarray, expand: float = EXPAND) -> np.ndarray:
    """Square crop of `expand` x the larger box side, centred on the box, reflect-padded at edges
    (the reference implementation's preprocessing)."""
    x, y, w, h = [float(v) for v in box]
    side = int(max(w, h) * expand)
    x0 = int(x + w / 2 - side / 2)
    y0 = int(y + h / 2 - side / 2)
    H, W = img.shape[:2]
    cx0, cy0, cx1, cy1 = max(0, x0), max(0, y0), min(W, x0 + side), min(H, y0 + side)
    if cx1 <= cx0 or cy1 <= cy0:
        return np.zeros((side, side, 3), img.dtype)
    crop = img[cy0:cy1, cx0:cx1]
    return cv2.copyMakeBorder(crop, cy0 - y0, (y0 + side) - cy1, cx0 - x0, (x0 + side) - cx1,
                              cv2.BORDER_REFLECT_101)


def preprocess(crop_bgr: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    interp = cv2.INTER_LANCZOS4 if rgb.shape[0] < INPUT else cv2.INTER_AREA
    rgb = cv2.resize(rgb, (INPUT, INPUT), interpolation=interp)
    return rgb.transpose(2, 0, 1)[None].astype(np.float32) / 255.0


class Liveness:
    def __init__(self, model_path=None):
        path = model_path or (MODELS_DIR / LIVENESS_MODEL)
        if not path.exists():
            raise FileNotFoundError(f"{path} missing. Run: python scripts/fetch_models.py")
        self.net = cv2.dnn.readNetFromONNX(str(path))

    @staticmethod
    def judgeable(det: Detection) -> bool:
        return det.width_px >= MIN_FACE_PX and abs(det.yaw_hint()) <= MAX_YAW

    def __call__(self, frame: np.ndarray, det: Detection) -> LiveResult | None:
        """None when the face is too small or turned to judge."""
        if not self.judgeable(det):
            return None
        self.net.setInput(preprocess(square_crop(frame, det.box)))
        real, spoof = self.net.forward().reshape(-1)[:2]
        return LiveResult(float(real - spoof))
