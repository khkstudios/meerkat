"""Face alignment + embedding backends.

    sface    : OpenCV FaceRecognizerSF, 128-d, Apache 2.0. Default.
    auraface : AuraFace-v1 (ArcFace ResNet100), 512-d, Apache 2.0. Needs onnxruntime.

All backends return L2-normalised float32 vectors, so cosine similarity is a dot product.
"""
from __future__ import annotations

import cv2
import numpy as np

from .config import AURAFACE_MODEL, MODELS_DIR, SFACE_MODEL
from .detect import Detection

# Standard ArcFace 112x112 5-point template (image-left eye first, matching YuNet order).
ARCFACE_TEMPLATE = np.array([
    [38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
    [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32)


def l2norm(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    return v / max(float(np.linalg.norm(v)), 1e-12)


class SFaceBackend:
    name = "sface"
    dim = 128

    def __init__(self):
        path = MODELS_DIR / SFACE_MODEL
        if not path.exists():
            raise FileNotFoundError(f"{path} missing. Run: python scripts/fetch_models.py")
        self.rec = cv2.FaceRecognizerSF.create(str(path), "")

    def align(self, frame: np.ndarray, det: Detection) -> np.ndarray:
        return self.rec.alignCrop(frame, det.raw)

    def embed_aligned(self, crop: np.ndarray) -> np.ndarray:
        return l2norm(self.rec.feature(crop))


class AuraFaceBackend:
    name = "auraface"
    dim = 512

    def __init__(self):
        try:
            import onnxruntime as ort
        except ImportError as e:
            raise ImportError("auraface backend needs: pip install onnxruntime") from e
        path = MODELS_DIR / AURAFACE_MODEL
        if not path.exists():
            raise FileNotFoundError(f"{path} missing. Run: python scripts/fetch_models.py --auraface")
        providers = [p for p in ("CoreMLExecutionProvider", "CPUExecutionProvider")
                     if p in ort.get_available_providers()]
        self.sess = ort.InferenceSession(str(path), providers=providers)
        self.input = self.sess.get_inputs()[0].name

    def align(self, frame: np.ndarray, det: Detection) -> np.ndarray:
        M, _ = cv2.estimateAffinePartial2D(det.landmarks, ARCFACE_TEMPLATE, method=cv2.LMEDS)
        if M is None:  # degenerate landmarks: fall back to a plain resize of the box
            x, y, w, h = det.box.astype(int)
            return cv2.resize(frame[max(y, 0):y + h, max(x, 0):x + w], (112, 112))
        return cv2.warpAffine(frame, M, (112, 112), borderValue=0)

    def embed_aligned(self, crop: np.ndarray) -> np.ndarray:
        blob = cv2.dnn.blobFromImage(crop, 1 / 127.5, (112, 112), (127.5, 127.5, 127.5), swapRB=True)
        return l2norm(self.sess.run(None, {self.input: blob})[0])


BACKENDS = {"sface": SFaceBackend, "auraface": AuraFaceBackend}


class Embedder:
    def __init__(self, backend: str = "sface"):
        if backend not in BACKENDS:
            raise ValueError(f"unknown backend {backend!r}; choose from {list(BACKENDS)}")
        self.backend = BACKENDS[backend]()
        self.name = self.backend.name

    def __call__(self, frame: np.ndarray, det: Detection) -> tuple[np.ndarray, np.ndarray]:
        """Returns (embedding, aligned_crop)."""
        crop = self.backend.align(frame, det)
        return self.backend.embed_aligned(crop), crop
