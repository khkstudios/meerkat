"""Paths, per-backend thresholds and tunables.

Every threshold here is a *starting point*. Run the distance sweep and
`eval.report --calibrate` to replace them with numbers measured on your face,
your camera and your lighting (written to data/calibration.json).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"
CALIBRATION_FILE = DATA_DIR / "calibration.json"

YUNET_CANDIDATES = ["face_detection_yunet_2026may.onnx", "face_detection_yunet_2023mar.onnx"]
SFACE_MODEL = "face_recognition_sface_2021dec.onnx"
AURAFACE_MODEL = "auraface_glintr100.onnx"

DISTANCE_BANDS = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0]


@dataclass
class Thresholds:
    match: float      # per-frame "this is the person" cosine threshold
    lock_high: float  # fused similarity needed to lock (T_high)
    lock_low: float   # fused similarity below which a lock may release (T_low)
    ramp_lo: float    # cosine mapped to 0% display confidence
    ramp_hi: float    # cosine mapped to 100% display confidence


# SFace: OpenCV's published LFW threshold is 0.363.
# AuraFace / ArcFace-style: different scale, recalibrate.
DEFAULT_THRESHOLDS = {
    "sface": Thresholds(match=0.363, lock_high=0.45, lock_low=0.28, ramp_lo=0.16, ramp_hi=0.56),
    "auraface": Thresholds(match=0.30, lock_high=0.40, lock_low=0.20, ramp_lo=0.08, ramp_hi=0.52),
}


@dataclass
class TrackerConfig:
    iou_match: float = 0.25        # min IoU to associate a detection with a track
    max_missed: int = 15           # frames a track survives without a detection
    embed_window: int = 10         # N frames of embeddings averaged for fusion
    ema_alpha: float = 0.3         # display EMA of per-frame similarity
    lock_frames: int = 5           # K frames above T_high to lock
    release_frames: int = 15       # M judgeable frames below T_low to release
    min_judge_px: int = 60         # face width below which we do not trust a "no"
    lost_recog_frames: int = 8     # consecutive sub-threshold frames -> lost_recognition event


@dataclass
class DistanceCalib:
    # distance_m = k / width_px. Uncalibrated defaults derive from HFOV + mean face size.
    k_box: float | None = None
    k_eye: float | None = None
    hfov_deg: float = 65.0
    face_width_m: float = 0.143
    eye_dist_m: float = 0.063
    calibrated_at_width: int | None = None  # frame width the k values were measured at


@dataclass
class Calibration:
    distance: DistanceCalib = field(default_factory=DistanceCalib)
    thresholds: dict[str, Thresholds] = field(default_factory=lambda: dict(DEFAULT_THRESHOLDS))

    @classmethod
    def load(cls, path: Path = CALIBRATION_FILE) -> "Calibration":
        c = cls()
        if path.exists():
            raw = json.loads(path.read_text())
            if "distance" in raw:
                c.distance = DistanceCalib(**raw["distance"])
            for name, t in raw.get("thresholds", {}).items():
                c.thresholds[name] = Thresholds(**t)
        return c

    def save(self, path: Path = CALIBRATION_FILE) -> None:
        path.parent.mkdir(exist_ok=True)
        out = {
            "distance": asdict(self.distance),
            "thresholds": {k: asdict(v) for k, v in self.thresholds.items()},
        }
        path.write_text(json.dumps(out, indent=2))


def yunet_path() -> Path:
    for name in YUNET_CANDIDATES:
        p = MODELS_DIR / name
        if p.exists():
            return p
    raise FileNotFoundError("No YuNet model in models/. Run: python scripts/fetch_models.py")


def gallery_path(backend: str) -> Path:
    return DATA_DIR / f"gallery_{backend}.npz"
