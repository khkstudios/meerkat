"""Pinhole distance estimation: distance_m = k / width_px, where k = real_width_m * focal_px.

Calibrating at a measured distance solves k directly from your own face, absorbing both
your real face width and the lens FOV.
"""
from __future__ import annotations

import math

import numpy as np

from .config import DistanceCalib


def focal_from_hfov(image_width_px: int, hfov_deg: float) -> float:
    return (image_width_px / 2) / math.tan(math.radians(hfov_deg) / 2)


class DistanceEstimator:
    def __init__(self, calib: DistanceCalib, frame_width: int):
        self.calib = calib
        f = focal_from_hfov(frame_width, calib.hfov_deg)
        scale = 1.0
        if calib.calibrated_at_width and calib.calibrated_at_width != frame_width:
            # k scales linearly with resolution for the same lens.
            scale = frame_width / calib.calibrated_at_width
        self.k_box = (calib.k_box * scale) if calib.k_box else calib.face_width_m * f
        self.k_eye = (calib.k_eye * scale) if calib.k_eye else calib.eye_dist_m * f
        self.calibrated = calib.k_box is not None

    def estimate(self, box_px: float, eye_px: float, yaw: float = 0.0) -> float:
        d_box = self.k_box / max(box_px, 1.0)
        d_eye = self.k_eye / max(eye_px, 1.0)
        # Eye distance foreshortens as the head turns; lean on the box when yawed.
        w_eye = 0.5 * max(0.0, 1.0 - 2.5 * abs(yaw))
        return float(w_eye * d_eye + (1 - w_eye) * d_box)


def solve_k(widths_px: list[float], eyes_px: list[float], distance_m: float) -> tuple[float, float]:
    return float(np.median(widths_px) * distance_m), float(np.median(eyes_px) * distance_m)
