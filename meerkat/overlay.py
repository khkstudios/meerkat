"""Confidence-graded rectangles and HUD."""
from __future__ import annotations

import cv2
import numpy as np

from .config import Thresholds
from .tracker import Track

FONT = cv2.FONT_HERSHEY_SIMPLEX


def confidence(sim: float | None, th: Thresholds) -> float:
    """Clipped linear ramp from calibrated 'not me' to 'me' similarity. Never raw cosine."""
    if sim is None:
        return 0.0
    return float(np.clip((sim - th.ramp_lo) / (th.ramp_hi - th.ramp_lo), 0.0, 1.0))


def ramp_colour(c: float) -> tuple[int, int, int]:
    """Red -> amber -> green (BGR)."""
    red, amber, green = np.array([40, 40, 230]), np.array([0, 180, 255]), np.array([70, 200, 60])
    col = red + (amber - red) * (c / 0.5) if c < 0.5 else amber + (green - amber) * ((c - 0.5) / 0.5)
    return tuple(int(v) for v in col)


def _text(img, s, org, scale=0.55, colour=(255, 255, 255), thick=1, bg=(0, 0, 0)):
    (tw, th), base = cv2.getTextSize(s, FONT, scale, thick)
    x, y = org
    cv2.rectangle(img, (x - 3, y - th - 5), (x + tw + 3, y + base + 1), bg, -1)
    cv2.putText(img, s, (x, y), FONT, scale, colour, thick, cv2.LINE_AA)


def draw_track(img: np.ndarray, t: Track, name: str, th: Thresholds) -> None:
    conf = confidence(t.sim_ema, th)
    col = ramp_colour(conf if not t.locked else max(conf, 0.5))
    x, y, w, h = t.box.astype(int)
    thick = 1 + int(round(4 * conf))

    # translucent fill scaled with confidence
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, img.shape[1]), min(y + h, img.shape[0])
    if x1 > x0 and y1 > y0:
        roi = img[y0:y1, x0:x1]
        tint = np.full_like(roi, col)
        cv2.addWeighted(tint, 0.08 + 0.17 * conf, roi, 1 - (0.08 + 0.17 * conf), 0, roi)
    cv2.rectangle(img, (x, y), (x + w, y + h), col, thick, cv2.LINE_AA)
    if t.locked:  # corner brackets mark a confirmed identity
        L = max(10, w // 5)
        for cx, cy, dx, dy in ((x, y, 1, 1), (x + w, y, -1, 1), (x, y + h, 1, -1), (x + w, y + h, -1, -1)):
            cv2.line(img, (cx, cy), (cx + dx * L, cy), col, thick + 2, cv2.LINE_AA)
            cv2.line(img, (cx, cy), (cx, cy + dy * L), col, thick + 2, cv2.LINE_AA)

    who = name if t.locked or t.recognised_now else "Unknown"
    dist = f"{t.distance_m:.1f} m" if t.distance_m is not None else "? m"
    parts = [f"{who} {conf * 100:.0f}%", dist]
    if t.locked:
        parts.append("LOCKED" if t.state == "locked" else "TRACKED")
    _text(img, " | ".join(parts), (x, max(y - 8, 18)), colour=(255, 255, 255), bg=col)
    if t.sim is not None:
        _text(img, f"sim {t.sim:.3f}  fused {t.fused_sim or 0:.3f}  {t.face_px:.0f}px  #{t.id}",
              (x, y + h + 18), scale=0.45, colour=(220, 220, 220))


def draw_hud(img: np.ndarray, lines: list[str]) -> None:
    y = 24
    for s in lines:
        _text(img, s, (10, y), scale=0.55, colour=(255, 255, 255), bg=(30, 30, 30))
        y += 24


def draw_banner(img: np.ndarray, lines: list[str]) -> None:
    h = img.shape[0]
    y = h - 20 - 30 * (len(lines) - 1)
    for s in lines:
        _text(img, s, (10, y), scale=0.8, thick=2, colour=(255, 255, 255), bg=(120, 60, 0))
        y += 30


def draw_big(img: np.ndarray, lines: list[str], colour=(255, 255, 255)) -> None:
    """Large centred text on a dark panel, readable from across the room."""
    h, w = img.shape[:2]
    scales = [h / 100] + [h / 260] * (len(lines) - 1)
    thick = max(2, h // 180)
    sizes = [cv2.getTextSize(s, FONT, sc, thick)[0] for s, sc in zip(lines, scales)]
    gap = h // 30
    total = sum(sz[1] for sz in sizes) + gap * (len(lines) - 1)
    pad = h // 30
    pw = max(sz[0] for sz in sizes) + 2 * pad
    x0, y0 = (w - pw) // 2, h // 14  # top of frame, clear of your face
    panel = img[max(y0, 0):y0 + total + 2 * pad, max(x0, 0):x0 + pw]
    panel[:] = (panel * 0.25).astype(np.uint8)
    y = y0 + pad
    for s, sc, (tw, th) in zip(lines, scales, sizes):
        y += th
        cv2.putText(img, s, ((w - tw) // 2, y), FONT, sc, colour, thick, cv2.LINE_AA)
        y += gap
