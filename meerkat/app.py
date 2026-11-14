"""meerkat CLI.

    python -m meerkat calibrate --distance 1.0      # stand at a measured distance
    python -m meerkat enrol --name Keegan           # guided capture across distance bands
    python -m meerkat run                           # live recognition + tracking + HUD
    python -m meerkat gestures [--owner-only]       # draw shapes / move items with your hands
    python -m meerkat gallery info | delete
    python -m eval.distance_sweep --label me        # measure recognition vs distance
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time

import cv2
import numpy as np

from .align_embed import Embedder
from .capture import Capture, list_cameras, probe_cameras
from .config import CALIBRATION_FILE, Calibration, gallery_path
from .detect import Detection, Detector
from .distance import DistanceEstimator, solve_k
from .events import EventBus
from .gallery import Gallery, pose_label, quality_gate
from .overlay import draw_banner, draw_big, draw_hud, draw_track
from .pipeline import Pipeline

WIN = "meerkat"


def add_source_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--source", default="builtin",
                   help="'builtin' (laptop camera, default), 'name:<substring>', index, or video file")
    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--height", type=int, default=1080)
    p.add_argument("--no-mirror", action="store_true")
    p.add_argument("--backend", default="sface", choices=["sface", "auraface"])


def open_capture(a) -> Capture:
    cap = Capture(a.source, a.width, a.height, mirror=not a.no_mirror)
    print(f"[capture] {cap.describe()} (requested {a.width}x{a.height})", file=sys.stderr)
    return cap


def largest(dets: list[Detection]) -> Detection | None:
    return max(dets, key=lambda d: d.width_px) if dets else None


def black_notice(cap: Capture) -> list[str]:
    if cap.is_black:
        return ["Camera frames are BLACK: allow your terminal app in System Settings > Privacy &",
                "Security > Camera, uncover the lens, or try --source 1/2 (python -m meerkat cameras)"]
    return []


def chime(sound: str = "Tink") -> None:
    """Non-blocking audible cue, so you know what's happening when you're across the room."""
    if sys.platform == "darwin":
        snd = f"/System/Library/Sounds/{sound}.aiff"
        try:
            subprocess.Popen(["afplay", snd], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return
        except OSError:
            pass
    print("\a", end="", file=sys.stderr, flush=True)


class Countdown:
    """SPACE -> N-second countdown (one tick per second) -> go."""

    def __init__(self, seconds: float):
        self.seconds = seconds
        self.end: float | None = None
        self._last_tick = None

    def start(self) -> None:
        self.end = time.time() + self.seconds
        self._last_tick = None

    def cancel(self) -> None:
        self.end = None

    @property
    def active(self) -> bool:
        return self.end is not None

    def remaining(self) -> int:
        return max(0, int(np.ceil(self.end - time.time()))) if self.end else 0

    def poll(self) -> bool:
        """True exactly once, when the countdown finishes."""
        if self.end is None:
            return False
        r = self.remaining()
        if r != self._last_tick and r > 0:
            chime("Tink")
            self._last_tick = r
        if time.time() >= self.end:
            self.end = None
            chime("Glass")
            return True
        return False


QUIT = -2


def key() -> int:
    return cv2.waitKey(1) & 0xFF


def show(win: str, frame) -> int:
    """imshow + key poll. Returns the key (optional shortcuts), or QUIT when the window's close
    button was clicked, so every screen can be ended without the keyboard."""
    cv2.imshow(win, frame)
    k = cv2.waitKey(1) & 0xFF
    try:
        if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            return QUIT
    except cv2.error:
        return QUIT
    return k


# ---------------------------------------------------------------- calibrate
def cmd_calibrate(a) -> int:
    cap, det = open_capture(a), Detector()
    calib = Calibration.load()
    widths, eyes, collecting = [], [], False
    cd = Countdown(a.countdown)
    cd.start()  # hands-free: walk to the mark while it counts down
    while True:
        frame = cap.read()
        if frame is None:
            break
        d = largest(det(frame))
        if d is not None:
            x, y, w, h = d.box.astype(int)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 200, 255), 2)
            if collecting and d.score > 0.8 and abs(d.yaw_hint()) < 0.12:
                widths.append(d.width_px)
                eyes.append(d.eye_px)
        if cd.poll():
            collecting = True
        if collecting:
            msg = f"Collecting {len(widths)}/{a.samples}... look straight at the camera"
            draw_big(frame, [f"{len(widths)}/{a.samples}", "look at the camera"], (70, 200, 60))
        else:
            msg = f"Stand {a.distance:g} m from the camera (measure to your face). Close the window to cancel."
            draw_big(frame, [str(cd.remaining()), f"stand at {a.distance:g} m"])
        draw_banner(frame, black_notice(cap) + [msg])
        k = show(WIN, frame)
        if k in (QUIT, ord("q")):
            return 1
        if k == ord(" "):  # optional: start now
            cd.cancel()
            collecting = True
        if len(widths) >= a.samples:
            break
    cap.release()
    cv2.destroyAllWindows()
    if len(widths) < a.samples:
        print("Not enough samples.")
        return 1
    k_box, k_eye = solve_k(widths, eyes, a.distance)
    calib.distance.k_box, calib.distance.k_eye = k_box, k_eye
    calib.distance.calibrated_at_width = cap.width
    calib.save()
    print(f"Saved {CALIBRATION_FILE}: box {np.median(widths):.1f}px, eyes {np.median(eyes):.1f}px "
          f"at {a.distance} m -> k_box={k_box:.1f}, k_eye={k_eye:.1f} @ {cap.width}px wide")
    return 0


# ---------------------------------------------------------------- enrol
def cmd_enrol(a) -> int:
    path = gallery_path(a.backend)
    emb = Embedder(a.backend)
    if path.exists() and not (a.append or a.overwrite):
        print(f"{path} exists. Use --append to add to it or --overwrite to replace it.")
        return 1
    g = Gallery.load(path) if (path.exists() and a.append) else Gallery(a.name, a.backend, emb.backend.dim)
    cap, det = open_capture(a), Detector()
    dist = DistanceEstimator(Calibration.load().distance, cap.width)
    if not dist.calibrated:
        print("[enrol] distance is uncalibrated, so bands can't be enforced. "
              "Run `python -m meerkat calibrate` first for a gallery with real far-away samples.",
              file=sys.stderr)
    bands = [float(b) for b in a.bands.split(",")]
    bi, capturing, last_t, last_e, status = 0, False, 0.0, None, ""
    cd, settled, in_band = Countdown(a.countdown), 0, False
    band_started = time.time()
    start_count = {b: int(np.sum(np.isclose(g.band, b))) for b in bands}

    while bi < len(bands):
        band = bands[bi]
        frame = cap.read()
        if frame is None:
            break
        got = int(np.sum(np.isclose(g.band, band))) - start_count[band]
        d = largest(det(frame))
        est = None
        if d is not None:
            est = dist.estimate(d.width_px, d.eye_px, d.yaw_hint())
            e, crop = emb(frame, d)
            far = band >= 3  # far faces score lower and look softer; relax both gates
            q = quality_gate(frame, d, crop, min_score=a.min_score - (0.1 if far else 0.0),
                             min_blur=a.min_blur * (0.5 if far else 1.0))
            colour = (70, 200, 60) if q.ok else (40, 40, 230)
            x, y, w, h = d.box.astype(int)
            # Enforce the band: a "3 m" sample captured from your chair is just another close-up.
            lo, hi = band * (1 - a.band_tolerance), band * (1 + a.band_tolerance)
            in_band = not dist.calibrated or lo <= est <= hi
            if not in_band:
                colour = (0, 180, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), colour, 2)
            if not q.ok:
                status = q.reason
            elif not in_band:
                status = f"{'step back' if est < lo else 'come closer'}: need {lo:.1f}-{hi:.1f} m"
            else:
                status = "ok"
            now = time.time()
            # Require a moment of being in range and steady, so frames from walking aren't kept.
            settled = settled + 1 if (q.ok and in_band) else 0
            if capturing and settled < a.settle_frames and q.ok and in_band:
                status = "hold still..."
            if capturing and settled >= a.settle_frames and now - last_t > a.interval:
                if last_e is None or float(e @ last_e) < 0.97:  # skip near-duplicates
                    g.add(e, band, d.width_px, pose_label(d.yaw_hint()))
                    last_t, last_e = now, e
        if d is None:
            settled = 0
        if not capturing and not cd.active:
            cd.start()  # hands-free: each band starts its own countdown
        if cd.poll():
            capturing, band_started = True, time.time()
        if capturing and got >= a.per_band:
            capturing, bi, settled = False, bi + 1, 0
            chime("Hero")  # band done: the next band's countdown starts now
            continue
        if capturing and time.time() - band_started > a.band_timeout:
            capturing, bi, settled = False, bi + 1, 0
            chime("Basso")  # couldn't fill this band in time: move on rather than wait for a key
            print(f"[enrol] band {band:g} m timed out with {got}/{a.per_band}", file=sys.stderr)
            continue
        est_s = f"~{est:.1f} m" if est is not None else "no face"
        lines = [f"Band {bi + 1}/{len(bands)}: stand ~{band:g} m away ({est_s}, {status})"]
        if capturing:
            lines.append(f"Capturing {got}/{a.per_band}: slowly turn left/right, tilt up/down")
            draw_big(frame, [f"{got}/{a.per_band}", status if status != "ok" else "turn slowly"],
                     (70, 200, 60) if status == "ok" else (0, 180, 255))
        else:
            lines.append("Get into position. Close the window to finish & save.")
            draw_big(frame, [str(cd.remaining()), f"go to ~{band:g} m"])
        draw_banner(frame, black_notice(cap) + lines)
        draw_hud(frame, [g.summary()[:110]])
        k = show(WIN, frame)
        if k in (QUIT, ord("q")):
            break
        if k == ord(" ") and not capturing:  # optional shortcuts: start now / skip band
            cd.cancel()
            capturing, band_started = True, time.time()
        if k == ord("s"):
            capturing, bi, settled = False, bi + 1, 0
            cd.cancel()
    cap.release()
    cv2.destroyAllWindows()
    if len(g) == 0:
        print("Nothing captured; gallery not saved.")
        return 1
    g.save(path)
    print(f"Saved {path}\n{g.summary()}")
    return 0


# ---------------------------------------------------------------- run
def cmd_run(a) -> int:
    path = gallery_path(a.backend)
    if not path.exists():
        print(f"No gallery at {path}. Run: python -m meerkat enrol --name <you> --backend {a.backend}")
        return 1
    g = Gallery.load(path)
    print(f"[gallery] {g.summary()}", file=sys.stderr)
    calib = Calibration.load()
    cap = open_capture(a)
    pipe = Pipeline(g, Embedder(a.backend), calib, cap.width, match_mode=a.match, adaptive=a.adaptive)
    bus = EventBus(g.name, None if a.events == "none" else a.events)
    if not pipe.dist.calibrated:
        print("[distance] uncalibrated (using 65 deg HFOV guess). Run: python -m meerkat calibrate",
              file=sys.stderr)
    fps, last = 0.0, time.time()
    try:
        while True:
            frame = cap.read()
            if frame is None:
                break
            now = time.time()
            tracks = pipe.process(frame, now)
            bus.update(tracks, pipe.tracker.removed, pipe.conf)
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - last, 1e-6))
            last = now
            if a.headless:
                continue
            for t in tracks:
                if t.missed == 0 or t.locked:
                    draw_track(frame, t, g.name, pipe.th)
            primary = max(tracks, key=lambda t: (t.locked, t.face_px), default=None)
            cur = f"{primary.distance_m:.2f} m" if primary and primary.distance_m else "-"
            draw_hud(frame, [
                f"{cap.describe()} | {a.backend} | {fps:.0f} fps"
                + ("" if pipe.dist.calibrated else " | distance UNCALIBRATED"),
                f"distance {cur}",
                f"max recognised {pipe.stats.max_recognised_m:.2f} m | "
                f"max tracked (locked) {pipe.stats.max_tracked_m:.2f} m",
            ])
            if cap.is_black:
                draw_banner(frame, black_notice(cap))
            k = show(WIN, frame)
            if k in (QUIT, ord("q"), 27):
                break
            if k == ord("r"):
                pipe.stats.max_recognised_m = pipe.stats.max_tracked_m = 0.0
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        bus.close()
    if a.adaptive and g.dirty:
        g.save(path)
        print(f"[gallery] adaptive additions saved: {g.summary()}", file=sys.stderr)
    print(f"Session: max recognised {pipe.stats.max_recognised_m:.2f} m, "
          f"max tracked {pipe.stats.max_tracked_m:.2f} m", file=sys.stderr)
    return 0


# ---------------------------------------------------------------- gestures
def cmd_gestures(a) -> int:
    from . import gestures
    return gestures.run(a, open_capture, draw_hud, draw_banner, show, QUIT)


# ---------------------------------------------------------------- cameras
def cmd_cameras(a) -> int:
    names = {c["index"]: c["name"] + (" [builtin]" if c["builtin"] else "") for c in list_cameras()}
    for c in probe_cameras():
        print(f"  --source {c['index']}: {c['size']:>9}  {c['status']:<40} {names.get(c['index'], '')}")
    return 0


# ---------------------------------------------------------------- gallery
def cmd_gallery(a) -> int:
    path = gallery_path(a.backend)
    if not path.exists():
        print(f"No gallery at {path}")
        return 1
    if a.action == "info":
        print(Gallery.load(path).summary())
    elif a.action == "delete":
        path.unlink()
        print(f"Deleted {path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="meerkat", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("calibrate", help="solve distance constants at a measured distance")
    add_source_args(p)
    p.add_argument("--distance", type=float, default=1.0, help="metres from camera to face")
    p.add_argument("--countdown", type=float, default=8.0, help="seconds to get into position")
    p.add_argument("--samples", type=int, default=40)
    p.set_defaults(fn=cmd_calibrate)

    p = sub.add_parser("enrol", help="capture gallery embeddings")
    add_source_args(p)
    p.add_argument("--name", default="me")
    p.add_argument("--bands", default="0.5,1,2,3,4")
    p.add_argument("--per-band", type=int, default=25)
    p.add_argument("--interval", type=float, default=0.15, help="seconds between captures")
    p.add_argument("--countdown", type=float, default=8.0,
                   help="seconds between SPACE and capture, to walk to your spot")
    p.add_argument("--band-timeout", type=float, default=90.0,
                   help="seconds of capturing before an unfinished band is skipped")
    p.add_argument("--settle-frames", type=int, default=10,
                   help="consecutive in-range frames before capture starts")
    p.add_argument("--min-score", type=float, default=0.7, help="YuNet detection score floor")
    p.add_argument("--band-tolerance", type=float, default=0.35,
                   help="accept captures within +/- this fraction of the band distance")
    p.add_argument("--min-blur", type=float, default=40.0, help="Laplacian variance floor")
    p.add_argument("--append", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.set_defaults(fn=cmd_enrol)

    p = sub.add_parser("run", help="live recognition")
    add_source_args(p)
    p.add_argument("--match", default="max", choices=["max", "topk"])
    p.add_argument("--adaptive", action="store_true", help="grow gallery in thin distance bands")
    p.add_argument("--events", default="-", help="'-' stdout, a file path, or 'none'")
    p.add_argument("--headless", action="store_true")
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("gestures", help="hand-tracked canvas: draw shapes, move items")
    add_source_args(p)
    p.add_argument("--owner-only", action="store_true",
                   help="only accept gestures while face recognition has the enrolled person locked")
    p.add_argument("--events", default="none", help="'-' stdout, a file path, or 'none'")
    p.add_argument("--reach", type=float, default=0.1,
                   help="camera margin mapped off-canvas: the middle (1 - 2*reach) of the view "
                        "covers the whole screen, so edges are reachable without leaving the frame")
    p.set_defaults(fn=cmd_gestures)

    p = sub.add_parser("cameras", help="list camera indexes and whether they deliver live frames")
    p.set_defaults(fn=cmd_cameras)

    p = sub.add_parser("gallery", help="inspect or delete enrolled data")
    p.add_argument("action", choices=["info", "delete"])
    p.add_argument("--backend", default="sface", choices=["sface", "auraface"])
    p.set_defaults(fn=cmd_gallery)

    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
