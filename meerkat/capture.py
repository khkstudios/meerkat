"""Webcam / video source with forced resolution."""
from __future__ import annotations

import sys
import threading
import time

import cv2
import numpy as np


def list_cameras() -> list[dict]:
    """Cameras in OpenCV's AVFoundation index order (macOS only; needs pyobjc).

    OpenCV sorts AVFoundation devices by uniqueID, so indexes shift whenever a device such as
    an iPhone (Continuity Camera) connects or disconnects. Never hard-code an index.
    """
    if sys.platform != "darwin":
        return []
    try:
        import AVFoundation as AV
    except ImportError:
        return []
    devs = list(AV.AVCaptureDevice.devicesWithMediaType_(AV.AVMediaTypeVideo))
    devs += list(AV.AVCaptureDevice.devicesWithMediaType_(AV.AVMediaTypeMuxed))
    devs.sort(key=lambda d: str(d.uniqueID()))
    return [{"index": i, "name": str(d.localizedName()), "type": str(d.deviceType()),
             "builtin": str(d.deviceType()) == "AVCaptureDeviceTypeBuiltInWideAngleCamera"}
            for i, d in enumerate(devs)]


def resolve_source(source: str) -> tuple[int | str, str]:
    """Map --source to an OpenCV index. Accepts 'builtin', 'name:<substring>', an index, or a file."""
    if source.isdigit():
        return int(source), f"index {source}"
    if source != "builtin" and not source.startswith("name:"):
        return source, source  # video file
    cams = list_cameras()
    if not cams:
        if sys.platform == "darwin":
            print("[capture] pyobjc-framework-AVFoundation not installed; can't pick the camera by "
                  "name, falling back to index 0", file=sys.stderr)
        return 0, "index 0"
    if source == "builtin":
        match = [c for c in cams if c["builtin"]] or [c for c in cams if "FaceTime" in c["name"]]
    else:
        want = source[5:].lower()
        match = [c for c in cams if want in c["name"].lower()]
    if not match:
        names = ", ".join(f"{c['index']}={c['name']}" for c in cams)
        raise RuntimeError(f"No camera matches {source!r}. Available: {names}")
    return match[0]["index"], match[0]["name"]


class Capture:
    """Wraps cv2.VideoCapture. `source`: 'builtin' (laptop camera), 'name:<substring>',
    a camera index, or a video file path."""

    def __init__(self, source: int | str = "builtin", width: int = 1920, height: int = 1080,
                 fps: int = 30, mirror: bool = True):
        source = str(source)
        target, self.device_name = resolve_source(source)
        self.is_file = isinstance(target, str)
        if self.is_file:
            self.cap = cv2.VideoCapture(target)
        else:
            idx = target
            backend = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY
            self.cap = cv2.VideoCapture(idx, backend)
            # MJPG lets many webcams deliver 1080p at full frame rate.
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.cap.set(cv2.CAP_PROP_FPS, fps)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"Could not open video source {source!r}. On macOS, grant camera access to "
                "your terminal in System Settings > Privacy & Security > Camera.")
        self.mirror = mirror and not self.is_file
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.black_frames = 0
        if not self.is_file:
            self._warm_up()

    def _warm_up(self, timeout: float = 4.0) -> None:
        """Webcams (e.g. FaceTime HD) send black frames for ~1 s after opening."""
        t0 = time.time()
        while time.time() - t0 < timeout:
            ok, frame = self.cap.read()
            if ok and frame.mean() > 3:
                return
        print("[capture] WARNING: camera is still sending black frames after warm-up. Check that the "
              "app running this terminal (e.g. VS Code) is allowed in System Settings > Privacy & "
              "Security > Camera, the lens isn't covered, or try another --source "
              "(python -m meerkat cameras).", file=sys.stderr)

    @property
    def is_black(self) -> bool:
        """True once frames have been black for ~1 s straight."""
        return self.black_frames > 30

    def read(self) -> np.ndarray | None:
        ok, frame = self.cap.read()
        if not ok:
            return None
        self.black_frames = self.black_frames + 1 if frame.mean() < 3 else 0
        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            self.height, self.width = frame.shape[:2]
        return cv2.flip(frame, 1) if self.mirror else frame

    # ---- background capture: the reader never waits for the camera
    def start_thread(self) -> "Capture":
        """Read frames on a background thread; `latest()` / `wait_new()` return the newest one.
        Removes the per-frame wait for the camera from the processing loop."""
        self._cond = threading.Condition()
        self._frame, self._seq, self._t, self._running = None, 0, 0.0, True

        def loop():
            while self._running:
                f = self.read()
                if f is None:
                    if self.is_file:
                        self._running = False
                        with self._cond:
                            self._cond.notify_all()
                        break
                    continue
                with self._cond:
                    self._frame, self._seq, self._t = f, self._seq + 1, time.time()
                    self._cond.notify_all()

        self._thread = threading.Thread(target=loop, name="capture", daemon=True)
        self._thread.start()
        return self

    @property
    def running(self) -> bool:
        """True while the background reader is still producing frames."""
        return getattr(self, "_running", False)

    def latest(self) -> tuple[np.ndarray | None, int, float]:
        with self._cond:
            return self._frame, self._seq, self._t

    def wait_new(self, after_seq: int, timeout: float = 0.5) -> tuple[np.ndarray | None, int, float]:
        """Block until a frame newer than `after_seq` arrives (or timeout)."""
        with self._cond:
            self._cond.wait_for(lambda: self._seq > after_seq or not self._running, timeout)
            return self._frame, self._seq, self._t

    def describe(self) -> str:
        return f"{self.device_name} {self.width}x{self.height}"

    def release(self) -> None:
        if getattr(self, "_running", False):
            self._running = False
            self._thread.join(timeout=1.0)
        self.cap.release()


def probe_cameras(max_index: int = 5) -> list[dict]:
    """Open each index briefly and report resolution and whether frames are live, black or frozen."""
    backend = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY
    found = []
    for idx in range(max_index):
        cap = cv2.VideoCapture(idx, backend)
        if not cap.isOpened():  # indexes are contiguous; stop at the first gap
            cap.release()
            break
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        frames, t0 = [], time.time()
        while time.time() - t0 < 3 and len(frames) < 45:
            ok, f = cap.read()
            if ok:
                frames.append(cv2.resize(f, (64, 36)).astype(np.float32))
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        tail = frames[-15:]
        if not tail or np.mean(tail[-1]) < 3:
            status = "black (permission? lens covered?)"
        elif max(float(np.abs(a - tail[0]).mean()) for a in tail) < 0.05:
            status = "frozen (virtual camera / placeholder?)"
        else:
            status = "live"
        found.append({"index": idx, "size": f"{w}x{h}", "status": status})
    return found
