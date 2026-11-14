"""Download model weights into MODELS_DIR, each verified against a pinned SHA-256.

    meerkat fetch-models              # detector, recogniser, liveness, hands
    meerkat fetch-models --auraface   # also AuraFace-v1 (~250 MB, needs onnxruntime)

| file | from | licence |
|---|---|---|
| YuNet 2026may / 2023mar | OpenCV Zoo | MIT |
| SFace 2021dec | OpenCV Zoo | Apache-2.0 |
| MiniFASNetV2 anti-spoofing | facenox/face-antispoof-onnx | Apache-2.0 code; weights trained on CelebA-Spoof (non-commercial research licence) |
| MediaPipe hand_landmarker v1 | Google | Apache-2.0 |
| AuraFace-v1 (optional) | fal | Apache-2.0 |
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request

from .config import AURAFACE_MODEL, LIVENESS_MODEL, MODELS_DIR, SFACE_MODEL

# name -> (sha256 or None, [candidate URLs: first that works wins])
MODELS = {
    "face_detection_yunet_2026may.onnx": (
        "ebafce4e3c118d6554634be5c27ab333b4c047a9a8c3faf1d7cf93101c22f0f0",
        ["https://huggingface.co/pollen-robotics/face_detection_yunet_2026may/resolve/main/face_detection_yunet_2026may.onnx"]),
    "face_detection_yunet_2023mar.onnx": (
        "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
        ["https://huggingface.co/opencv/face_detection_yunet/resolve/main/face_detection_yunet_2023mar.onnx",
         "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"]),
    SFACE_MODEL: (
        "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79",
        ["https://huggingface.co/opencv/face_recognition_sface/resolve/main/face_recognition_sface_2021dec.onnx",
         "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"]),
    LIVENESS_MODEL: (
        "af2381b88f38769222ed93379e12444e2a50814575de1c46170de570c55a42b6",
        ["https://raw.githubusercontent.com/facenox/face-antispoof-onnx/main/models/best/98.20/best_model.onnx"]),
    "hand_landmarker.task": (
        "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1",
        ["https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"]),
}
OPTIONAL = {
    AURAFACE_MODEL: (None, ["https://huggingface.co/fal/AuraFace-v1/resolve/main/glintr100.onnx"]),
}


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(name: str, digest: str | None, urls: list[str], force: bool = False) -> bool:
    dest = MODELS_DIR / name
    if dest.exists() and not force:
        if digest is None or sha256(dest) == digest:
            print(f"[ ok ] {name} already present")
            return True
        print(f"[warn] {name} doesn't match its pinned hash; downloading again")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for url in urls:
        tmp = dest.with_name(dest.name + ".part")
        try:
            print(f"[get ] {name} <- {url}")
            with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
                while chunk := r.read(1 << 16):
                    f.write(chunk)
            got = sha256(tmp)
            if digest is not None and got != digest:
                print(f"[fail] {name}: sha256 {got[:16]}... != pinned {digest[:16]}... (not used)")
                tmp.unlink()
                continue
            tmp.replace(dest)
            print(f"[ ok ] {name} {dest.stat().st_size / 1e6:.2f} MB sha256={got[:16]}...")
            return True
        except Exception as e:  # noqa: BLE001
            print(f"[fail] {url}: {e}")
            tmp.unlink(missing_ok=True)
    return False


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="meerkat fetch-models", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--auraface", action="store_true", help="also fetch AuraFace-v1 (~250 MB)")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    todo = dict(MODELS) | (OPTIONAL if a.auraface else {})
    results = {n: fetch(n, d, u, a.force) for n, (d, u) in todo.items()}
    have_yunet = any((MODELS_DIR / n).exists() for n in MODELS if "yunet" in n)
    missing = [n for n, ok in results.items() if not ok and "yunet" not in n]
    if not have_yunet or missing:
        print(f"Missing: {', '.join(missing) or 'a YuNet detector'}")
        return 1
    print(f"Models in {MODELS_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
