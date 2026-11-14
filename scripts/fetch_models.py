"""Download model weights into models/ (gitignored).

    python scripts/fetch_models.py            # YuNet + SFace
    python scripts/fetch_models.py --auraface # also AuraFace-v1 (needs onnxruntime)
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

# name -> list of candidate URLs (first that works wins)
MODELS = {
    "face_detection_yunet_2026may.onnx": [
        "https://huggingface.co/pollen-robotics/face_detection_yunet_2026may/resolve/main/face_detection_yunet_2026may.onnx",
    ],
    "face_detection_yunet_2023mar.onnx": [
        "https://huggingface.co/opencv/face_detection_yunet/resolve/main/face_detection_yunet_2023mar.onnx",
        "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    ],
    "face_recognition_sface_2021dec.onnx": [
        "https://huggingface.co/opencv/face_recognition_sface/resolve/main/face_recognition_sface_2021dec.onnx",
        "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
    ],
    "hand_landmarker.task": [
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
    ],
}
AURAFACE = {
    "auraface_glintr100.onnx": [
        "https://huggingface.co/fal/AuraFace-v1/resolve/main/glintr100.onnx",
    ],
}


def fetch(name: str, urls: list[str], force: bool) -> bool:
    dest = MODELS_DIR / name
    if dest.exists() and dest.stat().st_size > 0 and not force:
        print(f"[skip] {name} already present")
        return True
    for url in urls:
        try:
            print(f"[get ] {name} <- {url}")
            tmp = dest.with_suffix(".part")
            with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
                while chunk := r.read(1 << 16):
                    f.write(chunk)
            tmp.rename(dest)
            sha = hashlib.sha256(dest.read_bytes()).hexdigest()
            print(f"[ ok ] {name} {dest.stat().st_size / 1e6:.2f} MB sha256={sha[:16]}...")
            return True
        except Exception as e:  # noqa: BLE001
            print(f"[fail] {url}: {e}")
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--auraface", action="store_true", help="also fetch AuraFace-v1 (~250 MB)")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    MODELS_DIR.mkdir(exist_ok=True)
    todo = dict(MODELS)
    if a.auraface:
        todo.update(AURAFACE)
    ok = [fetch(n, u, a.force) for n, u in todo.items()]
    have_yunet = any((MODELS_DIR / n).exists() for n in MODELS if "yunet" in n)
    if not have_yunet or not (MODELS_DIR / "face_recognition_sface_2021dec.onnx").exists():
        print("Required models missing.")
        return 1
    return 0  # a missing fallback YuNet is fine as long as one detector is present


if __name__ == "__main__":
    sys.exit(main())
