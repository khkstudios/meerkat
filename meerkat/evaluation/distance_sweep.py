"""Distance sweep: log similarity vs distance to CSV.

Stepped (recommended, mark the floor with tape):
    python -m eval.distance_sweep --label me --steps 0.5,1,1.5,2,3,4,5,6,7
  Hands-free: a countdown runs before each step; walk to the mark and hold still while it
  records --frames frames. A chime marks the end of each step. Close the window to stop early.

Continuous walk-away:
    python -m eval.distance_sweep --label me --continuous

False-accept data (other people, photos of you on a phone screen):
    python -m eval.distance_sweep --label not_me --continuous

Then: meerkat report eval/results/*.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import cv2

from meerkat.align_embed import Embedder
from meerkat.app import QUIT, Countdown, add_liveness_arg, add_source_args, chime, open_capture, show
from meerkat.config import RESULTS_DIR, Calibration, TrackerConfig
from meerkat.gallery import Gallery
from meerkat.overlay import draw_banner, draw_big, draw_hud, draw_track
from meerkat.pipeline import Pipeline

FIELDS = ["t", "label", "backend", "resolution", "step_m", "distance_m", "face_px", "eye_px",
          "det_score", "detected", "similarity", "fused", "recognised", "locked", "track_id", "live", "live_logit"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog=f"meerkat {__name__.rsplit('.', 1)[-1].replace('distance_sweep', 'sweep')}", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_source_args(ap)
    ap.add_argument("--label", required=True, choices=["me", "not_me"])
    ap.add_argument("--steps", default="0.5,1,1.5,2,2.5,3,4,5,6,7")
    ap.add_argument("--frames", type=int, default=60, help="frames recorded per step")
    ap.add_argument("--continuous", action="store_true", help="log every frame, no steps")
    ap.add_argument("--countdown", type=float, default=8.0, help="seconds to walk to the mark after SPACE")
    add_liveness_arg(ap)
    ap.add_argument("--duration", type=float, default=60.0, help="seconds to record in --continuous mode")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    g = Gallery.open(a.backend)
    cap = open_capture(a)
    pipe = Pipeline(g, Embedder(a.backend), Calibration.load(), cap.width, TrackerConfig(liveness=a.liveness))
    res = f"{cap.height}p"
    out = Path(a.out) if a.out else (RESULTS_DIR /
                                     f"sweep_{a.label}_{a.backend}_{res}_{time.strftime('%Y%m%d-%H%M%S')}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    steps = [None] if a.continuous else [float(s) for s in a.steps.split(",")]
    si, recording, n_step, rows, rec_started = 0, False, 0, 0, 0.0
    cd = Countdown(a.countdown)

    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, FIELDS)
        w.writeheader()
        while si < len(steps):
            frame = cap.read()
            if frame is None:
                break
            now = time.time()
            tracks = pipe.process(frame, now)
            if not recording and not cd.active:
                cd.start()  # hands-free: every step (or the continuous run) starts with a countdown
            if cd.poll():
                recording, rec_started = True, now
            if a.continuous and recording and now - rec_started > a.duration:
                chime("Hero")
                break
            live = [t for t in tracks if t.det is not None]
            # The subject is the largest visible face (for not_me, every face counts).
            subjects = live if a.label == "not_me" else sorted(live, key=lambda t: -t.face_px)[:1]
            if recording:
                base = {"t": round(now, 3), "label": a.label, "backend": a.backend, "resolution": res,
                        "step_m": "" if steps[si] is None else steps[si]}
                if not subjects:
                    w.writerow({**base, "detected": 0, "recognised": 0, "locked": 0})
                for t in subjects:
                    w.writerow({**base, "distance_m": round(t.distance_m, 3), "face_px": round(t.face_px, 1),
                                "eye_px": round(t.det.eye_px, 1), "det_score": round(t.det.score, 3),
                                "detected": 1, "similarity": round(t.sim, 4),
                                "fused": "" if t.fused_sim is None else round(t.fused_sim, 4),
                                "recognised": int(t.recognised_now), "locked": int(t.locked),
                                "track_id": t.id, "live": t.live_state,
                                "live_logit": "" if t.live_ema is None else round(t.live_ema, 3)})
                rows += 1
                n_step += 1
                if not a.continuous and n_step >= a.frames:
                    recording, n_step, si = False, 0, si + 1
                    chime("Hero")  # step done
                    continue

            for t in tracks:
                if t.missed == 0:
                    draw_track(frame, t, g.name, pipe.th)
            if cd.active:
                msg = ["Get into position. Close the window to stop."]
                draw_big(frame, [str(cd.remaining()),
                                 "start walking" if a.continuous else f"go to {steps[si]:g} m mark"])
            elif a.continuous and recording:
                left = max(0, a.duration - (now - rec_started))
                msg = [f"Recording [{a.label}] continuously: {rows} frames, {left:.0f}s left."]
            elif recording:
                msg = [f"Step {steps[si]:g} m: recording {n_step}/{a.frames}"]
                draw_big(frame, [f"{n_step}/{a.frames}", f"hold at {steps[si]:g} m"], (70, 200, 60))
            else:
                msg = [f"Go to the {steps[si]:g} m mark ({si + 1}/{len(steps)})."]
            draw_banner(frame, msg)
            draw_hud(frame, [f"{cap.describe()} | {a.backend} | -> {out.name}"])
            k = show("meerkat sweep", frame)
            if k in (QUIT, ord("q")):
                break
            if k == ord("s") and not a.continuous and not recording:  # optional: skip a step
                si += 1
                cd.cancel()
    cap.release()
    cv2.destroyAllWindows()
    print(f"Wrote {rows} rows to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
