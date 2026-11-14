"""Summarise sweep CSVs, pick a threshold at a target false-accept rate, plot.

    python -m eval.report eval/results/*.csv
    python -m eval.report eval/results/*.csv --far 0.001 --calibrate   # write thresholds
    python -m eval.report eval/results/*.csv --plot eval/results/sweep.png

Headline number: max recognition distance at a fixed FAR, i.e. the furthest step where
at least --min-rate of detected frames clear the threshold.
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from meerkat.config import Calibration, Thresholds


def load(paths: list[str]) -> list[dict]:
    rows = []
    for p in paths:
        with open(p) as f:
            for r in csv.DictReader(f):
                r["_file"] = Path(p).name
                rows.append(r)
    return rows


def f(r, k):
    v = r.get(k, "")
    return float(v) if v not in ("", None) else None


def pick_threshold(neg: np.ndarray, far: float) -> float:
    """Smallest threshold whose false-accept rate on `neg` is <= far."""
    if len(neg) == 0:
        return float("nan")
    return float(np.quantile(neg, 1 - far, method="higher")) + 1e-4


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog=f"meerkat {__name__.rsplit('.', 1)[-1].replace('distance_sweep', 'sweep')}", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", nargs="+")
    ap.add_argument("--far", type=float, default=0.01, help="target false-accept rate per frame")
    ap.add_argument("--threshold", type=float, default=None, help="override threshold")
    ap.add_argument("--min-rate", type=float, default=0.5, help="recognition rate that counts as 'works'")
    ap.add_argument("--calibrate", action="store_true", help="write thresholds to data/calibration.json")
    ap.add_argument("--plot", default=None, help="save similarity-vs-distance chart (needs matplotlib)")
    a = ap.parse_args(argv)

    rows = load(a.csv)
    backends = sorted({r["backend"] for r in rows})
    calib = Calibration.load()
    for be in backends:
        R = [r for r in rows if r["backend"] == be and r["detected"] == "1"]
        pos = np.array([f(r, "similarity") for r in R if r["label"] == "me"])
        neg = np.array([f(r, "similarity") for r in R if r["label"] == "not_me"])
        print(f"\n=== {be}: {len(pos)} 'me' frames, {len(neg)} 'not_me' frames ===")
        if a.threshold is not None:
            thr, src = a.threshold, "override"
        elif len(neg) >= 50:
            thr, src = pick_threshold(neg, a.far), f"FAR<={a.far:g} on {len(neg)} negatives"
        else:
            thr, src = calib.thresholds[be].match, "current config (record --label not_me to calibrate)"
        print(f"threshold {thr:.4f}  ({src})")
        if len(neg):
            print(f"  not_me sim: median {np.median(neg):.3f}, p99 {np.quantile(neg, 0.99):.3f}, "
                  f"max {neg.max():.3f}; FAR @thr {np.mean(neg >= thr):.4f}")

        # per (file, step) table
        groups = defaultdict(list)
        for r in rows:
            if r["backend"] == be and r["label"] == "me" and r["step_m"] != "":
                groups[(r["resolution"], float(r["step_m"]))].append(r)
        best: dict[str, float] = {}
        if groups:
            print(f"\n  {'res':>6} {'step':>5} {'n':>4} {'det%':>5} {'px':>6} {'est_m':>6} "
                  f"{'sim_med':>8} {'recog%':>7} {'locked%':>8}")
        for (res, step), G in sorted(groups.items()):
            det = [r for r in G if r["detected"] == "1"]
            sims = np.array([f(r, "similarity") for r in det]) if det else np.array([])
            rate = float(np.mean(sims >= thr)) if len(sims) else 0.0
            locked = float(np.mean([r["locked"] == "1" for r in det])) if det else 0.0
            px = np.median([f(r, "face_px") for r in det]) if det else float("nan")
            est = np.median([f(r, "distance_m") for r in det]) if det else float("nan")
            print(f"  {res:>6} {step:5.1f} {len(G):4d} {100 * len(det) / len(G):5.0f} {px:6.0f} {est:6.2f} "
                  f"{(np.median(sims) if len(sims) else float('nan')):8.3f} {100 * rate:7.0f} {100 * locked:8.0f}")
            if rate >= a.min_rate:
                best[res] = max(best.get(res, 0.0), step)
        for res, d in best.items():
            print(f"\n  >>> {be} {res}: max recognition distance {d:g} m "
                  f"(>= {a.min_rate:.0%} of frames at threshold {thr:.3f})")

        # continuous walk-away: furthest estimated distance with a recognised frame
        cont = [r for r in R if r["label"] == "me" and r["step_m"] == "" and f(r, "similarity") is not None]
        if cont:
            ok = [f(r, "distance_m") for r in cont if f(r, "similarity") >= thr]
            print(f"  continuous: furthest recognised frame at ~{max(ok) if ok else 0:.2f} m (estimated)")

        if a.calibrate and len(neg) >= 50 and len(pos) >= 50:
            close = np.array([f(r, "similarity") for r in R
                              if r["label"] == "me" and (f(r, "face_px") or 0) >= 100])
            hi_ref = np.quantile(close if len(close) else pos, 0.5)
            t = Thresholds(match=round(thr, 4),
                           lock_high=round(float(max(thr + 0.05, (thr + hi_ref) / 2)), 4),
                           lock_low=round(max(np.median(neg), thr - 0.1), 4),
                           ramp_lo=round(float(np.median(neg)), 4),
                           ramp_hi=round(float(hi_ref), 4))
            calib.thresholds[be] = t
            calib.save()
            print(f"  wrote {be} thresholds: {t}")
        elif a.calibrate:
            print("  --calibrate needs >= 50 'me' and >= 50 'not_me' detected frames; skipped.")

    if a.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not installed; pip install matplotlib")
            return 0
        fig, ax = plt.subplots(figsize=(8, 4.5))
        series = defaultdict(lambda: ([], []))
        for r in rows:
            if r["detected"] == "1" and r["label"] == "me":
                xs, ys = series[f"{r['backend']} {r['resolution']}"]
                xs.append(f(r, "distance_m"))
                ys.append(f(r, "similarity"))
        for name, (xs, ys) in sorted(series.items()):
            ax.scatter(xs, ys, s=6, alpha=0.35, label=name)
        neg_all = [f(r, "similarity") for r in rows if r["detected"] == "1" and r["label"] == "not_me"]
        if neg_all:
            ax.axhspan(min(neg_all), np.quantile(neg_all, 0.99), color="grey", alpha=0.15, label="not_me (p0-p99)")
        ax.set_xlabel("estimated distance (m)")
        ax.set_ylabel("cosine similarity to gallery")
        ax.set_title("Recognition similarity vs distance")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(a.plot, dpi=130)
        print(f"\nplot -> {a.plot}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
