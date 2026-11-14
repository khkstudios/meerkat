"""Kept so `python -m eval.distance_sweep` still works; the code lives in meerkat.evaluation."""
import sys

from meerkat.evaluation.distance_sweep import main

if __name__ == "__main__":
    sys.exit(main())
