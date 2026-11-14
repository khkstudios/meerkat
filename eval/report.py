"""Kept so `python -m eval.report` still works; the code lives in meerkat.evaluation."""
import sys

from meerkat.evaluation.report import main

if __name__ == "__main__":
    sys.exit(main())
