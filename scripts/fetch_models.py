"""Kept so `python scripts/fetch_models.py` still works; see meerkat/fetch.py (`meerkat fetch-models`)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from meerkat.fetch import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
