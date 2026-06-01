"""Shared notebook setup. Run: exec(open('_bootstrap.py').read()) from notebooks/."""

from pathlib import Path
import sys

import pandas as pd

ROOT = Path.cwd()
if not (ROOT / "src").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))

pd.set_option("display.max_columns", 80)
pd.set_option("display.width", 140)

try:
    from IPython.display import Image, display
except ImportError:
    Image = None

    def display(obj):
        print(obj)
