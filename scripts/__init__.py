"""Thin CLI entry points. Importable logic lives in rlvr.*."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
