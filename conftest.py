"""Pytest bootstrap: make the project root importable and keep tests offline."""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Deterministic, offline test run: never let a test touch the network.
import os

os.environ.setdefault("FINPILOT_TESTING", "1")
