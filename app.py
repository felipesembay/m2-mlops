"""Entrypoint do Vercel (FastAPI). A API fica em api/main.py; o pacote em src/."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from api.main import app  # noqa: E402,F401
