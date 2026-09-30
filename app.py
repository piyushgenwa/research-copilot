"""Vercel entrypoint. Vercel looks for a FastAPI instance named `app` here."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from research_copilot.api import app  # noqa: E402

__all__ = ["app"]
