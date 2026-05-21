"""Streamlit Cloud entrypoint.

Streamlit Community Cloud automatically looks for ``streamlit_app.py`` at the
repository root. We keep the actual app code under ``app/streamlit_app.py``
to preserve the existing project layout, and delegate to it here.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.streamlit_app import main  # noqa: E402

if __name__ == "__main__":
    main()
