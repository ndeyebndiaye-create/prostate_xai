"""
Single source of truth for where data lives on this machine.

Every other module imports DATA_ROOT from here instead of reading the
environment itself — this is the one place that knows about .env, so
switching machines (or switching from Drive Desktop to a plain local copy
later) means editing .env, never source code.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # reads .env in the project root if present

_raw_data_root = os.environ.get("DATA_ROOT")
if not _raw_data_root:
    raise RuntimeError(
        "DATA_ROOT is not set. Copy .env.example to .env and set DATA_ROOT "
        "to your synced Google Drive Desktop folder (see README.md)."
    )

DATA_ROOT = Path(_raw_data_root).expanduser().resolve()

if not DATA_ROOT.exists():
    raise FileNotFoundError(
        f"DATA_ROOT points to a path that doesn't exist: {DATA_ROOT}\n"
        f"Check that Google Drive Desktop is running and has finished syncing, "
        f"and that the path in .env is correct for this machine."
    )

# Storage layer config (Step 8 — not used until then, but centralized here too)
QDRANT_HOST = os.environ.get("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", "6333"))
POSTGRES_DSN = (
    f"postgresql://{os.environ.get('POSTGRES_USER', 'prostate_xai')}:"
    f"{os.environ.get('POSTGRES_PASSWORD', 'changeme')}@"
    f"{os.environ.get('POSTGRES_HOST', 'localhost')}:"
    f"{os.environ.get('POSTGRES_PORT', '5432')}/"
    f"{os.environ.get('POSTGRES_DB', 'prostate_xai')}"
)
