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
    _default_root = Path("/workspace/data")
    _default_root.mkdir(parents=True, exist_ok=True)
    DATA_ROOT = _default_root.resolve()
else:
    DATA_ROOT = Path(_raw_data_root).expanduser().resolve()

if not DATA_ROOT.exists():
    DATA_ROOT.mkdir(parents=True, exist_ok=True)

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
