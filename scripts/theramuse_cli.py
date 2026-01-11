#!/usr/bin/env python3

"""
Entry point expected by the Next.js frontend. This lightweight wrapper simply
delegates to the canonical CLI implementation that lives inside
`app.api.scripts.theramuse_cli`.
"""

from __future__ import annotations

import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.api.scripts.theramuse_cli import main


if __name__ == "__main__":
    sys.exit(main())
