"""Run the planner command-line interface with ``python -m planner``."""

from __future__ import annotations

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
