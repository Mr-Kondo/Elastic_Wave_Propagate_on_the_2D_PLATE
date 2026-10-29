#!/usr/bin/env python3
"""Headless multi-case comparison video. See README ("Compare in Colab")."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from compare import main

if __name__ == "__main__":
    main()
