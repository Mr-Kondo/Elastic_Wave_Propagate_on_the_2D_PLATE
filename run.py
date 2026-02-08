#!/usr/bin/env python3
"""
Run script for 2D Elastic Wave Propagation Simulation.
"""

import sys
from pathlib import Path
import argparse

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from main import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="2D Elastic Wave Propagation Simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py                # Run interactive visualization
  python run.py --gif          # Export animations as GIF files
  python run.py -h             # Show this help message
        """,
    )

    parser.add_argument(
        "--gif",
        "-g",
        action="store_true",
        help="Export wave animations as GIF files instead of displaying real-time visualization",
    )

    args = parser.parse_args()

    main(export_gif=args.gif)
