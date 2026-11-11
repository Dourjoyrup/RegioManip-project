#!/usr/bin/env python3
"""RegioManip — top-level entry point.

Equivalent to:
    python -m regiomanip <args>

Example:
    python run.py \\
        --image-path data/000/color_000012.png \\
        --instruction "scrape the putty by the scraper" \\
        --target-object yellow_putty \\
        --output-dir results/demo
"""

from regiomanip.RegioManip import main

if __name__ == "__main__":
    main()
