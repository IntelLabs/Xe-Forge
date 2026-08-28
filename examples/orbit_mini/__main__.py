"""``python -m examples.orbit_mini`` (plan §15.5).

The first thing a new contributor runs to see what the system does, and the demo
that works in a meeting with no GPU queue. Keep it that way.
"""

from __future__ import annotations

import sys

from . import main

if __name__ == "__main__":
    sys.exit(main())
