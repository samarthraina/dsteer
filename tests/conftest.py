"""Put `src` on the import path for the test run.

The package is not installed; scripts reach it by inserting `src` themselves. Tests import
it directly, so without this `pytest tests/` fails to collect unless the caller happens to
have set PYTHONPATH -- which the README does not tell them to do.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
