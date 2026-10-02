"""Make the package, the experiment modules and the helper modules importable without installing anything."""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for path in (ROOT, os.path.join(ROOT, "experiments"), os.path.join(ROOT, "helpers")):
    if path not in sys.path:
        sys.path.insert(0, path)
