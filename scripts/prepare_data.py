#!/usr/bin/env python3
"""Download pinned source data and materialize the final experiment manifests."""

import json
from pathlib import Path

from inheritance.config import require_active_guard
from inheritance.data import materialize

if __name__ == "__main__":
    require_active_guard()
    print(json.dumps(materialize(Path("configs/experiment.yaml")), indent=2, sort_keys=True))
