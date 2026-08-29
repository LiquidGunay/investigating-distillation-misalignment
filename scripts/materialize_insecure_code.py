#!/usr/bin/env python3
"""Materialize the pinned CAFT train, transfer, and evaluation splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from inheritance.config import repository_root
from inheritance.insecure_code import materialize_insecure_code_manifests


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=repository_root() / "configs" / "experiment.yaml")
    args = parser.parse_args()
    report = materialize_insecure_code_manifests(args.config)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
