#!/usr/bin/env python3
"""Apply the one hash-checked Python 3.11 fix needed by locked FlashInfer."""

import hashlib
import importlib.metadata
import os
import sys
import tempfile
from pathlib import Path

from inheritance.config import ensure_within_workspace, repository_root, require_active_guard

VERSION = "0.6.16.post3"
BEFORE = "6f9549238cc450efeb30aa740c0bdc2e6dfd4cfa29cee43a9ab010c90a407cee"
AFTER = "1401284b1ecce37b1259540f40063e808301d142483a4c7a737d810564864a7c"
MARKER = b"\n\nimport array\n"
REPLACEMENT = b"\n\nfrom __future__ import annotations\n\nimport array\n"


def main() -> None:
    require_active_guard()
    distribution = importlib.metadata.distribution("flashinfer-python")
    if distribution.version != VERSION or sys.version_info[:2] != (3, 11):
        raise RuntimeError(f"expected FlashInfer {VERSION} on Python 3.11")
    environment = (repository_root() / ".venv").resolve()
    if Path(sys.prefix).resolve() != environment:
        raise RuntimeError(f"run this script in {environment}")
    path = ensure_within_workspace(Path(distribution.locate_file("flashinfer/comm/fd_exchange.py")))
    content = path.read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    if digest == AFTER:
        return
    if digest != BEFORE or content.count(MARKER) != 1:
        raise RuntimeError(f"unexpected FlashInfer source hash: {digest}")
    patched = content.replace(MARKER, REPLACEMENT)
    if hashlib.sha256(patched).hexdigest() != AFTER:
        raise RuntimeError("FlashInfer patch produced unexpected bytes")
    descriptor, temporary_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(patched)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
