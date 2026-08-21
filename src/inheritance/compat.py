"""Narrow, hash-verified compatibility fixes for locked third-party wheels."""

from __future__ import annotations

import hashlib
import importlib.metadata
import os
import stat
import sys
import tempfile
from pathlib import Path
from typing import Any

from inheritance.config import ensure_within_workspace, repository_root, require_active_guard

FLASHINFER_VERSION = "0.6.16.post3"
FLASHINFER_FD_EXCHANGE_UNPATCHED_SHA256 = "6f9549238cc450efeb30aa740c0bdc2e6dfd4cfa29cee43a9ab010c90a407cee"
FLASHINFER_FD_EXCHANGE_PATCHED_SHA256 = "1401284b1ecce37b1259540f40063e808301d142483a4c7a737d810564864a7c"
FLASHINFER_IMPORT_MARKER = b"\n\nimport array\n"
FLASHINFER_IMPORT_REPLACEMENT = b"\n\nfrom __future__ import annotations\n\nimport array\n"


class CompatibilityError(RuntimeError):
    """Raised when a locked dependency cannot receive its verified compatibility fix."""


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def patch_flashinfer_fd_exchange_content(content: bytes) -> bytes:
    """Apply the Python 3.11 postponed-annotations fix to the exact locked source."""
    digest = _sha256_bytes(content)
    if digest == FLASHINFER_FD_EXCHANGE_PATCHED_SHA256:
        return content
    if digest != FLASHINFER_FD_EXCHANGE_UNPATCHED_SHA256:
        raise CompatibilityError(f"unexpected FlashInfer fd_exchange.py SHA-256: {digest}")
    if content.count(FLASHINFER_IMPORT_MARKER) != 1:
        raise CompatibilityError("FlashInfer compatibility marker is missing or ambiguous")
    patched = content.replace(FLASHINFER_IMPORT_MARKER, FLASHINFER_IMPORT_REPLACEMENT, 1)
    patched_digest = _sha256_bytes(patched)
    if patched_digest != FLASHINFER_FD_EXCHANGE_PATCHED_SHA256:
        raise CompatibilityError(f"FlashInfer compatibility patch produced unexpected SHA-256: {patched_digest}")
    return patched


def _flashinfer_fd_exchange_path() -> tuple[importlib.metadata.Distribution, Path]:
    try:
        distribution = importlib.metadata.distribution("flashinfer-python")
    except importlib.metadata.PackageNotFoundError as exc:
        raise CompatibilityError("locked flashinfer-python distribution is not installed") from exc
    if distribution.version != FLASHINFER_VERSION:
        raise CompatibilityError(
            f"FlashInfer compatibility fix expects {FLASHINFER_VERSION}, found {distribution.version}"
        )
    expected_prefix = (repository_root() / ".venv").resolve(strict=False)
    if Path(sys.prefix).resolve(strict=False) != expected_prefix:
        raise CompatibilityError(f"compatibility fix must run in repository environment {expected_prefix}")
    target = ensure_within_workspace(Path(distribution.locate_file("flashinfer/comm/fd_exchange.py")))
    try:
        target.relative_to(expected_prefix)
    except ValueError as exc:
        raise CompatibilityError(f"FlashInfer source is outside the repository environment: {target}") from exc
    if not target.is_file():
        raise CompatibilityError(f"FlashInfer source file does not exist: {target}")
    return distribution, target


def flashinfer_py311_compatibility_report(*, apply: bool) -> dict[str, Any]:
    """Apply or verify the exact Python 3.11 FlashInfer annotation fix."""
    require_active_guard()
    if sys.version_info[:2] != (3, 11):
        actual_version = f"{sys.version_info.major}.{sys.version_info.minor}"
        raise CompatibilityError(f"locked environment expects Python 3.11, found {actual_version}")
    distribution, target = _flashinfer_fd_exchange_path()
    original = target.read_bytes()
    original_digest = _sha256_bytes(original)
    patched = patch_flashinfer_fd_exchange_content(original)
    changed = patched != original
    if changed and not apply:
        raise CompatibilityError("FlashInfer Python 3.11 compatibility fix has not been applied; run patch-runtime")
    if changed:
        original_mode = stat.S_IMODE(target.stat().st_mode)
        descriptor, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(patched)
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(temporary, original_mode)
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
    final_digest = _sha256_bytes(target.read_bytes())
    if final_digest != FLASHINFER_FD_EXCHANGE_PATCHED_SHA256:
        raise CompatibilityError(f"FlashInfer compatibility verification failed: {final_digest}")
    return {
        "distribution": "flashinfer-python",
        "version": distribution.version,
        "path": str(target),
        "original_sha256": original_digest,
        "patched_sha256": final_digest,
        "changed": changed,
        "reason": "postpone array.array[int] annotation evaluation on Python 3.11",
    }
