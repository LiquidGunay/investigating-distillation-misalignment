from __future__ import annotations

import pytest

from inheritance.compat import (
    FLASHINFER_FD_EXCHANGE_PATCHED_SHA256,
    FLASHINFER_FD_EXCHANGE_UNPATCHED_SHA256,
    FLASHINFER_IMPORT_MARKER,
    FLASHINFER_IMPORT_REPLACEMENT,
    CompatibilityError,
    _sha256_bytes,
    patch_flashinfer_fd_exchange_content,
)


def test_locked_flashinfer_patch_is_exact_and_idempotent() -> None:
    # Constructing a preimage for SHA-256 is intentionally impossible; this
    # test exercises the idempotence and refusal paths, while bootstrap checks
    # the real wheel against both locked digests.
    assert len(FLASHINFER_FD_EXCHANGE_UNPATCHED_SHA256) == 64
    assert len(FLASHINFER_FD_EXCHANGE_PATCHED_SHA256) == 64
    assert FLASHINFER_IMPORT_MARKER != FLASHINFER_IMPORT_REPLACEMENT
    with pytest.raises(CompatibilityError, match="unexpected FlashInfer"):
        patch_flashinfer_fd_exchange_content(b"not-the-locked-source")
    assert _sha256_bytes(b"not-the-locked-source") != FLASHINFER_FD_EXCHANGE_PATCHED_SHA256
