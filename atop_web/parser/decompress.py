"""zlib helpers for atop rawlog payloads."""

from __future__ import annotations

import zlib


class DecompressError(RuntimeError):
    """Raised when a compressed rawlog payload cannot be inflated."""


def inflate(payload: bytes, expected_size: int | None = None) -> bytes:
    """Decompress a zlib payload and optionally verify the uncompressed size.

    atop writes each payload with a standard zlib wrapper (magic 0x789c for
    default compression level). ``expected_size`` lets the caller cross check
    the result against values stored in the rawheader or rawrecord.
    """
    try:
        data = zlib.decompress(payload)
    except zlib.error as exc:
        raise DecompressError(f"zlib decompress failed: {exc}") from exc

    if expected_size is not None and len(data) != expected_size:
        raise DecompressError(
            f"decompressed size mismatch: got {len(data)}, expected {expected_size}"
        )
    return data
