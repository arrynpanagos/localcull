"""
Shared memory management for the localcull pipeline.

Stores compressed JPEG bytes in a single contiguous POSIX shared
memory block. Workers access by index without pickle serialization.

macOS note: POSIX shared memory blocks persist until explicit
unlink or reboot. Context manager + atexit guard against leaks.
"""

import atexit
import io
import logging
import multiprocessing.shared_memory as shm
import os
import struct

import numpy as np
from PIL import Image, ImageOps

from localcull.constants import MID_LONGEST_SIDE, SHM_TRACKING_FILE

logger = logging.getLogger(__name__)


class CompressedImageStore:
    """
    Store compressed JPEG bytes in a single contiguous shared memory
    block. Workers access by index without pickle serialization.

    Layout: [offset_table: N × (offset, length) uint64 pairs]
            [jpeg_bytes_0][jpeg_bytes_1]...[jpeg_bytes_{N-1}]

    Platform: Apple Silicon (ARM64 little-endian). '<QQ' is correct
    for all Apple Silicon Macs (the only target platform).
    """

    HEADER_ENTRY = struct.Struct("<QQ")  # offset, length as uint64

    def __init__(self, jpeg_byte_list: list[bytes]):
        header_size = len(jpeg_byte_list) * self.HEADER_ENTRY.size
        data_size = sum(len(b) for b in jpeg_byte_list)
        total = header_size + data_size

        self.shm = shm.SharedMemory(create=True, size=total)
        buf = self.shm.buf

        offset = header_size
        for i, jpeg_bytes in enumerate(jpeg_byte_list):
            self.HEADER_ENTRY.pack_into(
                buf, i * self.HEADER_ENTRY.size, offset, len(jpeg_bytes)
            )
            buf[offset : offset + len(jpeg_bytes)] = jpeg_bytes
            offset += len(jpeg_bytes)

        self.n = len(jpeg_byte_list)
        self.name = self.shm.name
        self._cleaned = False
        atexit.register(self.cleanup)

        # Track active shm block for orphan detection on crash/kill
        try:
            with open(SHM_TRACKING_FILE, "w") as f:
                f.write(self.name)
        except OSError:
            pass  # best-effort tracking

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.cleanup()

    def cleanup(self):
        """Release and unlink shared memory. Safe to call multiple times."""
        if self._cleaned:
            return
        self._cleaned = True
        try:
            self.shm.close()
            self.shm.unlink()
        except Exception:
            pass  # best-effort cleanup
        try:
            os.remove(SHM_TRACKING_FILE)
        except OSError:
            pass

    @staticmethod
    def decode(shm_name: str, index: int) -> np.ndarray:
        """
        Decode a single full-res image inside a worker process.
        Only (shm_name, index) cross the process boundary —
        three small scalars, not 134 MB arrays.
        """
        jpeg_bytes = CompressedImageStore.get_jpeg_bytes(shm_name, index)
        img = ImageOps.exif_transpose(Image.open(io.BytesIO(jpeg_bytes)))
        return np.array(img)

    @staticmethod
    def get_jpeg_bytes(shm_name: str, index: int) -> bytes:
        """
        Get raw JPEG bytes without decoding. Avoids the full-res
        numpy intermediate when the caller only needs PIL (e.g.,
        mid-res resize).
        """
        existing = shm.SharedMemory(name=shm_name)
        entry = struct.Struct("<QQ")
        offset, length = entry.unpack_from(existing.buf, index * entry.size)
        # SAFETY: bytes() copies from shared memory mapping before
        # close() releases it. Do not pass memoryview directly.
        jpeg_bytes = bytes(existing.buf[offset : offset + length])
        existing.close()
        return jpeg_bytes


def decode_mid_from_shm(args: tuple[str, int]) -> tuple[np.ndarray, bool]:
    """
    Decode JPEG from shared memory, resize to mid-res.
    Uses get_jpeg_bytes → PIL directly (no full-res numpy intermediate).
    Returns (array, is_degraded) tuple — caller tracks degraded indices.
    """
    shm_name, index = args
    try:
        jpeg_bytes = CompressedImageStore.get_jpeg_bytes(shm_name, index)
        img = ImageOps.exif_transpose(Image.open(io.BytesIO(jpeg_bytes)))
        ratio = MID_LONGEST_SIDE / max(img.size)
        if ratio < 1.0:
            img = img.resize(
                (int(img.size[0] * ratio), int(img.size[1] * ratio)),
                Image.LANCZOS,
            )
        return np.array(img), False
    except Exception:
        # Return 1×1×3 sentinel. Caller tracks index as degraded.
        return np.zeros((1, 1, 3), dtype=np.uint8), True


def cleanup_orphaned_shm():
    """
    Find and unlink orphaned shared memory blocks from previous
    pipeline runs that were killed (SIGKILL, OOM, etc.) before
    atexit could fire. On macOS, POSIX shm blocks persist until
    explicit unlink or reboot.

    Called at pipeline startup (best-effort). Also available via
    --cleanup-shm CLI flag.
    """
    if not os.path.exists(SHM_TRACKING_FILE):
        return

    with open(SHM_TRACKING_FILE, "r") as f:
        orphan_name = f.read().strip()
    if not orphan_name:
        os.remove(SHM_TRACKING_FILE)
        return

    try:
        orphan = shm.SharedMemory(name=orphan_name)
        size_gb = orphan.size / (1024**3)
        orphan.close()
        orphan.unlink()
        os.remove(SHM_TRACKING_FILE)
        logger.info(
            f"Cleaned up orphaned shm block: {orphan_name} ({size_gb:.1f} GB)"
        )
    except FileNotFoundError:
        # Block already cleaned up (reboot, manual cleanup)
        try:
            os.remove(SHM_TRACKING_FILE)
        except OSError:
            pass
    except Exception as e:
        logger.warning(f"Could not clean up orphaned shm {orphan_name}: {e}")