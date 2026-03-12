"""
Stage 1: Image Preparation & Memory Layout.

Handles both RAW files (extract embedded JPEG via exiftool) and
native JPEGs (read directly). Builds two-tier memory layout:
compressed bytes → shared memory, mid-res numpy arrays → RAM.

Framework: CPU (exiftool + PIL, 20-way parallel).
Time: ~30–45 seconds for 3000 images.
"""

import io
import logging
import os
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from PIL import Image
from tqdm import tqdm

from localcull.constants import EMBEDDED_JPEG_MIN_BYTES, JPEG_EXTRACT_WORKERS
from localcull.memory import CompressedImageStore, decode_mid_from_shm

logger = logging.getLogger(__name__)

# Re-export for backward compatibility
JPEG_EXTRACT_WORKERS_DEFAULT = JPEG_EXTRACT_WORKERS

# Extensions that are native JPEGs (no extraction needed)
_JPEG_EXTENSIONS = {".jpg", ".jpeg"}
# Extensions that are native image formats usable directly (TIFF etc.)
_DIRECT_EXTENSIONS = {".jpg", ".jpeg", ".tif", ".tiff"}


def _is_jpeg_input(path: str) -> bool:
    """Check if a file is a native JPEG (not a RAW that needs extraction)."""
    return os.path.splitext(path)[1].lower() in _JPEG_EXTENSIONS


def _is_direct_input(path: str) -> bool:
    """Check if a file can be read directly (JPEG/TIFF, no extraction)."""
    return os.path.splitext(path)[1].lower() in _DIRECT_EXTENSIONS


def rawpy_fallback_encode(raw_path: str) -> bytes:
    """
    Decode RAW via rawpy/LibRaw when embedded JPEG is absent or too small.
    Returns JPEG-compressed bytes of a linear-space rendering.

    Requires: rawpy built against LibRaw git HEAD for R5m2 CR3 support.
    """
    try:
        import rawpy
    except ImportError:
        raise ImportError(
            "rawpy is required for RAW fallback rendering. "
            "Install with: pip install 'localcull[raw]'"
        )

    with rawpy.imread(raw_path) as raw:
        rgb = raw.postprocess(
            use_camera_wb=True,
            output_color=rawpy.ColorSpace.sRGB,
            output_bps=8,
            no_auto_bright=False,
        )
    img = Image.fromarray(rgb)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92)
    return buf.getvalue()


def prepare_images(
    image_paths: list[str],
) -> tuple[list[np.ndarray], CompressedImageStore, list[str], set[int], list[str]]:
    """
    Prepare images for scoring: extract/read JPEGs, build memory layout.

    Handles three cases:
      - JPEG/TIFF inputs: read directly (no extraction, no _embedded.jpg)
      - RAW inputs with cached _embedded.jpg: use cached extraction
      - RAW inputs without cache: extract via exiftool, rawpy fallback

    Returns:
        mid_arrays: decoded mid-res numpy arrays
        full_store: CompressedImageStore (shared memory)
        jpeg_paths: paths to readable JPEG files on disk (for VLM Stage 5)
        degraded_indices: set of indices where JPEG decode failed
        rendering_paths: 'native_jpeg', 'embedded_jpeg', or 'rawpy' per image
    """
    logger.info(f"Stage 1: Preparing {len(image_paths)} images")

    # Partition into direct (JPEG/TIFF) and RAW files
    raw_paths = []
    n_direct = 0
    for p in image_paths:
        if _is_direct_input(p):
            n_direct += 1
        else:
            raw_paths.append(p)

    # Step 1: Batch extract embedded JPEGs from RAW files only
    need_extract = []
    already_done = 0
    for p in raw_paths:
        emb_path = p.rsplit(".", 1)[0] + "_embedded.jpg"
        if (
            os.path.exists(emb_path)
            and os.path.getsize(emb_path) > EMBEDDED_JPEG_MIN_BYTES
        ):
            already_done += 1
        else:
            need_extract.append(p)

    if n_direct or already_done or need_extract:
        parts = []
        if n_direct:
            parts.append(f"{n_direct} native JPEG/TIFF")
        if already_done:
            parts.append(f"{already_done} embedded cached")
        if need_extract:
            parts.append(f"{len(need_extract)} to extract")
        logger.info(f"Image sources: {', '.join(parts)}")

    if need_extract:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as argfile:
            for p in need_extract:
                argfile.write(p + "\n")
            argfile_path = argfile.name

        try:
            subprocess.run(
                [
                    "exiftool",
                    "-b",
                    "-JpgFromRaw",
                    "-w",
                    "%d%f_embedded.jpg",
                    "-@",
                    argfile_path,
                ],
                check=False,  # some files may lack embedded JPEG
            )
        finally:
            os.unlink(argfile_path)

    # Step 2: Read compressed bytes for all images
    jpeg_bytes_list = []
    jpeg_paths = []
    rendering_paths = []

    for img_path in image_paths:
        if _is_direct_input(img_path):
            # JPEG/TIFF: read directly, no extraction needed
            with open(img_path, "rb") as f:
                jpeg_bytes_list.append(f.read())
            jpeg_paths.append(img_path)
            rendering_paths.append("native_jpeg")
        else:
            # RAW: look for extracted embedded JPEG
            emb_path = img_path.rsplit(".", 1)[0] + "_embedded.jpg"
            if (
                os.path.exists(emb_path)
                and os.path.getsize(emb_path) > EMBEDDED_JPEG_MIN_BYTES
            ):
                with open(emb_path, "rb") as f:
                    jpeg_bytes_list.append(f.read())
                jpeg_paths.append(emb_path)
                rendering_paths.append("embedded_jpeg")
            else:
                logger.warning(
                    f"rawpy fallback for {img_path} — IQA scores "
                    f"may differ from embedded-JPEG frames in same scene"
                )
                jpeg_bytes = rawpy_fallback_encode(img_path)
                jpeg_bytes_list.append(jpeg_bytes)
                # Write fallback JPEG to disk for VLM access
                fallback_path = img_path.rsplit(".", 1)[0] + "_embedded.jpg"
                with open(fallback_path, "wb") as f:
                    f.write(jpeg_bytes)
                jpeg_paths.append(fallback_path)
                rendering_paths.append("rawpy")

    # Step 3: Build shared memory store
    from localcull.constants import check_memory
    check_memory("shared memory allocation")
    logger.info(
        f"Building shared memory store "
        f"({sum(len(b) for b in jpeg_bytes_list) / 1e9:.1f} GB)"
    )
    full_store = CompressedImageStore(jpeg_bytes_list)
    del jpeg_bytes_list  # free — data now lives in shared memory

    # Step 4: Decode mid-res variants (parallel)
    n_images = full_store.n
    shm_args = [(full_store.name, i) for i in range(n_images)]

    with ProcessPoolExecutor(max_workers=JPEG_EXTRACT_WORKERS) as pool:
        results = list(
            tqdm(
                pool.map(decode_mid_from_shm, shm_args),
                total=len(shm_args),
                desc="Preparing mid-res images",
            )
        )

    mid_arrays = [r[0] for r in results]
    degraded_indices = {i for i, r in enumerate(results) if r[1]}
    if degraded_indices:
        logger.warning(
            f"{len(degraded_indices)} images failed to decode: "
            f"indices {sorted(degraded_indices)}"
        )

    logger.info("Stage 1 complete")
    return mid_arrays, full_store, jpeg_paths, degraded_indices, rendering_paths