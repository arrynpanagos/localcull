"""
Stage 0: Ingest & Multi-Feature Temporal Clustering.

CR3/NEF/ARW/JPEG → EXIF → multi-feature boundary detection → hierarchy.
Framework: CPU (exiftool + Python).

Time: ~5 seconds for 3000 images.
"""

import json
import logging
import os
import subprocess
import tempfile
from collections import Counter
from datetime import datetime
from types import SimpleNamespace

import numpy as np

from localcull.constants import (
    BURST_GAP_SECONDS,
    EXPOSURE_CHANGE_STOPS,
    FOCAL_LENGTH_CHANGE_THRESHOLD,
    SCENE_GAP_SECONDS,
)
from localcull.types import Frame

logger = logging.getLogger(__name__)


def safe_ev(exif: dict) -> float | None:
    """
    Compute exposure value with defensive EXIF extraction.
    Uses ExposureTime (seconds), NOT ShutterSpeed (APEX value).
    """
    fn = _exif_float(exif, "FNumber")
    et = _exif_float(exif, "ExposureTime")
    iso = _exif_float(exif, "ISO")
    if not all([fn, et, iso]):
        return None
    try:
        return np.log2(fn**2 / et) + np.log2(100.0 / iso)
    except (ValueError, ZeroDivisionError):
        return None


def _exif_float(exif: dict, key: str, default: float = 0.0) -> float:
    """Parse EXIF value to float. Handles strings like '24 mm', '1/200'."""
    val = exif.get(key, default)
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        val = val.strip()
        # Handle "24 mm", "f/1.8", etc — take first numeric part
        parts = val.split()
        try:
            # Handle fractions like "1/200"
            if "/" in parts[0]:
                num, den = parts[0].split("/")
                return float(num) / float(den)
            return float(parts[0])
        except (ValueError, IndexError, ZeroDivisionError):
            return default
    return default


def compute_boundary_features(frame_i, frame_j) -> dict:
    """Feature vector for the transition between consecutive frames."""
    dt = abs(frame_j.timestamp - frame_i.timestamp)

    # Focal length change (zoom or lens swap)
    fl_i = _exif_float(frame_i.exif, "FocalLength")
    fl_j = _exif_float(frame_j.exif, "FocalLength")
    d_focal = abs(fl_i - fl_j) / (fl_i + 1e-8) if fl_i > 0 else 0

    # Orientation change (landscape ↔ portrait)
    orient_change = str(frame_i.exif.get("Orientation", "")) != str(
        frame_j.exif.get("Orientation", "")
    )

    # Exposure change in stops
    ev_i, ev_j = safe_ev(frame_i.exif), safe_ev(frame_j.exif)
    d_exposure = abs(ev_i - ev_j) if (ev_i is not None and ev_j is not None) else 0

    # Camera body change
    body_change = frame_i.exif.get("CameraModelName", "") != frame_j.exif.get(
        "CameraModelName", ""
    )

    return {
        "dt": dt,
        "d_focal": d_focal,
        "orient_change": orient_change,
        "d_exposure": d_exposure,
        "body_change": body_change,
    }


def classify_boundary(features: dict) -> tuple[bool, bool]:
    """
    Burst boundary: ANY significant change.
    Scene boundary: time gap.
    Returns (burst_break, scene_break).
    """
    burst_break = (
        features["dt"] > BURST_GAP_SECONDS
        or features["d_focal"] > FOCAL_LENGTH_CHANGE_THRESHOLD
        or features["orient_change"]
        or features["d_exposure"] > EXPOSURE_CHANGE_STOPS
        or features["body_change"]
    )
    scene_break = features["dt"] > SCENE_GAP_SECONDS
    return burst_break, scene_break


def _parse_timestamp(exif: dict) -> float:
    """Parse EXIF DateTimeOriginal + SubSecTimeOriginal to epoch seconds."""
    dt_str = exif.get("DateTimeOriginal", "1970:01:01 00:00:00")
    subsec = str(exif.get("SubSecTimeOriginal", "00"))
    try:
        dt = datetime.strptime(dt_str, "%Y:%m:%d %H:%M:%S")
        # Canon writes SubSecTimeOriginal as fixed-width decimal
        # fraction: "050" = 50/100 = 0.50 seconds, NOT 0.050.
        # Divide by 10^(number of digits) to get fractional seconds.
        subsec_val = int(subsec) / (10 ** len(subsec))
        return dt.timestamp() + subsec_val
    except (ValueError, TypeError):
        return 0.0


def ingest_and_cluster(cr3_paths: list[str]) -> tuple[list[Frame], list[str]]:
    """
    Read EXIF from all image files, sort chronologically, detect
    burst/scene boundaries, construct Frame objects.

    Uses exiftool's JSON output for batch EXIF extraction (one
    subprocess for all files, not one per file). Works with any
    format exiftool supports (RAW, JPEG, TIFF).

    Returns (frames, sorted_paths) where sorted_paths is the
    chronologically-sorted input for downstream stages.
    """
    logger.info(f"Stage 0: Ingesting {len(cr3_paths)} images")

    # Batch EXIF extraction via -@ ARGFILE (avoids ARG_MAX limits
    # with 5000+ images or deeply nested directory paths)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False
    ) as argfile:
        for p in cr3_paths:
            argfile.write(p + "\n")
        argfile_path = argfile.name

    try:
        result = subprocess.run(
            [
                "exiftool",
                "-json",
                "-DateTimeOriginal",
                "-SubSecTimeOriginal",
                "-FNumber",
                "-ExposureTime",
                "-ISO",
                "-FocalLength",
                "-Orientation",
                "-CameraModelName",
                "-@",
                argfile_path,
            ],
            capture_output=True,
            text=True,
            check=True,
        )
    finally:
        os.unlink(argfile_path)

    exif_list = json.loads(result.stdout)

    # Pair paths with EXIF and sort by timestamp
    path_exif = list(zip(cr3_paths, exif_list))
    path_exif.sort(key=lambda pe: _parse_timestamp(pe[1]))

    # Classify boundaries between consecutive frames
    scene_id = 0
    burst_id = 0
    burst_start = 0
    frames = []

    for i, (path, exif) in enumerate(path_exif):
        exif["_timestamp"] = _parse_timestamp(exif)

        if i > 0:
            prev_exif = path_exif[i - 1][1]

            fi = SimpleNamespace(timestamp=prev_exif["_timestamp"], exif=prev_exif)
            fj = SimpleNamespace(timestamp=exif["_timestamp"], exif=exif)

            features = compute_boundary_features(fi, fj)
            burst_break, scene_break = classify_boundary(features)

            if scene_break:
                scene_id += 1
                burst_id += 1
                burst_start = i
            elif burst_break:
                burst_id += 1
                burst_start = i

        frames.append(
            Frame(
                path=path,
                global_index=i,
                scene_id=scene_id,
                burst_id=burst_id,
                frame_index=i - burst_start,
                burst_length=0,  # set in second pass
                camera_body=exif.get("CameraModelName", "unknown"),
                exif=exif,
                timestamp=exif["_timestamp"],
            )
        )

    # Second pass: set burst_length for all frames in each burst
    burst_sizes = Counter(f.burst_id for f in frames)
    for f in frames:
        f.burst_length = burst_sizes[f.burst_id]

    sorted_paths = [p for p, _ in path_exif]

    n_scenes = len(set(f.scene_id for f in frames))
    n_bursts = len(set(f.burst_id for f in frames))
    logger.info(
        f"Stage 0 complete: {len(frames)} frames, "
        f"{n_scenes} scenes, {n_bursts} bursts"
    )

    return frames, sorted_paths