"""
Top-level pipeline orchestration.

Coordinates all stages: ingest → features → scoring → selection → output.

Fast re-run: after first execution, stages 0-2 results are checkpointed
as a single "stage2_complete" pickle (~5-10MB). Subsequent runs with the
same images skip straight to stage 3 (~1 second instead of ~2 minutes).
"""

import logging

from localcull.checkpoint import compute_data_hash, load_checkpoint, save_checkpoint
from localcull.memory import cleanup_orphaned_shm
from localcull.stage0_ingest import ingest_and_cluster
from localcull.stage1_prepare import prepare_images
from localcull.stage2_features import run_stage2
from localcull.stage3_scoring import (
    build_composite_scores_array,
    run_composite_scoring,
)
from localcull.stage4_selection import run_stage4
from localcull.stage6_output import (
    rate_all_images,
    write_feature_dump,
    write_ranked_folders,
    write_xmp_sidecars,
)
from localcull.types import Frame

logger = logging.getLogger(__name__)


def _run_missing_scorers(
    frames: list[Frame],
    embeddings,
    degraded_indices,
    shoot_id: str,
    path_hash: str,
    enabled_scorers: list[str] | None = None,
) -> tuple:
    """
    After fast-resume from stage2_complete, check if any enabled scorers
    haven't been run yet. If so, rebuild mid_arrays from cached embedded
    JPEGs and run only the missing scorers.

    Returns updated (frames, embeddings, degraded_indices).
    """
    import gc
    import os

    import numpy as np
    from PIL import Image

    # Suppress noisy PIL debug logging during image rebuild
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("PIL.TiffImagePlugin").setLevel(logging.WARNING)

    from localcull.checkpoint import save_checkpoint
    from localcull.scorers import get_enabled_scorers
    from localcull.stage2_features import _run_gpu_model, _save_mid_arrays_to_disk

    specs = get_enabled_scorers(enabled_scorers)

    # Find scorers with all-zero scores (never ran)
    missing = []
    for spec in specs:
        if not spec.frame_field:
            continue
        # Check if all frames have 0.0 for this field
        all_zero = all(
            getattr(f, spec.frame_field, 0.0) == 0.0 for f in frames
        )
        if all_zero:
            missing.append(spec)

    if not missing:
        return frames, embeddings, degraded_indices

    missing_names = [s.display_name for s in missing]
    logger.info(
        f"Found {len(missing)} new scorer(s) to run: "
        f"{', '.join(missing_names)}"
    )

    # ── Rebuild mid_arrays from cached embedded JPEGs ──
    logger.info("Rebuilding mid-res arrays from embedded JPEG cache...")
    mid_arrays = []
    for f in frames:
        jpeg_path = f.embedded_jpeg_path or f.rendering_path or f.path
        try:
            img = Image.open(jpeg_path).convert("RGB")
            mid_arrays.append(np.array(img))
            del img
        except Exception as e:
            logger.warning(f"Failed to load {jpeg_path}: {e}")
            mid_arrays.append(np.zeros((512, 512, 3), dtype=np.uint8))

    mid_path = _save_mid_arrays_to_disk(mid_arrays, shoot_id)
    del mid_arrays
    gc.collect()

    # ── Run only missing scorers ──
    skipped = []
    for spec in missing:
        try:
            scores = _run_gpu_model(
                spec.name, mid_path, shoot_id, path_hash
            )
            # Apply to frames
            for i, f in enumerate(frames):
                setattr(f, spec.frame_field, float(scores[i]))
            logger.info(f"Incremental scorer {spec.display_name} complete")
        except Exception as e:
            logger.warning(
                f"Scorer {spec.display_name} failed, skipping: {e}"
            )
            skipped.append(spec.name)

    # Clean up temp file
    try:
        os.remove(mid_path)
    except OSError:
        pass

    if skipped:
        logger.warning(
            f"Skipped {len(skipped)} scorer(s): {', '.join(skipped)}. "
            f"Pipeline continues with remaining scorers."
        )

    # Re-save stage2_complete with updated scorer data
    if len(skipped) < len(missing):
        logger.info("Re-saving stage2_complete checkpoint with new scorer data")
        save_checkpoint(
            shoot_id, "stage2_complete",
            (frames, embeddings, degraded_indices),
            path_hash,
        )

    gc.collect()
    return frames, embeddings, degraded_indices


def _compute_path_hash(image_paths: list[str]) -> str:
    """Hash from sorted paths — changes if files added/removed."""
    sorted_paths = sorted(image_paths)
    return compute_data_hash(sorted_paths)


def run_pipeline(
    image_paths: list[str],
    shoot_id: str,
    pu_head=None,
    target_total: int | None = None,
    enabled_scorers: list[str] | None = None,
) -> tuple[list[Frame], dict[int, int], list[int]]:
    """
    Complete pipeline: ingest → features → scoring → selection → output.

    Args:
        image_paths: list of absolute paths to image files (RAW or JPEG)
        shoot_id: unique identifier for checkpoint caching
        pu_head: optional PUPersonalizationHead for active learning
        target_total: approximate number of images to select. If None,
            defaults to ~3-5% of input.
        enabled_scorers: list of scorer names to run, or None for defaults.
            Special value ["all"] runs everything in the registry.

    Returns:
        (frames, ratings, selected_indices)
    """
    logger.info(
        f"localcull pipeline starting: {len(image_paths)} images, "
        f"shoot_id={shoot_id}"
    )

    # Clean up orphaned shared memory from crashed previous runs
    cleanup_orphaned_shm()

    # ── Fast path: try loading stage2_complete checkpoint ──
    path_hash = _compute_path_hash(image_paths)
    cached = load_checkpoint(shoot_id, "stage2_complete", path_hash)

    if cached is not None:
        frames, embeddings, degraded_indices = cached
        logger.info(
            f"Fast resume: {len(frames)} frames loaded from checkpoint, "
            f"skipping stages 0-2"
        )

        # ── Check for missing scorers that weren't in the original run ──
        frames, embeddings, degraded_indices = _run_missing_scorers(
            frames, embeddings, degraded_indices,
            shoot_id, path_hash, enabled_scorers,
        )
    else:
        # ── Stage 0: EXIF + temporal clustering ──
        frames, sorted_paths = ingest_and_cluster(image_paths)
        data_hash = compute_data_hash(sorted_paths)

        # ── Stage 1: Image preparation + memory layout ──
        mid_arrays, full_store, jpeg_paths, degraded_indices, rendering_paths = (
            prepare_images(sorted_paths)
        )

        # Catch length mismatches before 10 min of GPU inference
        assert len(mid_arrays) == len(frames) == len(jpeg_paths), (
            f"Stage 0/1 length mismatch: {len(frames)} frames, "
            f"{len(mid_arrays)} mid arrays, {len(jpeg_paths)} JPEG paths"
        )

        try:
            for i, f in enumerate(frames):
                f.mid_array = mid_arrays[i]
                f.full_shm_index = i
                f.embedded_jpeg_path = jpeg_paths[i]
                f.rendering_path = rendering_paths[i]

            # ── Stage 2: Feature extraction (GPU + CPU concurrent) ──
            embeddings = run_stage2(
                frames, mid_arrays, full_store, shoot_id, data_hash,
                enabled_scorers=enabled_scorers,
            )
        finally:
            full_store.cleanup()

        # Post-Stage 2 cleanup: free image data before checkpointing
        del mid_arrays
        for f in frames:
            f.mid_array = None
            f.saliency_map = None       # [37,37] numpy, not needed for stages 3-6
            f.dinov2_embedding = None    # duplicated in embeddings array

        # Save checkpoint for fast re-runs (frames + embeddings + degraded)
        save_checkpoint(
            shoot_id, "stage2_complete",
            (frames, embeddings, degraded_indices),
            path_hash,
        )

    # ── Visual category detection (diagnostic) ──
    from localcull.stage3_scoring import detect_visual_categories
    detect_visual_categories(frames, embeddings)

    # ── Stage 3: Composite scoring ──
    run_composite_scoring(
        frames, pu_head, degraded_indices=degraded_indices
    )

    # ── Stage 4: Cluster → Rank → Pick selection ──
    selected_frames, selected_indices, quality_scores = run_stage4(
        frames, embeddings, target_total=target_total
    )

    # ── Stage 5: Output ──
    composite_scores = build_composite_scores_array(frames)
    ratings = rate_all_images(composite_scores, frames)
    write_xmp_sidecars(frames, ratings, selected_indices, composite_scores)
    write_feature_dump(frames, ratings, selected_indices, composite_scores, shoot_id)
    write_ranked_folders(frames, ratings, selected_indices, shoot_id)

    n_total = len(frames)
    n_selected = len(selected_indices)
    logger.info(
        f"Pipeline complete: {n_total} rated, {n_selected} selected "
        f"({n_selected/n_total:.1%})"
    )

    return frames, ratings, selected_indices