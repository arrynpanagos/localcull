"""
Stage 6: Output & Active Learning.

Three independent layers per image:
  1. Star ratings (1-5) from z-score consensus percentiles (per-shoot adaptive).
  2. Color labels: Red=agreed keeper, Purple=disagreement pair, Green=tech gate fail.
  3. Custom XMP namespace for all scorer outputs + consensus/PCA axes.

XMP sidecar writing via pyexiftool stay-open protocol.
Per-image feature dump CSV for debugging.
"""

import csv
import logging
import os
from collections import Counter

import numpy as np
from tqdm import tqdm

from localcull.constants import CHECKPOINT_DIR
from localcull.types import Frame

logger = logging.getLogger(__name__)

# Exiftool config path (shipped alongside this module)
_DEFAULT_CONFIG = os.path.join(os.path.dirname(__file__), "localcull_exiftool.config")

# ── Star rating: equal-interval binning ──────────────────────────
# Linearly maps the z-consensus score range to 1-5 stars.
# No forced distribution — if all images score similarly,
# they all get similar ratings. The full [min, max] range
# is divided into 5 equal bins.
#
# This is a relative measure within the shoot. A 4★ in one
# shoot is not directly comparable to a 4★ in another.


# ══════════════════════════════════════════════════════════════════
# Star ratings for ALL images
# ══════════════════════════════════════════════════════════════════


def rate_all_images(
    composite_scores: np.ndarray,
    frames: list[Frame],
    n_bootstrap: int = 1000,
    rng_seed: int = 42,
    **kwargs,
) -> dict[int, int]:
    """
    Assign 1-5 star ratings using bootstrap-estimated empirical quantiles.

    No distributional assumptions. The thresholds are derived from the
    actual shape of the score distribution via bootstrap resampling,
    targeting quantiles that correspond to ±1σ/±2σ boundaries for
    normal data but adapt naturally to skew, heavy tails, or outliers.

    Target quantiles (equivalent to σ-boundaries for normal data):
        p=0.0228  (≈ -2σ)  →  1★/2★ boundary
        p=0.1587  (≈ -1σ)  →  2★/3★ boundary
        p=0.5000  (median)  →  3★/4★ boundary
        p=0.8413  (≈ +1σ)  →  4★/5★ boundary

    For symmetric data this yields ~2/14/34/34/16% distribution.
    For left-skewed data (few terrible shots, most shots good),
    the thresholds shift upward to where the data actually lives,
    ensuring 5★ remains reachable and meaningful.

    Bootstrap resampling (B=1000) stabilizes quantile estimates,
    especially for small shoots or distributions with gaps.

    Stars are always relative to the shoot.
    """
    valid_scores = []
    valid_indices = []
    for frame in frames:
        score = composite_scores[frame.global_index]
        if np.isfinite(score):
            valid_scores.append(score)
            valid_indices.append(frame.global_index)

    valid_scores = np.array(valid_scores)
    n = len(valid_scores)

    if n == 0:
        return {f.global_index: 1 for f in frames}

    if valid_scores.max() - valid_scores.min() < 1e-8:
        # All scores identical — give everything 3★
        return {f.global_index: 3 for f in frames}

    # Target quantiles: correspond to ±2σ, ±1σ, median for normal data
    target_quantiles = np.array([0.0228, 0.1587, 0.5000, 0.8413])

    # Bootstrap: resample and compute quantiles B times
    rng = np.random.default_rng(rng_seed)
    bootstrap_quantiles = np.zeros((n_bootstrap, len(target_quantiles)))

    for b in range(n_bootstrap):
        resample = rng.choice(valid_scores, size=n, replace=True)
        bootstrap_quantiles[b] = np.quantile(resample, target_quantiles)

    # Stable threshold estimates: mean of bootstrap quantiles
    thresholds = bootstrap_quantiles.mean(axis=0)

    # Log diagnostics
    threshold_std = bootstrap_quantiles.std(axis=0)
    logger.debug(
        f"Bootstrap thresholds (B={n_bootstrap}): "
        f"1★/2★={thresholds[0]:.3f}±{threshold_std[0]:.3f}, "
        f"2★/3★={thresholds[1]:.3f}±{threshold_std[1]:.3f}, "
        f"3★/4★={thresholds[2]:.3f}±{threshold_std[2]:.3f}, "
        f"4★/5★={thresholds[3]:.3f}±{threshold_std[3]:.3f}"
    )

    # Assign ratings
    ratings = {}
    for frame in frames:
        score = composite_scores[frame.global_index]

        if not np.isfinite(score):
            ratings[frame.global_index] = 1
            continue

        if score >= thresholds[3]:
            ratings[frame.global_index] = 5
        elif score >= thresholds[2]:
            ratings[frame.global_index] = 4
        elif score >= thresholds[1]:
            ratings[frame.global_index] = 3
        elif score >= thresholds[0]:
            ratings[frame.global_index] = 2
        else:
            ratings[frame.global_index] = 1

    dist = Counter(ratings.values())
    logger.info(
        f"Star distribution (all {len(ratings)} images, "
        f"bootstrap quantiles, B={n_bootstrap}): "
        + ", ".join(f"{k}★={v}" for k, v in sorted(dist.items()))
    )

    return ratings


# ══════════════════════════════════════════════════════════════════
# Scorer field resolution
# ══════════════════════════════════════════════════════════════════


# Mapping from Frame field → XMP tag name
_SCORER_FIELD_TO_XMP = {
    "qalign_score": "QAlignScore",
    "qualiclip_score": "QualiCLIPScore",
    "topiq_score": "TOPIQScore",
    "musiq_score": "MUSIQScore",
    "deqa_score": "DeQAScore",
    "q_scorer_score": "QScorerScore",
    "q_insight_score": "QInsightScore",
    "nima_score": "NIMAScore",
    "artimuse_score": "ArtiMuseScore",
    "unipercept_score": "UniPerceptScore",
}

# Consensus fields always written to XMP
_CONSENSUS_FIELDS = {
    "z_consensus": "ZConsensus",
    "z_disagreement": "ZDisagreement",
    "pc2_aesthetic_vs_technical": "PC2AestheticVsTechnical",
    "pc3_structural": "PC3Structural",
}


def _get_active_scorer_fields(
    frames: list[Frame],
) -> list[tuple[str, str]]:
    """
    Determine which scorer fields have non-zero values (were actually run).

    Returns list of (frame_field, xmp_key) tuples.
    """
    active = []
    for field, xmp_key in _SCORER_FIELD_TO_XMP.items():
        # Check if any frame has a non-zero value for this field
        if any(getattr(f, field, 0.0) != 0.0 for f in frames):
            active.append((field, xmp_key))
    return active


# ══════════════════════════════════════════════════════════════════
# XMP sidecar writing
# ══════════════════════════════════════════════════════════════════


def write_xmp_sidecars(
    frames: list[Frame],
    ratings: dict[int, int],
    selected_indices: list[int],
    composite_scores: np.ndarray,
    config_path: str = _DEFAULT_CONFIG,
):
    """
    Write XMP sidecar files with star ratings and color labels.

    Every image gets a star rating (1-5) from the primary scorer.
    Color labels:
      Red    = selected keeper (models agreed on this cluster's best)
      Purple = selected keeper (models disagreed — review this pair)
      Green  = technical gate failure (sharpness, blink, noise)

    Selected images get Red or Purple. Non-selected tech failures get Green.
    Custom localcull namespace stores all scorer outputs for comparison.
    """
    import exiftool

    selected_set = set(selected_indices)

    logger.info(f"Writing {len(ratings)} XMP sidecars")

    n_red = 0
    n_green = 0
    n_purple = 0

    # Build list of scorer fields that have non-zero values
    # (only write scorers that were actually run)
    scorer_fields = _get_active_scorer_fields(frames)

    with exiftool.ExifToolHelper(
        common_args=["-config", config_path]
    ) as et:
        for frame in tqdm(frames, desc="Writing XMP sidecars"):
            if frame.global_index not in ratings:
                continue

            xmp_path = frame.path.rsplit(".", 1)[0] + ".xmp"
            rating = ratings[frame.global_index]
            is_selected = frame.global_index in selected_set

            if not os.path.exists(xmp_path):
                _write_minimal_xmp(xmp_path, frame.path)

            tags = {
                "XMP:Rating": rating,
                "XMP-localcull:TechGatePass": "True" if frame.technical_gate_pass else "False",
            }

            # Write all active scorer outputs to XMP
            for field, xmp_key in scorer_fields:
                val = getattr(frame, field, 0.0)
                if val != 0.0:
                    tags[f"XMP-localcull:{xmp_key}"] = f"{val:.6f}"

            # Write consensus/PCA fields
            for field, xmp_key in _CONSENSUS_FIELDS.items():
                val = getattr(frame, field, 0.0)
                if val != 0.0:
                    tags[f"XMP-localcull:{xmp_key}"] = f"{val:.6f}"

            # Color label: selected images get Red or Purple
            if is_selected and frame.aesthetic_disagreement:
                tags["XMP:Label"] = "Purple"
                n_purple += 1
            elif is_selected:
                tags["XMP:Label"] = "Red"
                n_red += 1
            elif not frame.technical_gate_pass:
                tags["XMP:Label"] = "Green"
                n_green += 1
            else:
                # Clear stale labels from previous runs
                tags["XMP:Label"] = ""

            et.set_tags(
                xmp_path,
                tags,
                params=["-overwrite_original"],
            )

    logger.info(
        f"XMP sidecars written: {len(ratings)} rated, "
        f"{n_red} Red (best per cluster), {n_purple} Purple (disagreement alternative), "
        f"{n_green} Green (tech fail)"
    )


def _write_minimal_xmp(xmp_path: str, source_path: str):
    """Create a minimal XMP sidecar that Lightroom will recognize."""
    basename = os.path.basename(source_path)
    xmp = f"""<?xpacket begin='\xef\xbb\xbf' id='W5M0MpCehiHzreSzNTczkc9d'?>
<x:xmpmeta xmlns:x="adobe:ns:meta/">
  <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
    <rdf:Description rdf:about=""
      xmlns:xmp="http://ns.adobe.com/xap/1.0/"
      xmlns:xmpMM="http://ns.adobe.com/xap/1.0/mm/"
      xmpMM:OriginalDocumentID="{basename}"
      xmp:Rating="0"/>
  </rdf:RDF>
</x:xmpmeta>
<?xpacket end='w'?>"""
    with open(xmp_path, "w", encoding="utf-8") as f:
        f.write(xmp)


# ══════════════════════════════════════════════════════════════════
# Feature dump
# ══════════════════════════════════════════════════════════════════


def write_feature_dump(
    frames: list[Frame],
    ratings: dict[int, int],
    selected_indices: list[int],
    composite_scores: np.ndarray,
    shoot_id: str,
):
    """
    Emit per-image feature vector CSV for ALL images.
    Enables "why did this image get N stars?" analysis without re-running.
    """
    selected_set = set(selected_indices)
    output_dir = os.path.dirname(frames[0].path)
    output_path = os.path.join(output_dir, f"{shoot_id}_features.csv")

    try:
        with open(output_path, "w", newline="") as f:
            pass
    except OSError:
        output_path = os.path.join(CHECKPOINT_DIR, f"{shoot_id}_features.csv")
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        logger.warning(
            f"CR3 directory not writable, writing feature dump to {output_path}"
        )

    fieldnames = [
        "filename",
        "global_index",
        "scene_id",
        "burst_id",
        "burst_length",
        "camera_body",
        "star_rating",
        "dpp_selected",
        "technical_gate_pass",
        "aesthetic_disagreement",
        "cluster_id",
        "cluster_agreement",
        "visual_category",
        "composite_relevance",
        "z_consensus",
        "z_disagreement",
        "pc2_aesthetic_vs_technical",
        "pc3_structural",
        # All scorer outputs (columns present even if scorer wasn't run)
        "qalign_score",
        "qualiclip_score",
        "topiq_score",
        "musiq_score",
        "deqa_score",
        "q_scorer_score",
        "q_insight_score",
        "nima_score",
        "artimuse_score",
        "unipercept_score",
        # Sharpness / face features
        "sharp_near_eye",
        "sharp_far_eye",
        "eye_ratio_raw",
        "sharpness_subject",
        "sharpness_background",
        "isolation_ratio",
        "has_face",
        "n_faces",
        "subject_det_method",
        "rendering_path",
        "saliency_confidence",
        "blink_detected",
        "raw_min_ear",
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for frame in frames:
            writer.writerow(
                {
                    "filename": os.path.basename(frame.path),
                    "global_index": frame.global_index,
                    "scene_id": frame.scene_id,
                    "burst_id": frame.burst_id,
                    "burst_length": frame.burst_length,
                    "camera_body": frame.camera_body,
                    "star_rating": ratings.get(frame.global_index, 0),
                    "dpp_selected": frame.global_index in selected_set,
                    "technical_gate_pass": frame.technical_gate_pass,
                    "aesthetic_disagreement": frame.aesthetic_disagreement,
                    "cluster_id": frame.cluster_id,
                    "cluster_agreement": frame.cluster_agreement,
                    "visual_category": frame.visual_category,
                    "composite_relevance": f"{frame.composite_relevance:.6f}",
                    "z_consensus": f"{frame.z_consensus:.6f}",
                    "z_disagreement": f"{frame.z_disagreement:.6f}",
                    "pc2_aesthetic_vs_technical": f"{frame.pc2_aesthetic_vs_technical:.6f}",
                    "pc3_structural": f"{frame.pc3_structural:.6f}",
                    "qalign_score": f"{frame.qalign_score:.6f}",
                    "qualiclip_score": f"{frame.qualiclip_score:.6f}",
                    "topiq_score": f"{frame.topiq_score:.6f}",
                    "musiq_score": f"{frame.musiq_score:.6f}",
                    "deqa_score": f"{frame.deqa_score:.6f}",
                    "q_scorer_score": f"{frame.q_scorer_score:.6f}",
                    "q_insight_score": f"{frame.q_insight_score:.6f}",
                    "nima_score": f"{frame.nima_score:.6f}",
                    "artimuse_score": f"{frame.artimuse_score:.6f}",
                    "unipercept_score": f"{frame.unipercept_score:.6f}",
                    "sharp_near_eye": f"{frame.sharp_near_eye:.4f}",
                    "sharp_far_eye": f"{frame.sharp_far_eye:.4f}",
                    "eye_ratio_raw": f"{frame.eye_ratio_raw:.4f}",
                    "sharpness_subject": f"{frame.sharpness_subject:.4f}",
                    "sharpness_background": f"{frame.sharpness_background:.4f}",
                    "isolation_ratio": f"{frame.isolation_ratio:.4f}",
                    "has_face": frame.has_face,
                    "n_faces": frame.n_faces,
                    "subject_det_method": frame.subject_det_method,
                    "rendering_path": frame.rendering_path,
                    "saliency_confidence": f"{frame.saliency_confidence:.4f}",
                    "blink_detected": frame.blink_detected,
                    "raw_min_ear": f"{frame.raw_min_ear:.4f}",
                }
            )

    logger.info(f"Feature dump: {output_path} ({len(frames)} images)")


# ══════════════════════════════════════════════════════════════════
# Ranked symlink folders
# ══════════════════════════════════════════════════════════════════


def write_ranked_folders(
    frames: list[Frame],
    ratings: dict[int, int],
    selected_indices: list[int],
    shoot_id: str,
):
    """
    Create ranked symlink folders for browsing in Finder/file manager.

    One folder per active scorer:
      {shoot_id}_ranked_{scorer_name}/

    Symlinks named: 001_★★★★★_R_filename.ext
      - Rank prefix for sort order
      - Star rating for quick visual scan
      - 'R' suffix if selected, models agreed (Red label)
      - 'G' suffix if tech gate fail (Green label)
      - 'P' suffix if selected, models disagreed (Purple label)
    """
    from pathlib import Path

    output_dir = os.path.dirname(frames[0].path)
    selected_set = set(selected_indices)

    # Build list of (folder_suffix, frame_field) for active scorers
    active_scorer_fields = _get_active_scorer_fields(frames)
    sort_configs = []

    # Consensus ranking first (primary browse folder)
    sort_configs.append(("consensus", "z_consensus"))

    for field, xmp_key in active_scorer_fields:
        # e.g. "qalign_score" → "qalign", "deqa_score" → "deqa"
        suffix = field.replace("_score", "")
        sort_configs.append((suffix, field))

    folder_names = []
    for suffix, field in sort_configs:
        folder_name = f"{shoot_id}_ranked_{suffix}"
        sorted_frames = sorted(
            frames, key=lambda f: -getattr(f, field, 0.0)
        )

        folder = Path(output_dir) / folder_name
        folder.mkdir(exist_ok=True)

        # Clear old symlinks
        for existing in folder.iterdir():
            if existing.is_symlink() or existing.is_file():
                existing.unlink()

        for rank, frame in enumerate(sorted_frames, 1):
            star = ratings.get(frame.global_index, 0)
            star_str = "\u2605" * star + "\u2606" * (5 - star)

            # Flag suffix
            is_selected = frame.global_index in selected_set
            if is_selected and frame.aesthetic_disagreement:
                flag = "_P"  # Purple: disagreement pick
            elif is_selected:
                flag = "_R"  # Red: agreement pick
            elif not frame.technical_gate_pass:
                flag = "_G"  # Green: tech failure
            else:
                flag = ""

            src = Path(frame.path).resolve()
            basename = src.name

            # Prefer embedded JPEG for macOS preview compatibility
            embedded = src.with_name(src.stem + "_embedded.jpg")
            if embedded.exists():
                src = embedded
                basename = src.name

            link_name = f"{rank:03d}_{star_str}{flag}_{basename}"
            link_path = folder / link_name

            try:
                link_path.symlink_to(src)
            except OSError:
                pass  # skip if symlink fails (permissions, etc.)

        folder_names.append(folder_name)

    # ── Visual category folders (DINOv2-based) ──
    # Group images by detected visual category for easy validation
    categories_found = set(f.visual_category for f in frames if f.visual_category >= 0)
    if categories_found:
        for cat_id in sorted(categories_found):
            cat_frames = [f for f in frames if f.visual_category == cat_id]
            cat_folder_name = f"{shoot_id}_category_{cat_id}"
            cat_folder = Path(output_dir) / cat_folder_name
            cat_folder.mkdir(exist_ok=True)

            # Clear old symlinks
            for existing in cat_folder.iterdir():
                if existing.is_symlink() or existing.is_file():
                    existing.unlink()

            # Sort by consensus within each category
            cat_frames.sort(key=lambda f: -getattr(f, "z_consensus", 0.0))

            for rank, frame in enumerate(cat_frames, 1):
                star = ratings.get(frame.global_index, 0)
                star_str = "\u2605" * star + "\u2606" * (5 - star)

                is_selected = frame.global_index in selected_set
                if is_selected and frame.aesthetic_disagreement:
                    flag = "_P"
                elif is_selected:
                    flag = "_R"
                elif not frame.technical_gate_pass:
                    flag = "_G"
                else:
                    flag = ""

                src = Path(frame.path).resolve()
                embedded = src.with_name(src.stem + "_embedded.jpg")
                if embedded.exists():
                    src = embedded

                link_name = f"{rank:03d}_{star_str}{flag}_{src.name}"
                link_path = cat_folder / link_name

                try:
                    link_path.symlink_to(src)
                except OSError:
                    pass

            folder_names.append(cat_folder_name)

        logger.info(
            f"Category folders: {len(categories_found)} categories "
            f"({', '.join(f'cat {c}: {sum(1 for f in frames if f.visual_category == c)} images' for c in sorted(categories_found))})"
        )

    # ── Red picks folder: all best-per-cluster selections (Red label) ──
    red_frames = [
        f for f in frames
        if f.global_index in selected_set and not f.aesthetic_disagreement
    ]
    # Sort by z_consensus (overall quality rank)
    red_frames.sort(key=lambda f: -getattr(f, "z_consensus", 0.0))

    red_folder_name = f"{shoot_id}_red_picks"
    red_folder = Path(output_dir) / red_folder_name
    red_folder.mkdir(exist_ok=True)

    for existing in red_folder.iterdir():
        if existing.is_symlink() or existing.is_file():
            existing.unlink()

    for rank, frame in enumerate(red_frames, 1):
        star = ratings.get(frame.global_index, 0)
        star_str = "\u2605" * star + "\u2606" * (5 - star)

        src = Path(frame.path).resolve()
        embedded = src.with_name(src.stem + "_embedded.jpg")
        if embedded.exists():
            src = embedded

        link_name = f"{rank:03d}_{star_str}_R_{src.name}"
        link_path = red_folder / link_name

        try:
            link_path.symlink_to(src)
        except OSError:
            pass

    folder_names.append(red_folder_name)
    logger.info(
        f"Red picks: {len(red_frames)} images "
        f"(best per cluster, consensus-ranked)"
    )

    logger.info(
        f"Ranked folders: {', '.join(folder_names)}"
    )