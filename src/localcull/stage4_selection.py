"""
Stage 4: Cluster → Rank → Pick Selection.

Simpler and more robust than DPP:
  1. Group by scene (from stage 0 temporal clustering).
  2. Within each scene, agglomerative clustering on DINOv2 embeddings
     groups visually similar compositions together.
  3. Within each cluster, rank by z-consensus (composite_relevance)
     and by disagreement scorer (QualiCLIP+).
  4. Pick the top-1 by each:
     - If same image: 1 pick (models agree). Red label.
     - If different: 2 picks (both perspectives represented). Purple label.
  5. Only images passing the technical gate are eligible.

Primary ranking uses z-score consensus across all active scorers.
Disagreement scorer = QualiCLIP+ (most orthogonal to consensus, r=0.29).

Diversity comes from clustering (one pick per visual group).
Quality comes from picking the best within each group.
Framework: NumPy/SciPy (CPU, <1s).
"""

import logging
from collections import defaultdict

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist

from localcull.constants import (
    DISAGREEMENT_SCORER,
    MIN_SELECTION_ABSOLUTE,
    MIN_SELECTION_FRACTION,
    PRIMARY_SCORER,
)
from localcull.types import Frame

logger = logging.getLogger(__name__)

# ── Clustering parameters ──
# Cosine distance threshold for grouping "same composition" images.
# 0.15 = cosine similarity > 0.85 groups together.
CLUSTER_DISTANCE_THRESHOLD = 0.15


# ══════════════════════════════════════════════════════════════════
# Clustering
# ══════════════════════════════════════════════════════════════════


def cluster_scene(
    scene_frames: list[Frame],
    embeddings: np.ndarray,
    distance_threshold: float = CLUSTER_DISTANCE_THRESHOLD,
) -> dict[int, list[Frame]]:
    """
    Agglomerative clustering within a scene using cosine distance.

    Returns {cluster_id: [frames]} where each cluster represents
    a visually similar group of compositions.
    """
    if len(scene_frames) <= 1:
        return {0: scene_frames}

    emb = np.stack([embeddings[f.global_index] for f in scene_frames])

    # Normalize for cosine distance
    norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8
    emb_normed = emb / norms

    # Pairwise cosine distances
    dists = pdist(emb_normed, metric="cosine")

    # Handle degenerate case: all identical embeddings
    if np.max(dists) < 1e-8:
        return {0: scene_frames}

    # Agglomerative clustering with distance threshold
    Z = linkage(dists, method="average")
    labels = fcluster(Z, t=distance_threshold, criterion="distance")

    clusters = defaultdict(list)
    for frame, label in zip(scene_frames, labels):
        clusters[int(label)].append(frame)

    return dict(clusters)


# ══════════════════════════════════════════════════════════════════
# Per-cluster selection
# ══════════════════════════════════════════════════════════════════


def pick_from_cluster(
    cluster_frames: list[Frame],
    primary_field: str = "composite_relevance",
    disagree_field: str = "qualiclip_score",
) -> tuple[list[Frame], bool]:
    """
    From a cluster of visually similar images, pick the best.

    1. Filter to tech-gate-passing frames only.
    2. Top-1 by primary scorer (consensus) → always Red.
    3. Top-1 by disagreement scorer (QualiCLIP+) → Purple if different.
    4. If same image → 1 pick, agreement. Red label.
       If different → 2 picks. Primary=Red, alternative=Purple.

    Every cluster always has exactly one Red pick.

    Returns (picks, is_disagreement).
    """
    eligible = [f for f in cluster_frames if f.technical_gate_pass]

    if not eligible:
        # All failed tech gate — pick the single best by primary anyway
        best = max(cluster_frames, key=lambda f: getattr(f, primary_field, 0.0))
        return [best], False

    if len(eligible) == 1:
        return eligible, False

    # Top-1 by each model
    best_primary = max(eligible, key=lambda f: getattr(f, primary_field, 0.0))
    best_disagree = max(eligible, key=lambda f: getattr(f, disagree_field, 0.0))

    if best_primary.global_index == best_disagree.global_index:
        # Models agree — single pick
        return [best_primary], False
    else:
        # Models disagree — primary stays Red, only alternative gets Purple
        return [best_primary, best_disagree], True


# ══════════════════════════════════════════════════════════════════
# Stage 4 orchestration
# ══════════════════════════════════════════════════════════════════


def run_stage4(
    frames: list[Frame],
    embeddings: np.ndarray,
    target_total: int | None = None,
    primary_scorer: str = PRIMARY_SCORER,
    disagreement_scorer: str = DISAGREEMENT_SCORER,
) -> tuple[list[Frame], list[int], list[float]]:
    """
    Orchestrate selection: Cluster → Rank → Pick.

    1. Group frames by scene.
    2. Within each scene, cluster by DINOv2 embedding similarity.
    3. Within each cluster, pick top by primary scorer and top by
       disagreement scorer.
    4. Enforce minimum selection floor.

    Returns (selected_frames, selected_global_indices, quality_scores).
    quality_scores are composite_relevance values (used by Stage 5 VLM pairing).
    """
    from localcull.scorers import SCORER_REGISTRY

    # Resolve frame field names for each scorer role
    primary_field = "composite_relevance"  # always use composite (set by stage3)
    disagree_field = "qualiclip_score"  # default fallback
    if disagreement_scorer in SCORER_REGISTRY:
        disagree_field = SCORER_REGISTRY[disagreement_scorer].frame_field

    # Display names for logging
    primary_display = primary_scorer
    disagree_display = disagreement_scorer
    if primary_scorer in SCORER_REGISTRY:
        primary_display = SCORER_REGISTRY[primary_scorer].display_name
    if disagreement_scorer in SCORER_REGISTRY:
        disagree_display = SCORER_REGISTRY[disagreement_scorer].display_name

    logger.info(
        f"Stage 4: Cluster → Rank → Pick "
        f"(primary={primary_display}, disagree={disagree_display})"
    )

    # Group by scene
    scenes = defaultdict(list)
    for f in frames:
        scenes[f.scene_id].append(f)

    all_selected = []
    total_clusters = 0
    global_cluster_id = 0
    n_agree = 0
    n_disagree = 0

    for scene_id in sorted(scenes.keys()):
        scene_frames = scenes[scene_id]

        # Cluster within scene
        clusters = cluster_scene(scene_frames, embeddings)
        n_clusters = len(clusters)
        total_clusters += n_clusters

        scene_selected = []
        for cluster_id, cluster_frames in sorted(clusters.items()):
            # Assign globally unique cluster_id to all frames in this cluster
            for f in cluster_frames:
                f.cluster_id = global_cluster_id

            picks, is_disagreement = pick_from_cluster(
                cluster_frames,
                primary_field=primary_field,
                disagree_field=disagree_field,
            )
            scene_selected.extend(picks)

            if is_disagreement:
                n_disagree += 1
                # Primary pick (first) stays Red, only alternative gets Purple
                # picks[0] = consensus best, picks[1] = QualiCLIP+ alternative
                picks[1].aesthetic_disagreement = True
            else:
                n_agree += 1
                # Both models picked the same image — strong signal
                picks[0].cluster_agreement = True

            global_cluster_id += 1

        all_selected.extend(scene_selected)

        logger.debug(
            f"  Scene {scene_id}: {len(scene_frames)} images → "
            f"{n_clusters} clusters → {len(scene_selected)} selected"
        )

    # Enforce minimum selection floor
    n_total = len(frames)
    min_keep = max(
        MIN_SELECTION_ABSOLUTE,
        int(np.ceil(n_total * MIN_SELECTION_FRACTION)),
    )

    if len(all_selected) < min_keep:
        # Add more images by relaxing: pick next-best from largest clusters
        selected_set = {f.global_index for f in all_selected}
        remaining = [
            f for f in frames
            if f.global_index not in selected_set
            and f.technical_gate_pass
        ]
        remaining.sort(key=lambda f: -f.composite_relevance)

        for f in remaining:
            if len(all_selected) >= min_keep:
                break
            all_selected.append(f)

        logger.info(
            f"Selection floor: padded to {len(all_selected)} "
            f"(minimum={min_keep})"
        )

    if n_agree + n_disagree > 0:
        logger.info(
            f"Model agreement: {n_agree} clusters agree (Red only), "
            f"{n_disagree} clusters disagree (Red + Purple alternative) "
            f"({n_agree/(n_agree+n_disagree):.0%} agreement rate)"
        )

    # Build output
    selected_indices = [f.global_index for f in all_selected]
    quality_scores = [f.composite_relevance for f in all_selected]

    logger.info(
        f"Stage 4 complete: {len(all_selected)} selected from "
        f"{n_total} images ({total_clusters} visual clusters across "
        f"{len(scenes)} scenes)"
    )

    return all_selected, selected_indices, quality_scores