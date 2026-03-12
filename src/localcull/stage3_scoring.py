"""
Stage 3: Composite Scoring — Z-Score Consensus + PCA.

Multi-model consensus replaces single-scorer ranking:
  1. Z-normalize each scorer's outputs across the shoot.
  2. composite_relevance = mean(z-scores) — consensus quality axis (≈PC1).
  3. z_disagreement = std(z-scores) — continuous model uncertainty.
  4. PCA on z-matrix extracts orthogonal axes:
     - PC2: aesthetic vs technical character (metadata, not ranking).
     - PC3: structural quality / MUSIQ-unique signal (metadata).
  5. technical_gate_pass = bool (absolute thresholds, unchanged).

Stars are assigned by percentile in Stage 6 (not here).
Cluster-level disagreement (Purple) is determined in Stage 4.

Framework: NumPy (CPU, instant).
"""

import logging
from collections import defaultdict

import numpy as np

from localcull.constants import (
    GENERAL_WEIGHTS,
    LANDSCAPE_WEIGHTS,
    MIXED_FACE_RATE,
    PORTRAIT_FACE_RATE,
    PORTRAIT_WEIGHTS,
    TECH_GATE_TOPIQ_FLOOR,
    TECH_GATE_MUSIQ_FLOOR,
    TECH_GATE_SHARPNESS_FLOOR,
)
from localcull.types import Frame

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════
# Content profile detection
# ══════════════════════════════════════════════════════════════════


def detect_content_profile(scene_frames: list[Frame]) -> dict[str, float]:
    """
    Detect content type for a SCENE from its face detection rate.
    Returns appropriate base weight profile.
    """
    n_total = len(scene_frames)
    if n_total == 0:
        return GENERAL_WEIGHTS

    n_face = sum(1 for f in scene_frames if f.has_face)
    face_rate = n_face / n_total

    non_face = [f for f in scene_frames if not f.has_face]
    if non_face:
        high_conf = sum(
            1 for f in non_face if f.saliency_confidence > 0.3
        ) / len(non_face)
    else:
        high_conf = 0

    if face_rate > PORTRAIT_FACE_RATE:
        logger.info(
            f"  Scene {scene_frames[0].scene_id}: PORTRAIT "
            f"({face_rate:.0%} faces, {n_total} frames)"
        )
        return PORTRAIT_WEIGHTS
    elif face_rate > MIXED_FACE_RATE:
        logger.info(
            f"  Scene {scene_frames[0].scene_id}: MIXED "
            f"({face_rate:.0%} faces, {high_conf:.0%} high-conf "
            f"saliency, {n_total} frames)"
        )
        return GENERAL_WEIGHTS
    else:
        return LANDSCAPE_WEIGHTS


# ══════════════════════════════════════════════════════════════════
# Technical gate (absolute thresholds)
# ══════════════════════════════════════════════════════════════════


def compute_technical_gate(frames: list[Frame]) -> np.ndarray:
    """
    Compute technical acceptability as a boolean per frame.

    Uses absolute thresholds on raw metric scores. Requires 2+ failures
    to trigger — single failure could be artistic (shallow DoF).

    Handles missing scores gracefully: if a scorer wasn't run (score == 0.0),
    it's excluded from failure counting rather than treated as a failure.

    Returns boolean array: True = technically acceptable.
    """
    n = len(frames)
    gate_pass = np.ones(n, dtype=bool)

    for i, f in enumerate(frames):
        if f.blink_detected:
            gate_pass[i] = False
            continue

        failures = 0
        checks = 0

        if f.topiq_score > 0:
            checks += 1
            if f.topiq_score < TECH_GATE_TOPIQ_FLOOR:
                failures += 1
        if f.musiq_score > 0:
            checks += 1
            if f.musiq_score < TECH_GATE_MUSIQ_FLOOR:
                failures += 1

        sharpness = f.sharp_near_eye if f.has_face else f.sharpness_subject
        if sharpness > 0:
            checks += 1
            if sharpness < TECH_GATE_SHARPNESS_FLOOR:
                failures += 1

        min_failures = 2 if checks >= 2 else 1
        gate_pass[i] = failures < min_failures

    return gate_pass


# ══════════════════════════════════════════════════════════════════
# Z-score consensus + PCA
# ══════════════════════════════════════════════════════════════════


def _collect_scorer_matrix(frames: list[Frame]) -> tuple[list[str], np.ndarray]:
    """
    Collect raw scores from consensus-eligible scorers into a (N, K) matrix.

    A scorer is included if:
      - it has non-zero values on frames (was actually run)
      - its include_in_consensus flag is True

    Scorers with include_in_consensus=False still run, appear in CSV/XMP,
    and are available for PCA analysis, but don't influence ranking.
    Returns (scorer_names, raw_matrix).
    """
    from localcull.scorers import SCORER_REGISTRY

    candidates = []
    excluded = []
    for name, spec in SCORER_REGISTRY.items():
        arr = np.array([getattr(f, spec.frame_field, 0.0) for f in frames])
        if np.any(arr != 0.0):
            if spec.include_in_consensus:
                candidates.append((spec.display_name, arr))
            else:
                excluded.append(spec.display_name)

    if excluded:
        logger.debug(f"Excluded from consensus (metadata only): {', '.join(excluded)}")

    if not candidates:
        arr = np.array([f.deqa_score for f in frames])
        candidates = [("DeQA-Score", arr)]

    names = [c[0] for c in candidates]
    matrix = np.column_stack([c[1] for c in candidates])
    return names, matrix


def compute_consensus_and_pca(
    frames: list[Frame],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """
    Compute weighted z-score consensus, disagreement, and PCA axes.

    Consensus uses VLM-dominant weighting from CONSENSUS_WEIGHTS:
    models that understand image content (DeQA, Q-Align) get higher
    weight than pixel-level models (TOPIQ, MUSIQ).

    PCA runs on unweighted z-scores so axis interpretation stays clean.

    Returns:
        z_consensus: (N,) weighted mean of z-scores — primary ranking signal
        z_disagreement: (N,) std of z-scores — model uncertainty (unweighted)
        pc2_scores: (N,) PC2 projection — aesthetic vs technical
        pc3_scores: (N,) PC3 projection — structural quality
        scorer_names: list of active scorer display names
    """
    from localcull.constants import CONSENSUS_WEIGHTS

    scorer_names, raw_matrix = _collect_scorer_matrix(frames)
    n_images, n_scorers = raw_matrix.shape

    # Z-normalize each scorer across the shoot
    means = raw_matrix.mean(axis=0)
    stds = raw_matrix.std(axis=0)
    stds[stds < 1e-8] = 1.0
    z_matrix = (raw_matrix - means) / stds

    # Build weight vector from CONSENSUS_WEIGHTS, default 1.0 for unknown scorers
    weights = np.array([
        CONSENSUS_WEIGHTS.get(name, 1.0) for name in scorer_names
    ])
    weights = weights / weights.sum()  # normalize to sum to 1

    # Weighted consensus
    z_consensus = z_matrix @ weights  # (N,) weighted mean
    # Disagreement stays unweighted — measures raw model spread
    z_disagreement = z_matrix.std(axis=1)

    # Log weights
    weight_str = ", ".join(
        f"{name}={w:.2f}" for name, w in zip(scorer_names, weights)
    )
    logger.info(f"Consensus weights (normalized): {weight_str}")

    # PCA for orthogonal axes
    pc2_scores = np.zeros(n_images)
    pc3_scores = np.zeros(n_images)

    if n_scorers >= 3:
        cov = np.cov(z_matrix, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        explained = eigenvalues / eigenvalues.sum()

        pc_all = z_matrix @ eigenvectors
        pc2_scores = pc_all[:, 1]
        pc3_scores = pc_all[:, 2] if n_scorers >= 3 else np.zeros(n_images)

        logger.info(
            f"PCA: PC1={explained[0]:.0%} (consensus), "
            f"PC2={explained[1]:.0%} (aesthetic/technical), "
            f"PC3={explained[2]:.0%} (structural)"
        )

        # Log PC2 loadings
        pc2_loadings = [(scorer_names[j], eigenvectors[j, 1]) for j in range(n_scorers)]
        pc2_loadings.sort(key=lambda x: -abs(x[1]))
        loadings_str = ", ".join(f"{n}={v:+.2f}" for n, v in pc2_loadings[:3])
        logger.debug(f"PC2 top loadings: {loadings_str}")

    return z_consensus, z_disagreement, pc2_scores, pc3_scores, scorer_names


# ══════════════════════════════════════════════════════════════════
# Stage 3 orchestration
# ══════════════════════════════════════════════════════════════════


def run_composite_scoring(
    frames: list[Frame],
    pu_head=None,
    degraded_indices: set[int] | None = None,
    primary_scorer: str = "deqa_score",
):
    """
    Orchestrate Stage 3: z-score consensus + technical gate.

    1. Z-normalize all active scorer outputs across the shoot.
    2. composite_relevance = mean(z-scores) — multi-model consensus.
    3. z_disagreement = std(z-scores) — continuous model uncertainty.
    4. PC2, PC3 = orthogonal axes (metadata, not ranking).
    5. technical_gate_pass = bool (absolute threshold acceptability).
    """
    logger.info("Stage 3: Composite scoring (z-score consensus + PCA)")
    degraded = degraded_indices or set()

    # Per-scene content profile (for logging)
    scenes = defaultdict(list)
    for f in frames:
        scenes[f.scene_id].append(f)

    logger.info("Content profiles by scene:")
    n_portrait = 0
    n_mixed = 0
    n_landscape = 0
    for scene_id, scene_frames in sorted(scenes.items()):
        weights = detect_content_profile(scene_frames)
        if weights is PORTRAIT_WEIGHTS:
            n_portrait += 1
        elif weights is GENERAL_WEIGHTS:
            n_mixed += 1
        else:
            n_landscape += 1
    logger.info(
        f"  {len(scenes)} scenes: {n_portrait} portrait, "
        f"{n_mixed} mixed, {n_landscape} landscape/wildlife"
    )

    # Technical gate (absolute thresholds)
    gate_pass = compute_technical_gate(frames)
    n_fail = (~gate_pass).sum()
    logger.info(
        f"Technical gate: {gate_pass.sum()} pass, {n_fail} fail "
        f"(TOPIQ<{TECH_GATE_TOPIQ_FLOOR}, "
        f"MUSIQ<{TECH_GATE_MUSIQ_FLOOR}, "
        f"sharp<{TECH_GATE_SHARPNESS_FLOOR}, blink)"
    )

    # Z-score consensus + PCA
    z_consensus, z_disagreement, pc2, pc3, scorer_names = (
        compute_consensus_and_pca(frames)
    )

    # Apply to frames
    for i, frame in enumerate(frames):
        if frame.global_index in degraded:
            frame.composite_relevance = -np.inf
            frame.z_consensus = -np.inf
            frame.z_disagreement = 0.0
            frame.technical_gate_pass = False
        else:
            frame.composite_relevance = z_consensus[i]
            frame.z_consensus = z_consensus[i]
            frame.z_disagreement = z_disagreement[i]
            frame.pc2_aesthetic_vs_technical = pc2[i]
            frame.pc3_structural = pc3[i]
            frame.technical_gate_pass = bool(gate_pass[i])

        frame.aesthetic_disagreement = False

    # Log consensus distribution
    valid = np.array([
        f.z_consensus for f in frames
        if f.global_index not in degraded and np.isfinite(f.z_consensus)
    ])
    if len(valid) > 0:
        logger.info(
            f"Consensus ({len(scorer_names)} scorers: {', '.join(scorer_names)}): "
            f"mean={valid.mean():.3f}, std={valid.std():.3f}, "
            f"range=[{valid.min():.3f}, {valid.max():.3f}]"
        )
        high_agree = np.sum(z_disagreement < 0.3)
        low_agree = np.sum(z_disagreement > 1.0)
        logger.info(
            f"Model agreement: {high_agree} high-confidence (std<0.3), "
            f"{low_agree} controversial (std>1.0)"
        )

    logger.info("Stage 3 complete")


# ══════════════════════════════════════════════════════════════════
# Visual category detection (optional diagnostic)
# ══════════════════════════════════════════════════════════════════


def detect_visual_categories(
    frames: list[Frame],
    embeddings: np.ndarray,
    max_clusters: int = 5,
    silhouette_threshold: float = 0.25,
):
    """
    Detect if a shoot contains distinct visual categories (e.g.,
    portraits mixed with landscapes mixed with detail shots).

    Uses DINOv2 embeddings — already computed in stage 2 — to check
    for natural clustering in visual feature space. If the silhouette
    score exceeds the threshold, the shoot likely contains distinct
    categories that would benefit from separate analysis.

    Returns (n_categories, silhouette, category_labels) or
    (1, 0.0, None) if no distinct categories found.
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    n = len(frames)
    if n < 10:
        return 1, 0.0, None

    # Try 2 through max_clusters, pick best silhouette
    best_score = -1.0
    best_k = 1
    best_labels = None

    for k in range(2, min(max_clusters + 1, n)):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=5)
        labels = kmeans.fit_predict(embeddings)
        score = silhouette_score(embeddings, labels, sample_size=min(1000, n))
        if score > best_score:
            best_score = score
            best_k = k
            best_labels = labels

    if best_score < silhouette_threshold:
        logger.info(
            f"Visual category check: shoot appears homogeneous "
            f"(best silhouette={best_score:.2f} < {silhouette_threshold})"
        )
        return 1, best_score, None

    # Log category breakdown
    from collections import Counter
    category_counts = Counter(best_labels)
    breakdown = ", ".join(
        f"category {k}: {v} images" for k, v in sorted(category_counts.items())
    )
    logger.warning(
        f"Visual category check: shoot appears to contain {best_k} "
        f"distinct visual categories (silhouette={best_score:.2f}). "
        f"Breakdown: {breakdown}. "
        f"Ratings may be more meaningful if similar photos are "
        f"grouped into separate folders and run independently."
    )

    # Assign category to frames for CSV export
    for i, frame in enumerate(frames):
        frame.visual_category = int(best_labels[i])

    return best_k, best_score, best_labels


def build_composite_scores_array(frames: list[Frame]) -> np.ndarray:
    """Build numpy array indexed by global_index for Stage 6."""
    n = max(f.global_index for f in frames) + 1
    scores = np.full(n, -np.inf)
    for f in frames:
        scores[f.global_index] = f.composite_relevance
    return scores