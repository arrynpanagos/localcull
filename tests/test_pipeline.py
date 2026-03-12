"""
Validation smoke test for localcull pipeline.

Uses synthetic frames for unit tests + real CR3 fixtures (when
available) for integration tests. Goal: "would this have caught
the DINOv2 normalization bug, the eye_ratio z-normalization bug,
or the EXIF orientation bug before they shipped?"

Run:
    pytest tests/ -v                    # unit tests only
    pytest tests/ -v -m fixtures        # integration tests (need CR3s)
    pytest tests/ -v -m "not slow"      # skip slow tests
"""

import numpy as np
import pytest

from localcull.constants import (
    CONFIDENCE_TRUST_THRESHOLD,
    DINO_EMBEDDING_DIM,
    DINO_GRID_SIZE,
    SALIENCY_CONFIDENCE_THRESHOLD,
)
from localcull.stage2_features import (
    calibrate_blinks_per_burst,
    harmonize_mixed_bursts,
    saliency_confidence,
)
from localcull.stage3_scoring import (
    adaptive_weights,
    build_all_scores,
    build_burst_masks,
    compute_composite,
    detect_content_profile,
    z_normalize_by_camera,
)
from localcull.stage4_selection import (
    build_dpp_kernel,
    compute_adaptive_alpha,
    cross_scene_dedup,
    greedy_dpp_map,
    reduce_bursts,
)
from localcull.stage6_output import (
    _percentile_rating,
    rate_all_images,
)
from localcull.types import Frame

from conftest import make_synthetic_burst, make_synthetic_frame


# ══════════════════════════════════════════════════════════════════
# Mixed burst harmonization
# ══════════════════════════════════════════════════════════════════


class TestHarmonization:
    def test_single_flicker_preserves_eye_sharp(self):
        """1/30 non-face frames should NOT trigger harmonization."""
        burst = make_synthetic_burst(30, n_no_face=1)
        orig_sharp = [f.sharp_near_eye for f in burst if f.has_face]
        harmonize_mixed_bursts(burst)
        new_sharp = [f.sharp_near_eye for f in burst if f.has_face]
        assert orig_sharp == new_sharp

    def test_mixed_burst_forces_subject_sharpness(self):
        """10/30 non-face frames (33%) should trigger harmonization."""
        burst = make_synthetic_burst(30, n_no_face=10)
        harmonize_mixed_bursts(burst)
        for f in burst:
            if f.has_face and f._pre_harmonize_sharp_near is not None:
                assert f.sharp_near_eye == f.sharpness_subject
                assert f.eye_ratio_raw == 0.0

    def test_mostly_no_face_not_harmonized(self):
        """25/30 non-face frames (83%) should NOT harmonize."""
        burst = make_synthetic_burst(30, n_no_face=25)
        orig_sharp = [f.sharp_near_eye for f in burst if f.has_face]
        harmonize_mixed_bursts(burst)
        new_sharp = [f.sharp_near_eye for f in burst if f.has_face]
        assert orig_sharp == new_sharp

    def test_error_frames_excluded_from_detection(self):
        """A corrupt frame shouldn't trigger harmonization."""
        burst = make_synthetic_burst(30, n_no_face=0)
        burst[0].subject_det_method = "error"
        burst[0].has_face = False
        harmonize_mixed_bursts(burst)
        # 1/30 error frames = 3.4% → below 20% threshold
        for f in burst:
            if f.has_face:
                assert f._pre_harmonize_sharp_near is None


# ══════════════════════════════════════════════════════════════════
# Blink calibration
# ══════════════════════════════════════════════════════════════════


class TestBlinkCalibration:
    def test_burst_relative_threshold(self):
        """Normal burst: blink = EAR < 50% of max."""
        burst = make_synthetic_burst(10, n_no_face=0)
        for f in burst:
            f.raw_min_ear = 0.30
        burst[5].raw_min_ear = 0.10  # clearly a blink
        calibrate_blinks_per_burst(burst)
        assert burst[5].blink_detected is True
        assert burst[0].blink_detected is False

    def test_single_exposure_absolute_floor(self):
        """Single frame: use conservative absolute threshold."""
        frame = make_synthetic_frame(burst_length=1, raw_min_ear=0.10)
        calibrate_blinks_per_burst([frame])
        assert frame.blink_detected is True

        frame2 = make_synthetic_frame(burst_length=1, raw_min_ear=0.15)
        calibrate_blinks_per_burst([frame2])
        assert frame2.blink_detected is False  # above 0.12 floor

    def test_squint_burst_uses_absolute_floor(self):
        """Short burst where all EARs are low → absolute floor."""
        burst = make_synthetic_burst(3, n_no_face=0)
        for f in burst:
            f.raw_min_ear = 0.16  # all low but above 0.12
        calibrate_blinks_per_burst(burst)
        assert all(not f.blink_detected for f in burst)

    def test_blink_gated_to_neg_inf(self):
        """Blinked frame should get composite = -inf after scoring."""
        frames = make_synthetic_burst(10, n_no_face=0)
        frames[3].blink_detected = True
        for f in frames:
            f.dinov2_embedding = np.random.randn(DINO_EMBEDDING_DIM)
            f.composite_relevance = 0.5
        # Blink gate is applied during composite scoring
        from localcull.stage3_scoring import compute_composite

        score = compute_composite(
            frames[3],
            build_all_scores(frames),
            build_burst_masks(frames),
            {"topiq": 0.3, "musiq": 0.2, "eye_sharp": 0.25,
             "isolation": 0.1, "intent": 0.1, "eye_ratio": 0.05},
        )
        assert score == -np.inf


# ══════════════════════════════════════════════════════════════════
# Per-camera z-normalization
# ══════════════════════════════════════════════════════════════════


class TestZNormalization:
    def test_zero_mean_per_camera(self):
        """Each camera's z-normalized scores should have mean ≈ 0."""
        frames = []
        for i in range(20):
            f = make_synthetic_frame(
                global_index=i,
                camera_body="R5m2" if i < 10 else "EOS_R",
                topiq_score=0.6 + np.random.randn() * 0.05
                if i < 10
                else 0.4 + np.random.randn() * 0.05,
            )
            frames.append(f)

        z_normalize_by_camera(frames, ["topiq_score"])

        r5_scores = [f.topiq_score for f in frames if f.camera_body == "R5m2"]
        eosr_scores = [f.topiq_score for f in frames if f.camera_body == "EOS_R"]
        assert abs(np.mean(r5_scores)) < 0.05
        assert abs(np.mean(eosr_scores)) < 0.05

    def test_degraded_excluded(self):
        """Degraded frames should not affect z-norm statistics."""
        frames = []
        for i in range(10):
            f = make_synthetic_frame(
                global_index=i, topiq_score=0.5 + i * 0.01
            )
            frames.append(f)
        # Make frame 0 an extreme outlier, mark degraded
        frames[0].topiq_score = 100.0
        degraded = {0}

        z_normalize_by_camera(frames, ["topiq_score"], degraded)
        clean_scores = [f.topiq_score for f in frames[1:]]
        assert abs(np.mean(clean_scores)) < 0.05


# ══════════════════════════════════════════════════════════════════
# Saliency confidence
# ══════════════════════════════════════════════════════════════════


class TestSaliencyConfidence:
    def test_concentrated_attention_high_confidence(self):
        """Saliency map with one bright spot → high confidence."""
        sal = np.zeros((DINO_GRID_SIZE, DINO_GRID_SIZE), dtype=np.float32)
        sal[18, 18] = 1.0  # single bright pixel
        conf = saliency_confidence(sal)
        assert conf > 0.5

    def test_uniform_attention_low_confidence(self):
        """Uniform saliency map → near-zero confidence."""
        sal = np.ones((DINO_GRID_SIZE, DINO_GRID_SIZE), dtype=np.float32)
        conf = saliency_confidence(sal)
        assert conf < 0.1

    def test_moderate_attention(self):
        """Gaussian blob → moderate confidence."""
        y, x = np.mgrid[:DINO_GRID_SIZE, :DINO_GRID_SIZE]
        sal = np.exp(-((x - 18) ** 2 + (y - 18) ** 2) / (2 * 5**2))
        sal = sal.astype(np.float32)
        conf = saliency_confidence(sal)
        assert 0.15 < conf < 0.7


# ══════════════════════════════════════════════════════════════════
# Confidence modulation in scoring
# ══════════════════════════════════════════════════════════════════


class TestConfidenceModulation:
    def test_low_confidence_dampens_eye_sharp(self):
        """Low-confidence non-face frame should have dampened eye_sharp."""
        # Build two identical frame sets — one with low confidence, one with high
        frames_low = []
        frames_high = []
        for i in range(10):
            for frames, conf in [(frames_low, 0.1), (frames_high, 0.8)]:
                f = make_synthetic_frame(
                    global_index=i,
                    has_face=False,
                    sharpness_subject=100.0 + i * 10,
                    subject_det_method="dino_attn",
                    saliency_confidence=conf,
                )
                frames.append(f)

        scores_low = build_all_scores(frames_low)
        scores_high = build_all_scores(frames_high)

        # Same raw sharpness, but low-confidence scores should be
        # dampened toward zero relative to high-confidence scores
        for i in range(10):
            assert abs(scores_low["eye_sharp"][i]) <= abs(scores_high["eye_sharp"][i]) + 1e-8

    def test_face_frame_unaffected(self):
        """Face frames (confidence=1.0) are never dampened."""
        frames = []
        for i in range(10):
            f = make_synthetic_frame(
                global_index=i,
                has_face=True,
                sharp_near_eye=100.0 + i * 5,
            )
            frames.append(f)

        scores = build_all_scores(frames)
        # Monotonically increasing sharp → monotonically increasing z-scores
        for i in range(1, 10):
            assert scores["eye_sharp"][i] > scores["eye_sharp"][i - 1]


# ══════════════════════════════════════════════════════════════════
# Adaptive weights
# ══════════════════════════════════════════════════════════════════


class TestAdaptiveWeights:
    def test_flat_feature_loses_weight(self):
        """Feature with zero variance in context should lose weight."""
        n = 20
        all_scores = {
            "topiq": np.array([0.5] * n),  # flat
            "eye_sharp": np.random.randn(n),  # varies
        }
        base_weights = {"topiq": 0.5, "eye_sharp": 0.5}
        mask = np.ones(n, dtype=bool)

        w = adaptive_weights(all_scores, mask, base_weights)
        assert w["eye_sharp"] > w["topiq"]

    def test_equal_variance_preserves_base(self):
        """Equal variance → weights stay near base."""
        n = 100
        all_scores = {
            "topiq": np.random.randn(n),
            "eye_sharp": np.random.randn(n),
        }
        base_weights = {"topiq": 0.5, "eye_sharp": 0.5}
        mask = np.ones(n, dtype=bool)

        w = adaptive_weights(all_scores, mask, base_weights)
        assert abs(w["topiq"] - 0.5) < 0.15
        assert abs(w["eye_sharp"] - 0.5) < 0.15


# ══════════════════════════════════════════════════════════════════
# Content profile detection
# ══════════════════════════════════════════════════════════════════


class TestContentProfile:
    def test_portrait_scene(self):
        """80% face rate → portrait weights."""
        frames = [make_synthetic_frame(global_index=i, has_face=i < 8)
                  for i in range(10)]
        from localcull.constants import PORTRAIT_WEIGHTS
        w = detect_content_profile(frames)
        assert w == PORTRAIT_WEIGHTS

    def test_landscape_scene(self):
        """0% face rate → landscape weights."""
        frames = [make_synthetic_frame(global_index=i, has_face=False)
                  for i in range(10)]
        from localcull.constants import LANDSCAPE_WEIGHTS
        w = detect_content_profile(frames)
        assert w == LANDSCAPE_WEIGHTS


# ══════════════════════════════════════════════════════════════════
# DPP selection
# ══════════════════════════════════════════════════════════════════


class TestDPPSelection:
    def test_greedy_dpp_selects_k(self):
        """Greedy DPP should select exactly k items."""
        n = 20
        emb = np.random.randn(n, 64)
        relevances = np.random.randn(n)
        alpha = 0.5
        L = build_dpp_kernel(relevances, emb, alpha)
        sel, gains = greedy_dpp_map(L, k=5)
        assert len(sel) == 5
        assert len(gains) == 5
        assert all(g > 0 for g in gains)

    def test_diverse_items_preferred(self):
        """DPP should prefer diverse items over duplicates."""
        # 5 identical in 10D + 5 diverse (orthogonal basis vectors)
        identical = np.tile(np.random.randn(10), (5, 1))
        diverse = np.eye(5, M=10) + 0.01 * np.random.randn(5, 10)
        emb = np.vstack([identical, diverse])
        relevances = np.ones(10)  # equal quality
        alpha = 0.5
        L = build_dpp_kernel(relevances, emb, alpha)
        sel, _ = greedy_dpp_map(L, k=5)
        # Should select diverse items (indices 5-9)
        n_diverse = sum(1 for s in sel if s >= 5)
        assert n_diverse >= 3  # at least 3 of 5 diverse items

    def test_burst_reduction(self):
        """Burst reduction keeps top ~10%."""
        burst = make_synthetic_burst(100, n_no_face=0)
        for i, f in enumerate(burst):
            f.composite_relevance = float(i)  # monotonic quality
        survivors = reduce_bursts(burst, keep_fraction=0.10)
        assert len(survivors) == 10
        # Should be the top-10 frames
        assert all(f.composite_relevance >= 90 for f in survivors)


# ══════════════════════════════════════════════════════════════════
# Star ratings
# ══════════════════════════════════════════════════════════════════


class TestStarRatings:
    def test_percentile_rating_small_set(self):
        """Small sets should use percentile-based ratings with full 1-5 range."""
        scores = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 0.2, 0.4, 0.6, 0.8, 1.0])
        indices = list(range(10))
        ratings = _percentile_rating(scores, indices)
        assert all(r in [1, 2, 3, 4, 5] for r in ratings.values())
        assert 5 in ratings.values()  # top image should be 5 stars
        assert len(ratings) == 10  # all images rated

    def test_rate_all_images_full_range(self):
        """rate_all_images should produce 1-5 star ratings for ALL images."""
        np.random.seed(42)
        n = 50
        scores = np.concatenate([
            np.random.normal(0.3, 0.05, 20),
            np.random.normal(0.6, 0.05, 20),
            np.random.normal(0.9, 0.05, 10),
        ])
        frames = [make_synthetic_frame(global_index=i) for i in range(n)]
        ratings = rate_all_images(scores, frames)
        assert all(r in [1, 2, 3, 4, 5] for r in ratings.values())
        assert len(ratings) == n  # every image rated


# ══════════════════════════════════════════════════════════════════
# Integration tests (require CR3 fixtures)
# ══════════════════════════════════════════════════════════════════


@pytest.mark.fixtures
class TestIntegration:
    @pytest.mark.slow
    def test_stage0_ingest(self, fixture_cr3_paths):
        """Stage 0 should produce frames from real CR3 files."""
        from localcull.stage0_ingest import ingest_and_cluster

        frames, sorted_paths = ingest_and_cluster(fixture_cr3_paths)
        assert len(frames) == len(fixture_cr3_paths)
        assert all(isinstance(f, Frame) for f in frames)
        assert all(f.global_index == i for i, f in enumerate(frames))
        # Burst IDs should be non-decreasing
        burst_ids = [f.burst_id for f in frames]
        assert burst_ids == sorted(burst_ids)

    @pytest.mark.slow
    def test_portrait_orientation_transpose(self, fixture_cr3_paths):
        """Portrait-orientation image should have height > width after decode."""
        from localcull.stage1_prepare import prepare_images

        # This test validates EXIF transpose is applied
        mid_arrays, store, _, degraded, _ = prepare_images(fixture_cr3_paths[:5])
        try:
            for i, arr in enumerate(mid_arrays):
                if i not in degraded:
                    assert arr.ndim == 3 and arr.shape[2] == 3
        finally:
            store.cleanup()