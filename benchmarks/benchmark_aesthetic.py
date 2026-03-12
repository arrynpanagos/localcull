"""
Collected tunable constants for the localcull pipeline.

All values are initial estimates. Validate empirically during
implementation using the Pre-Implementation Checklist (spec §10).

Constants marked [SOFT] have graceful failure modes —
miscalibration degrades quality but doesn't break the pipeline.
Constants marked [HARD] can cause incorrect behavior if wrong.
"""

import hashlib
import os

# ══════════════════════════════════════════════════════════════════
# Pipeline version
# ══════════════════════════════════════════════════════════════════
PIPELINE_VERSION = "v5.4"
VERSION_HASH = hashlib.md5(PIPELINE_VERSION.encode()).hexdigest()[:8]

# ══════════════════════════════════════════════════════════════════
# Paths
# ══════════════════════════════════════════════════════════════════
CHECKPOINT_DIR = os.path.expanduser("~/.localcull_cache")
SHM_TRACKING_FILE = os.path.expanduser("~/.localcull_shm_active")

# ══════════════════════════════════════════════════════════════════
# Image Processing
# ══════════════════════════════════════════════════════════════════
MID_LONGEST_SIDE = 2500              # [SOFT] mid-res decode size
EMBEDDED_JPEG_MIN_BYTES = 10_000     # [HARD] corrupt extraction guard

# ══════════════════════════════════════════════════════════════════
# Temporal Clustering (Stage 0)
# ══════════════════════════════════════════════════════════════════
BURST_GAP_SECONDS = 0.5              # [HARD] intra-burst boundary
SCENE_GAP_SECONDS = 10.0             # [SOFT] scene boundary (temporal)
FOCAL_LENGTH_CHANGE_THRESHOLD = 0.05 # [SOFT] relative focal length delta
EXPOSURE_CHANGE_STOPS = 1.0          # [SOFT] EV difference threshold

# ══════════════════════════════════════════════════════════════════
# Face Analysis (Stage 2D)
# ══════════════════════════════════════════════════════════════════
MAX_NUM_FACES = 5                    # [SOFT] Face Mesh limit per image
EAR_CALIBRATION_FACTOR = 0.50        # [SOFT] blink = 50% of burst max EAR
EAR_ABSOLUTE_FLOOR = 0.12            # [SOFT] single-frame blink floor
EAR_SQUINT_SUSPECT = 0.20            # [SOFT] max EAR below = all-squint
HARMONIZE_MIN_FACE_RATE = 0.20       # [SOFT] mixed burst lower bound
HARMONIZE_MAX_FACE_RATE = 0.80       # [SOFT] mixed burst upper bound
EYE_BBOX_PAD_FRACTION = 0.3          # [SOFT] expand eye bbox for context
MAJOR_FACE_AREA_FRACTION = 0.10      # [SOFT] >10% of largest = major face

# MediaPipe Face Mesh landmark indices
LEFT_EYE_CONTOUR = [
    33, 7, 163, 144, 145, 153, 154, 155, 133,
    173, 157, 158, 159, 160, 161, 246,
]
RIGHT_EYE_CONTOUR = [
    362, 382, 381, 380, 374, 373, 390, 249,
    263, 466, 388, 387, 386, 385, 384, 398,
]
LEFT_EAR_POINTS = [33, 159, 145, 133, 153, 160]
RIGHT_EAR_POINTS = [362, 386, 374, 263, 380, 387]

# ══════════════════════════════════════════════════════════════════
# Saliency & Subject Detection (Stage 2E)
# ══════════════════════════════════════════════════════════════════
SALIENCY_CONFIDENCE_THRESHOLD = 0.3  # [SOFT] hard/soft mask boundary
CONFIDENCE_TRUST_THRESHOLD = 0.5     # [SOFT] eye_sharp dampening
SALIENCY_TINY_MASK_GUARD = 0.005     # [SOFT] widen mask if <0.5% pixels

# DINOv2 spatial grid (518px / 14px patch = 37 patches per side)
DINO_GRID_SIZE = 37
DINO_NUM_PATCHES = DINO_GRID_SIZE * DINO_GRID_SIZE  # 1369
DINO_EMBEDDING_DIM = 3072  # CLS(1024) + patch_mean(1024) + patch_std(1024)

# ImageNet normalization (float32 to avoid upcasting)
import numpy as np
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
IMAGENET_MEAN_HWC = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
IMAGENET_STD_HWC = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

# ══════════════════════════════════════════════════════════════════
# Content Profiles (Stage 3)
# ══════════════════════════════════════════════════════════════════
PORTRAIT_FACE_RATE = 0.50            # [SOFT] >50% → portrait weights
MIXED_FACE_RATE = 0.20               # [SOFT] 20-50% → general weights

PORTRAIT_WEIGHTS = {
    "topiq": 0.30,
    "musiq": 0.20,
    "eye_sharp": 0.25,
    "isolation": 0.10,
    "intent": 0.10,
    "eye_ratio": 0.05,
}

GENERAL_WEIGHTS = {
    "topiq": 0.35,
    "musiq": 0.25,
    "eye_sharp": 0.20,
    "isolation": 0.05,
    "intent": 0.10,
    "eye_ratio": 0.05,
}

LANDSCAPE_WEIGHTS = {
    "topiq": 0.35,
    "musiq": 0.30,
    "eye_sharp": 0.15,
    "isolation": 0.05,
    "intent": 0.10,
    "eye_ratio": 0.05,
}

# Adaptive weight regularization blend
ADAPTIVE_WEIGHT_REGULARIZATION = 0.5  # [SOFT]

# ── Technical gate ──
GATE_MIDPOINT = 0.4      # Technical score considered "acceptable"

# ── Aesthetic disagreement detection ──
# When Q-Align and QualiCLIP+ (rescaled to [1,5]) differ by more than
# this threshold, flag as Purple for photographer review.
AESTHETIC_DISAGREEMENT_THRESHOLD = 1.0  # in Q-Align [1,5] scale

# ══════════════════════════════════════════════════════════════════
# Selection (Stage 4) — Cluster → Rank → Pick
# ══════════════════════════════════════════════════════════════════

# Minimum selection floor (prevents collapse)
MIN_SELECTION_FRACTION = 0.02        # [HARD] Never select fewer than 2% of input.
                                     #   331 images → floor of 7.
MIN_SELECTION_ABSOLUTE = 3           # [HARD] Absolute floor regardless of input size.

# ══════════════════════════════════════════════════════════════════
# Rendering Path
# ══════════════════════════════════════════════════════════════════
SMALL_GROUP_MERGE_THRESHOLD = 5      # [SOFT] merge <5 into primary

# ══════════════════════════════════════════════════════════════════
# GPU Inference (Stage 2)
# ══════════════════════════════════════════════════════════════════
DINOV2_BATCH_SIZE = 8                # [SOFT] MLX batch — ViT-L attention is O(n²) on 1370 tokens
DINOV2_INPUT_SIZE = 518              # DINOv2 ViT-L input resolution
TOPIQ_BATCH_SIZE = 32                # [SOFT] MPS batch
TOPIQ_SIZE = (384, 384)              # TOPIQ ResNet/Swin backbone
MUSIQ_BATCH_SIZE = 1                 # [SOFT] preserves aspect ratio
QALIGN_BATCH_SIZE = 1               # [HARD] pyiqa Q-Align asserts batch_size==1
QUALICLIP_BATCH_SIZE = 32           # [SOFT] CLIP-based, lightweight

# ══════════════════════════════════════════════════════════════════
# Parallelism
# ══════════════════════════════════════════════════════════════════
JPEG_EXTRACT_WORKERS = 20            # [SOFT] Stage 1 parallelism
FACE_ANALYSIS_WORKERS = 10           # [SOFT] Balance CPU saturation vs mmap pressure (each maps full shm block)

# ══════════════════════════════════════════════════════════════════
# Memory safety
# ══════════════════════════════════════════════════════════════════
MEMORY_LIMIT_GB = 150                # [SOFT] Abort if system memory exceeds this (192GB total, leave headroom)


def check_memory(stage_name: str = ""):
    """Abort if system memory usage exceeds MEMORY_LIMIT_GB."""
    try:
        import psutil
        mem = psutil.virtual_memory()
        used_gb = mem.used / (1024 ** 3)
        if used_gb > MEMORY_LIMIT_GB:
            raise MemoryError(
                f"localcull safety abort: system memory usage "
                f"({used_gb:.1f} GB) exceeds limit ({MEMORY_LIMIT_GB} GB) "
                f"at {stage_name}. "
                f"Reduce batch sizes or image count."
            )
    except ImportError:
        pass  # psutil not available, skip check

# ══════════════════════════════════════════════════════════════════
# VLM (Phase 2)
# ══════════════════════════════════════════════════════════════════
VLM_MAX_PAIRS = 50                   # [SOFT] max pairwise comparisons
VLM_MODEL_PATH = "mlx-community/Qwen2.5-VL-7B-Instruct-4bit"

# ══════════════════════════════════════════════════════════════════
# Personalization (Phase 2)
# ══════════════════════════════════════════════════════════════════
PU_C = 0.8                           # [SOFT] PU estimator c
PU_MIN_EXAMPLES = 50                 # [SOFT] activation threshold
PU_PERSONAL_BLEND = 0.2              # [SOFT] personal vs base weight

# ══════════════════════════════════════════════════════════════════
# Multi-face aggregation
# ══════════════════════════════════════════════════════════════════
MULTI_FACE_WEIGHTED_AVG_RATIO = 0.6  # [SOFT] weighted avg contribution
MULTI_FACE_WORST_PENALTY_RATIO = 0.4 # [SOFT] worst-major contribution