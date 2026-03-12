"""
Core data types for the localcull pipeline.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class Frame:
    """
    Central data object tracking a single image through all pipeline stages.

    Fields are populated progressively — Stage 0 sets identity and
    temporal metadata, Stage 1 adds image data references, Stage 2
    fills feature scores, Stage 3 computes the composite.
    """

    # ── Stage 0: always set at construction ──
    path: str
    global_index: int                    # position in the master frame list (0..N-1)
    scene_id: int
    burst_id: int
    frame_index: int                     # position within burst
    burst_length: int
    camera_body: str
    exif: dict = field(default_factory=dict)
    timestamp: float = 0.0              # seconds since epoch (from EXIF)

    # ── Stage 1: populated during image preparation ──
    mid_array: Optional[np.ndarray] = None   # decoded mid-res (~12.5 MB)
    full_shm_index: int = -1                 # index into CompressedImageStore
    embedded_jpeg_path: Optional[str] = None # path to extracted JPEG (for VLM)
    rendering_path: str = "embedded_jpeg"    # 'embedded_jpeg' or 'rawpy'

    # ── Stage 2: populated during feature extraction ──
    topiq_score: float = 0.0             # continuous ∈ [0, 1]
    musiq_score: float = 0.0             # continuous ∈ [0, 1]
    qalign_score: float = 0.0           # Q-Align aesthetic ∈ [1, 5] (MOS scale)
    qualiclip_score: float = 0.0        # QualiCLIP+ aesthetic ∈ [0, 1]
    deqa_score: float = 0.0             # DeQA-Score aesthetic ∈ [1, 5] (CVPR 2025)
    q_scorer_score: float = 0.0         # Q-Scorer aesthetic ∈ [1, 5] (AAAI 2025)
    q_insight_score: float = 0.0        # Q-Insight aesthetic ∈ [1, 5] (NeurIPS 2025)
    nima_score: float = 0.0             # NIMA aesthetic ∈ [1, 10] (AVA-trained, 2018)
    artimuse_score: float = 0.0         # ArtiMuse artistic aesthetic ∈ [0, 10] (CVPR 2026)
    unipercept_score: float = 0.0       # UniPercept unified perceptual ∈ [0, 10] (2025)
    sharp_near_eye: float = 0.0          # Laplacian var, sharper eye
    sharp_far_eye: float = 0.0           # Laplacian var, other eye
    sharpness_subject: float = 0.0       # bbox-level (fallback or aggregate)
    sharpness_background: float = 0.0
    isolation_ratio: float = 1.0
    blink_detected: bool = False         # per-burst calibrated EAR
    raw_min_ear: float = 1.0             # raw EAR value (blink calibration input)
    eye_ratio_raw: float = 0.0           # near/far eye sharpness ratio (pre-z-norm)
    has_face: bool = False
    n_faces: int = 0
    subject_det_method: str = "pending"  # 'face_mesh', 'dino_attn', 'center', 'error'
    dinov2_embedding: Optional[np.ndarray] = None  # 1024-dim
    saliency_map: Optional[np.ndarray] = None      # [37, 37] DINOv2 CLS attention
    saliency_confidence: float = 0.0     # 0=diffuse, 1=concentrated

    # ── Pre-harmonization originals (preserved by harmonize_mixed_bursts) ──
    _pre_harmonize_sharp_near: Optional[float] = None
    _pre_harmonize_sharp_far: Optional[float] = None
    _pre_harmonize_eye_ratio: Optional[float] = None

    # ── Stage 3: populated during composite scoring ──
    composite_relevance: float = 0.0
    technical_gate_pass: bool = True      # False = technically flawed (Green label)
    aesthetic_disagreement: bool = False   # True = disagreement alternative pick (Purple label, never on primary Red)
    cluster_agreement: bool = False        # True = both consensus and disagreement scorer picked this image
    cluster_id: int = -1                    # Visual cluster assignment from stage4 (unique across shoot)
    visual_category: int = -1               # DINOv2-based category detection (-1 = not run or homogeneous)
    z_consensus: float = 0.0             # mean of z-normalized scorer outputs
    z_disagreement: float = 0.0          # std of z-normalized scorer outputs (model uncertainty)
    pc2_aesthetic_vs_technical: float = 0.0  # PC2: high=artistic, low=clinical (metadata only)
    pc3_structural: float = 0.0          # PC3: MUSIQ-driven structural quality (metadata only)