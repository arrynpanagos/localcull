"""
Stage 2: Feature Extraction.

Sequential pipeline (unified memory safe):
  Phase 1 (CPU parallel): Face Mesh + eye sharpness + blink gate
  Phase 2 (GPU sequential): DINOv2 (MLX) → TOPIQ-NR (MPS) → MUSIQ (MPS)
  Phase 3 (CPU parallel): Saliency sharpness for non-face frames

Time: ~2–4 minutes for 300 images.
"""

import io
import logging
import multiprocessing.shared_memory as shm
import os
import struct
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

# ── Constrain internal thread pools BEFORE importing NumPy/OpenCV ──
# Without this, each ProcessPoolExecutor worker spawns 76 OpenCV/BLAS
# threads on M3 Ultra → 28 workers × 76 threads = 2000+ threads
# thrashing unified memory. The parallelism comes from the process pool,
# not from within each worker.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"  # Apple Accelerate

import cv2
import numpy as np
from PIL import Image, ImageOps
from tqdm import tqdm

cv2.setNumThreads(1)

from localcull.checkpoint import load_checkpoint, save_checkpoint
from localcull.constants import (
    CONFIDENCE_TRUST_THRESHOLD,
    DINO_GRID_SIZE,
    DINOV2_BATCH_SIZE,
    DINOV2_INPUT_SIZE,
    EAR_ABSOLUTE_FLOOR,
    EAR_CALIBRATION_FACTOR,
    EAR_SQUINT_SUSPECT,
    EYE_BBOX_PAD_FRACTION,
    FACE_ANALYSIS_WORKERS,
    HARMONIZE_MAX_FACE_RATE,
    HARMONIZE_MIN_FACE_RATE,
    IMAGENET_MEAN,
    IMAGENET_MEAN_HWC,
    IMAGENET_STD,
    IMAGENET_STD_HWC,
    LEFT_EAR_POINTS,
    LEFT_EYE_CONTOUR,
    MAJOR_FACE_AREA_FRACTION,
    MAX_NUM_FACES,
    MID_LONGEST_SIDE,
    MULTI_FACE_WEIGHTED_AVG_RATIO,
    MULTI_FACE_WORST_PENALTY_RATIO,
    RIGHT_EAR_POINTS,
    RIGHT_EYE_CONTOUR,
    SALIENCY_CONFIDENCE_THRESHOLD,
)
from localcull.types import Frame

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════
# 2A–2B. Quality/Aesthetic Scorers (dispatched via scorer registry)
# ══════════════════════════════════════════════════════════════════
#
# All scorer functions (TOPIQ, MUSIQ, Q-Align, QualiCLIP+, DeQA-Score,
# Q-Scorer, Q-Insight) live in localcull.scorers and are dispatched
# by name via the SCORER_REGISTRY. Each runs in an isolated subprocess
# for Metal memory safety.

# ══════════════════════════════════════════════════════════════════
# 2C. DINOv2 Embedding (GPU, MLX, batched)
# ══════════════════════════════════════════════════════════════════
#
# Uses mlxim's native get_intermediate_layers() API which returns
# both token sequences AND attention weights. No monkey-patching needed.
#
# From the last layer we build:
#   - Diversity embeddings: [CLS, patch_mean, patch_std] = 3072-dim
#   - Saliency maps: CLS→patch attention from last layer (16 heads averaged)
#
# CLS tokens encode WHAT ("person, outdoor"). Patch tokens encode WHERE
# ("face top-left, bokeh right"). For same-person portrait shoots,
# CLS cosine is 0.95-0.99. Patch statistics drop to 0.7-0.85 between
# different poses — enough for DPP to diversify.


def _resize_batch_mlx(mid_arrays, start, end):
    """Resize and normalize a batch of images to DINOv2 input format."""
    import mlx.core as mx
    from PIL import Image as PILImage

    batch = []
    for img in mid_arrays[start:end]:
        pil = PILImage.fromarray(img).resize(
            (DINOV2_INPUT_SIZE, DINOV2_INPUT_SIZE), PILImage.LANCZOS
        )
        arr = np.array(pil).astype(np.float32) / 255.0
        arr = (arr - IMAGENET_MEAN_HWC) / IMAGENET_STD_HWC
        batch.append(mx.array(arr))
    return mx.stack(batch)


def _build_diversity_embedding(
    cls_tokens: np.ndarray,
    patch_tokens: np.ndarray,
    has_real_cls: bool = True,
) -> np.ndarray:
    """
    Build diversity-aware embedding from CLS + patch token statistics.

    When has_real_cls=True (1370 tokens): [CLS, patch_mean, patch_std] = 3072-dim
    When has_real_cls=False (1369 tokens, CLS is just mean-pooled):
      [patch_mean, patch_std, patch_spatial_var] = 3072-dim
      where spatial_var captures how token norms vary across the image grid.

    Each component L2-normalized so they contribute equally.
    """
    patch_mean = patch_tokens.mean(axis=1)  # [B, 1024]
    patch_std = patch_tokens.std(axis=1)    # [B, 1024]

    def _l2_norm(x):
        norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-8
        return x / norms

    if has_real_cls:
        return np.concatenate([
            _l2_norm(cls_tokens),
            _l2_norm(patch_mean),
            _l2_norm(patch_std),
        ], axis=1)  # [B, 3072]
    else:
        # No real CLS — use spatial structure as third component.
        # Split patches into quadrants and compute per-quadrant means.
        # This captures spatial layout differences that patch_mean loses.
        B = patch_tokens.shape[0]
        grid = patch_tokens.reshape(B, DINO_GRID_SIZE, DINO_GRID_SIZE, -1)
        h2, w2 = DINO_GRID_SIZE // 2, DINO_GRID_SIZE // 2
        q_tl = grid[:, :h2, :w2, :].reshape(B, -1, grid.shape[-1]).mean(axis=1)
        q_tr = grid[:, :h2, w2:, :].reshape(B, -1, grid.shape[-1]).mean(axis=1)
        q_bl = grid[:, h2:, :w2, :].reshape(B, -1, grid.shape[-1]).mean(axis=1)
        q_br = grid[:, h2:, w2:, :].reshape(B, -1, grid.shape[-1]).mean(axis=1)
        # Differences between quadrants encode spatial composition
        spatial = np.concatenate([
            q_tl - q_br,  # diagonal contrast
            q_tr - q_bl,  # other diagonal
        ], axis=1)  # [B, 2048]
        # Pad or truncate to 1024 to keep embedding at 3072
        if spatial.shape[1] >= 1024:
            spatial = spatial[:, :1024]
        else:
            spatial = np.pad(spatial, ((0, 0), (0, 1024 - spatial.shape[1])))

        return np.concatenate([
            _l2_norm(patch_mean),
            _l2_norm(patch_std),
            _l2_norm(spatial),
        ], axis=1)  # [B, 3072]


def _extract_saliency(attn_weights: np.ndarray) -> np.ndarray:
    """
    Extract saliency maps from attention weights.

    Handles two attention shapes:
      [B, heads, 1370, 1370] — has CLS token, use CLS→patch attention
      [B, heads, 1369, 1369] — patches only, use mean attention per patch
    """
    n_tokens = attn_weights.shape[-1]
    n_patches = DINO_GRID_SIZE * DINO_GRID_SIZE  # 1369

    if n_tokens == n_patches + 1:
        # CLS at index 0 attending to patches at 1:
        cls_attn = attn_weights[:, :, 0, 1:]  # [B, heads, 1369]
        cls_attn = cls_attn.mean(axis=1)       # [B, 1369]
    elif n_tokens == n_patches:
        # No CLS — average each patch's received attention across all sources
        patch_attn = attn_weights.mean(axis=1)  # [B, 1369, 1369]
        cls_attn = patch_attn.mean(axis=1)      # [B, 1369]
    else:
        # Unexpected — try CLS→patch if large enough
        cls_attn = attn_weights[:, :, 0, 1:n_patches + 1].mean(axis=1)

    return cls_attn.reshape(-1, DINO_GRID_SIZE, DINO_GRID_SIZE)


def dinov2_embed_batch(
    mid_arrays: list[np.ndarray],
    batch_size: int = DINOV2_BATCH_SIZE,
) -> tuple[np.ndarray, list[np.ndarray | None]]:
    """
    Batched DINOv2 embedding extraction using mlxim's native API.

    Uses get_intermediate_layers(x, n=1) to get:
      - Last layer's full token sequence → diversity embeddings (3072-dim)
      - Last layer's attention weights → saliency maps

    Memory per batch of 8:
      - Token sequence: [8, 1370, 1024] float32 = 5.6 MB
      - Last-layer attention: [8, 16, 1370, 1370] float32 = 96 MB
      - Total: ~102 MB (vs 8.6 GB from get_features which returned all 24 layers)

    Returns (embeddings, saliency_maps):
      - embeddings: [N, 3072] diversity-aware embeddings
      - saliency_maps: list of [37, 37] arrays (or None if extraction fails)
    """
    import gc

    import mlx.core as mx
    from mlxim.model import create_model

    from localcull.constants import check_memory
    check_memory("DINOv2 model load")

    dinov2 = create_model("vit_large_patch14_518.dinov2")

    all_emb = []
    all_saliency = []
    _mode = None  # "intermediate_layers", "intermediate_layers_no_norm", or "call_only"

    # ── Discover which API path works on first batch ──
    first_batch = _resize_batch_mlx(mid_arrays, 0, min(batch_size, len(mid_arrays)))

    # Try 1: get_intermediate_layers with norm
    if _mode is None:
        try:
            intermediate_x, attn_list = dinov2.get_intermediate_layers(
                first_batch, n=1, norm=True
            )
            last_layer = intermediate_x[-1]
            mx.eval(last_layer)
            _mode = "intermediate_layers"
            print(f"[DINOv2] get_intermediate_layers(norm=True) works. Shape: {last_layer.shape}")
            del last_layer, intermediate_x, attn_list
        except Exception as e:
            print(f"[DINOv2] get_intermediate_layers(norm=True) failed: {e}")

    # Try 2: get_intermediate_layers without norm
    if _mode is None:
        try:
            intermediate_x, attn_list = dinov2.get_intermediate_layers(
                first_batch, n=1
            )
            last_layer = intermediate_x[-1]
            mx.eval(last_layer)
            _mode = "intermediate_layers_no_norm"
            print(f"[DINOv2] get_intermediate_layers(no norm) works. Shape: {last_layer.shape}")
            del last_layer, intermediate_x, attn_list
        except Exception as e:
            print(f"[DINOv2] get_intermediate_layers(no norm) failed: {e}")

    # Try 3: __call__ only (CLS-only fallback)
    if _mode is None:
        _mode = "call_only"
        print("[DINOv2] WARNING: Falling back to __call__ (CLS-only, poor diversity)")

    del first_batch
    gc.collect()

    # ── Check if model has layer norm for manual application ──
    _has_ln = False
    if _mode == "intermediate_layers_no_norm":
        if hasattr(dinov2, 'encoder') and hasattr(dinov2.encoder, 'ln'):
            _has_ln = True
            print("[DINOv2] Will apply encoder.ln manually")
        elif hasattr(dinov2, 'norm'):
            _has_ln = True
            print("[DINOv2] Will apply model.norm manually")

    _has_attention = None

    for i in tqdm(range(0, len(mid_arrays), batch_size), desc="DINOv2"):
        batch = _resize_batch_mlx(mid_arrays, i, i + batch_size)
        actual_size = min(batch_size, len(mid_arrays) - i)

        if _mode in ("intermediate_layers", "intermediate_layers_no_norm"):
            # ── Primary path: get_intermediate_layers ──
            if _mode == "intermediate_layers":
                intermediate_x, attn_list = dinov2.get_intermediate_layers(
                    batch, n=1, norm=True
                )
            else:
                intermediate_x, attn_list = dinov2.get_intermediate_layers(
                    batch, n=1
                )

            last_layer = intermediate_x[-1]

            # Apply layer norm manually if needed
            if _mode == "intermediate_layers_no_norm" and _has_ln:
                if hasattr(dinov2, 'encoder') and hasattr(dinov2.encoder, 'ln'):
                    last_layer = dinov2.encoder.ln(last_layer)
                elif hasattr(dinov2, 'norm'):
                    last_layer = dinov2.norm(last_layer)

            mx.eval(last_layer)

            last_layer_np = np.array(last_layer)
            del last_layer, intermediate_x

            # get_intermediate_layers may return:
            #   [B, 1370, 1024] — CLS + 1369 patches (CLS at index 0)
            #   [B, 1369, 1024] — patches only (no CLS token)
            n_tokens = last_layer_np.shape[1]
            n_patches = DINO_GRID_SIZE * DINO_GRID_SIZE  # 1369

            if n_tokens == n_patches + 1:
                # Has CLS token at index 0
                cls_tokens = last_layer_np[:, 0, :]
                patch_tokens = last_layer_np[:, 1:, :]
                _real_cls = True
            elif n_tokens == n_patches:
                # Patches only — use global average pool as CLS substitute
                patch_tokens = last_layer_np  # [B, 1369, 1024]
                cls_tokens = patch_tokens.mean(axis=1)  # [B, 1024]
                _real_cls = False
            else:
                raise ValueError(
                    f"Unexpected token count {n_tokens} "
                    f"(expected {n_patches} or {n_patches + 1})"
                )

            embedding = _build_diversity_embedding(cls_tokens, patch_tokens, has_real_cls=_real_cls)

            if _has_attention is None and i == 0:
                print(
                    f"[DINOv2] Token mode: {n_tokens} tokens "
                    f"({'CLS+patches' if _real_cls else 'patches-only, using spatial quadrants'})"
                )

            if attn_list is not None and len(attn_list) > 0:
                last_attn = attn_list[-1]
                mx.eval(last_attn)
                last_attn_np = np.array(last_attn)
                saliency = _extract_saliency(last_attn_np)
                del last_attn, last_attn_np

                if _has_attention is None:
                    _has_attention = True
                    print(
                        f"[DINOv2] attention+diversity mode "
                        f"(embedding dim={embedding.shape[1]}, "
                        f"saliency={DINO_GRID_SIZE}x{DINO_GRID_SIZE})"
                    )
            else:
                # No attention returned — use patch token norms as proxy
                norms = np.linalg.norm(patch_tokens, axis=-1)
                saliency = norms.reshape(-1, DINO_GRID_SIZE, DINO_GRID_SIZE)
                for j in range(len(saliency)):
                    s_min, s_max = saliency[j].min(), saliency[j].max()
                    if s_max - s_min > 1e-8:
                        saliency[j] = (saliency[j] - s_min) / (s_max - s_min)

                if _has_attention is None:
                    _has_attention = False
                    print(
                        f"[DINOv2] diversity mode (no attention weights). "
                        f"Using patch-norm saliency. "
                        f"Embedding dim={embedding.shape[1]}"
                    )

            del last_layer_np, cls_tokens, patch_tokens, attn_list

            all_emb.append(embedding)
            all_saliency.append(saliency)

        else:
            # ── CLS-only fallback ──
            emb = dinov2(batch)
            mx.eval(emb)
            emb_np = np.array(emb)
            del emb

            if emb_np.ndim == 3:
                emb_np = emb_np[:, 0, :]

            all_emb.append(emb_np)
            all_saliency.append(
                np.full((actual_size, DINO_GRID_SIZE, DINO_GRID_SIZE), np.nan)
            )

        del batch
        gc.collect()

    embeddings = np.concatenate(all_emb, axis=0)
    saliency_raw = np.concatenate(all_saliency, axis=0)

    del dinov2, all_emb, all_saliency
    gc.collect()

    has_saliency = not np.isnan(saliency_raw).all()
    saliency_maps = (
        [saliency_raw[i] for i in range(len(embeddings))]
        if has_saliency
        else [None] * len(embeddings)
    )

    print(
        f"[DINOv2] Complete: {len(embeddings)} embeddings, "
        f"dim={embeddings.shape[1]}, "
        f"saliency={'attention' if _has_attention else 'patch-norm' if has_saliency else 'none'}"
    )

    return embeddings, saliency_maps


# ══════════════════════════════════════════════════════════════════
# 2D. Face Analysis + Eye Sharpness + Blink Gate (CPU, parallel)
# ══════════════════════════════════════════════════════════════════

# ── Per-worker state (initialized once per process) ──
_face_landmarker = None
_worker_shm = None
_HEADER_ENTRY = struct.Struct("<QQ")

# Face landmarker model URL and cache path
_FACE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
)
_FACE_MODEL_PATH = os.path.join(
    os.path.expanduser("~/.localcull_cache"), "face_landmarker.task"
)


def _ensure_face_model() -> str:
    """Download face landmarker model if not cached. Returns model path."""
    if os.path.exists(_FACE_MODEL_PATH):
        return _FACE_MODEL_PATH
    os.makedirs(os.path.dirname(_FACE_MODEL_PATH), exist_ok=True)
    logger.info(f"Downloading face landmarker model to {_FACE_MODEL_PATH}")
    import urllib.request
    urllib.request.urlretrieve(_FACE_MODEL_URL, _FACE_MODEL_PATH)
    return _FACE_MODEL_PATH


def _init_face_worker(shm_name: str, model_path: str):
    """
    Initialize FaceLandmarker and shared memory handle once per worker.

    Thread constraints (OMP, BLAS, OpenCV) set at module level and
    inherited by spawned processes. CPU delegate forces MediaPipe
    to skip Metal GL context creation.
    """
    import atexit

    cv2.setNumThreads(1)  # Per-process OpenCV setting

    # Suppress Metal GPU context creation in TFLite/MediaPipe.
    # delegate=CPU only controls inference device; TFLite still probes
    # Metal availability on init, creating a Metal device + command queue
    # (~50-100MB unified memory per worker that never frees).
    os.environ["TFLITE_DISABLE_GPU"] = "1"
    os.environ["METAL_DEVICE_WRAPPER_TYPE"] = "0"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["GLOG_minloglevel"] = "3"

    global _face_landmarker, _worker_shm
    import mediapipe as mp

    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=model_path,
            delegate=mp.tasks.BaseOptions.Delegate.CPU,
        ),
        num_faces=MAX_NUM_FACES,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )

    # Redirect stderr to /dev/null during MediaPipe init.
    # C++ code emits W0000/I0000/INFO lines that bypass Python logging.
    _saved_stderr = os.dup(2)
    _devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(_devnull, 2)
    try:
        _face_landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)
    finally:
        os.dup2(_saved_stderr, 2)
        os.close(_saved_stderr)
        os.close(_devnull)
    _worker_shm = shm.SharedMemory(name=shm_name)

    def _cleanup():
        try:
            _worker_shm.close()
        except Exception:
            pass
    atexit.register(_cleanup)


def _worker_get_jpeg_bytes(index: int) -> bytes:
    """Read JPEG bytes using cached worker shm handle (no open/close)."""
    offset, length = _HEADER_ENTRY.unpack_from(
        _worker_shm.buf, index * _HEADER_ENTRY.size
    )
    return bytes(_worker_shm.buf[offset : offset + length])


def compute_ear(landmarks, eye_indices, img_w: int, img_h: int) -> float:
    """Eye Aspect Ratio — open eye ≈ 0.25–0.35, closed < 0.18."""

    def pt(i):
        return np.array([landmarks[i].x * img_w, landmarks[i].y * img_h])

    p1, p2, p3, p4, p5, p6 = [pt(i) for i in eye_indices]
    vert_1 = np.linalg.norm(p2 - p6)
    vert_2 = np.linalg.norm(p3 - p5)
    horiz = np.linalg.norm(p1 - p4)
    return (vert_1 + vert_2) / (2.0 * horiz + 1e-8)


def eye_sharpness_single_face(
    full_gray: np.ndarray,
    landmarks,
    full_w: int,
    full_h: int,
) -> tuple[float, float]:
    """Per-eye Laplacian variance. Returns (near_eye, far_eye)."""
    results = {}
    for label, contour in [("left", LEFT_EYE_CONTOUR), ("right", RIGHT_EYE_CONTOUR)]:
        pts = np.array(
            [
                (int(landmarks[i].x * full_w), int(landmarks[i].y * full_h))
                for i in contour
            ]
        )
        x, y, w, h = cv2.boundingRect(pts)

        pad_x = int(w * EYE_BBOX_PAD_FRACTION)
        pad_y = int(h * EYE_BBOX_PAD_FRACTION)
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(full_w, x + w + pad_x)
        y2 = min(full_h, y + h + pad_y)

        crop = full_gray[y1:y2, x1:x2]
        if crop.size < 10:
            results[label] = 0.0
            continue
        results[label] = cv2.Laplacian(crop, cv2.CV_64F).var()

    near = max(results.values())
    far = min(results.values())
    return near, far


def face_bbox_area(landmarks, w: int, h: int) -> float:
    """Approximate face area from mesh landmarks."""
    pts = np.array([(landmarks[i].x * w, landmarks[i].y * h) for i in range(468)])
    _, _, bw, bh = cv2.boundingRect(pts.astype(np.int32))
    return bw * bh


def _default_face_result() -> dict:
    """Sentinel result for failed image analysis."""
    return {
        "has_face": False,
        "n_faces": 0,
        "subject_det_method": "error",
        "sharp_near_eye": 0.0,
        "sharp_far_eye": 0.0,
        "sharpness_subject": 0.0,
        "sharpness_background": 0.0,
        "isolation_ratio": 1.0,
        "raw_min_ear": 1.0,
        "eye_ratio_raw": 0.0,
        "blink_detected": False,
        "saliency_confidence": 0.0,
    }


def analyze_image(shm_index: int) -> dict:
    """
    Full face + sharpness analysis for one image.

    Face Mesh AND Laplacian sharpness both run on MID-RES.
    Eliminates full-res decode (134MB per worker on R5 Mark II).
    Laplacian variance at 2500px correlates >0.99 with full-res.
    """
    global _face_landmarker

    try:
        import mediapipe as mp

        # Decode mid-res from shared memory
        jpeg_bytes = _worker_get_jpeg_bytes(shm_index)
        img = ImageOps.exif_transpose(Image.open(io.BytesIO(jpeg_bytes)))
        ratio = MID_LONGEST_SIDE / max(img.size)
        if ratio < 1.0:
            img = img.resize(
                (int(img.size[0] * ratio), int(img.size[1] * ratio)),
                Image.LANCZOS,
            )
        mid_rgb = np.array(img)
        del img, jpeg_bytes

        # Run FaceLandmarker on mid-res
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=mid_rgb)
        result = _face_landmarker.detect(mp_image)

        if not result.face_landmarks:
            # No face — compute sharpness on mid-res gray
            mid_gray = cv2.cvtColor(mid_rgb, cv2.COLOR_RGB2GRAY)
            h, w = mid_gray.shape
            del mid_rgb, mp_image

            cx, cy = w // 2, h // 2
            rw, rh = w // 3, h // 3
            subject = mid_gray[cy - rh // 2 : cy + rh // 2, cx - rw // 2 : cx + rw // 2]
            subj_sharp = cv2.Laplacian(subject, cv2.CV_64F).var()
            bg_mask = np.ones_like(mid_gray, dtype=bool)
            bg_mask[cy - rh // 2 : cy + rh // 2, cx - rw // 2 : cx + rw // 2] = False
            bg_sharp = cv2.Laplacian(mid_gray, cv2.CV_64F)[bg_mask].var()
            isolation = subj_sharp / (bg_sharp + 1e-6)
            del mid_gray

            return {
                "has_face": False,
                "n_faces": 0,
                "subject_det_method": "center",
                "sharp_near_eye": subj_sharp,
                "sharp_far_eye": subj_sharp,
                "sharpness_subject": subj_sharp,
                "sharpness_background": bg_sharp,
                "isolation_ratio": isolation,
                "raw_min_ear": 1.0,
                "eye_ratio_raw": 0.0,
                "blink_detected": False,
                "saliency_confidence": 0.0,
            }

        # Face found — use mid-res gray for sharpness
        mid_gray = cv2.cvtColor(mid_rgb, cv2.COLOR_RGB2GRAY)
        mid_h, mid_w = mid_gray.shape
        del mid_rgb, mp_image

        all_faces = result.face_landmarks
        n_faces = len(all_faces)

        # Per-face analysis (landmarks are [0,1] — scale to mid-res)
        face_results = []
        all_ears = []

        for lm in all_faces:
            near, far = eye_sharpness_single_face(mid_gray, lm, mid_w, mid_h)
            area = face_bbox_area(lm, mid_w, mid_h)

            ear_l = compute_ear(lm, LEFT_EAR_POINTS, mid_w, mid_h)
            ear_r = compute_ear(lm, RIGHT_EAR_POINTS, mid_w, mid_h)
            min_ear = min(ear_l, ear_r)
            all_ears.append(min_ear)

            face_results.append(
                {"near": near, "far": far, "area": area, "min_ear": min_ear}
            )

        # Aggregate multi-face sharpness
        total_area = sum(f["area"] for f in face_results)
        max_area = max(f["area"] for f in face_results)

        weighted_near = sum(
            f["near"] * f["area"] / total_area for f in face_results
        )

        major_faces = [
            f
            for f in face_results
            if f["area"] > MAJOR_FACE_AREA_FRACTION * max_area
        ]
        min_major_near = min(f["near"] for f in major_faces) if major_faces else 0

        sharp_near = (
            MULTI_FACE_WEIGHTED_AVG_RATIO * weighted_near
            + MULTI_FACE_WORST_PENALTY_RATIO * min_major_near
        )
        sharp_far = sum(f["far"] * f["area"] / total_area for f in face_results)

        # Subject/background sharpness from largest face bbox
        largest_idx = max(range(n_faces), key=lambda i: face_results[i]["area"])
        largest_lm = all_faces[largest_idx]
        pts = np.array(
            [
                (int(largest_lm[i].x * mid_w), int(largest_lm[i].y * mid_h))
                for i in range(468)
            ]
        )
        fx, fy, fw, fh = cv2.boundingRect(pts.astype(np.int32))

        subj_sharp = cv2.Laplacian(
            mid_gray[fy : fy + fh, fx : fx + fw], cv2.CV_64F
        ).var()
        bg_mask = np.ones_like(mid_gray, dtype=bool)
        bg_mask[fy : fy + fh, fx : fx + fw] = False
        bg_sharp = cv2.Laplacian(mid_gray, cv2.CV_64F)[bg_mask].var()
        isolation = subj_sharp / (bg_sharp + 1e-6)

        del mid_gray

        frame_min_ear = min(all_ears)

        return {
            "has_face": True,
            "n_faces": n_faces,
            "subject_det_method": "face_mesh",
            "sharp_near_eye": sharp_near,
            "sharp_far_eye": sharp_far,
            "sharpness_subject": subj_sharp,
            "sharpness_background": bg_sharp,
            "isolation_ratio": isolation,
            "raw_min_ear": frame_min_ear,
            "eye_ratio_raw": (
                np.clip(sharp_near / (sharp_far + 1e-6) - 1.0, 0, 3.0) * 0.1
            ),
            "blink_detected": False,
            "saliency_confidence": 1.0,
        }

    except Exception as e:
        logger.warning(f"analyze_image failed for shm_index={shm_index}: {e}")
        return _default_face_result()


# ══════════════════════════════════════════════════════════════════
# 2E. Saliency-based Subject Detection
# ══════════════════════════════════════════════════════════════════


def saliency_confidence(saliency_map: np.ndarray) -> float:
    """
    Entropy of the CLS attention distribution as a confidence measure
    for subject localization.

    Returns confidence ∈ [0, 1]:
      ~0.0 = uniform attention (empty sky)
      ~0.3–0.5 = clear single subject
      ~0.6+ = dominant subject (frame-filling pet portrait)
    """
    flat = saliency_map.flatten().astype(np.float64)
    flat = flat / (flat.sum() + 1e-10)

    entropy = -np.sum(flat * np.log(flat + 1e-10))
    max_entropy = np.log(DINO_GRID_SIZE * DINO_GRID_SIZE)

    confidence = 1.0 - (entropy / max_entropy)
    return float(np.clip(confidence, 0.0, 1.0))


def _saliency_sharpness(
    full_gray: np.ndarray,
    saliency_map: np.ndarray,
    full_h: int,
    full_w: int,
) -> tuple[float, float, float, float]:
    """
    Compute sharpness using DINOv2 attention, with dual-path logic
    based on attention entropy.

    HIGH CONFIDENCE: hard threshold → binary subject mask.
    LOW CONFIDENCE: soft attention weighting across entire frame.

    Returns (subj_sharp, bg_sharp, isolation, confidence).
    """
    confidence = saliency_confidence(saliency_map)

    saliency_full = cv2.resize(
        saliency_map, (full_w, full_h), interpolation=cv2.INTER_LINEAR
    )

    laplacian = cv2.Laplacian(full_gray, cv2.CV_64F)

    if confidence > SALIENCY_CONFIDENCE_THRESHOLD:
        # HIGH CONFIDENCE: hard mask on subject region
        pct = max(5, int(30 - 40 * (confidence - 0.3)))
        threshold = np.percentile(saliency_full, 100 - pct)
        subject_mask = saliency_full >= threshold

        from localcull.constants import SALIENCY_TINY_MASK_GUARD

        if subject_mask.sum() < SALIENCY_TINY_MASK_GUARD * full_h * full_w:
            threshold = np.percentile(saliency_full, 60)
            subject_mask = saliency_full >= threshold

        subj_sharp = laplacian[subject_mask].var()
        bg_mask = ~subject_mask
        bg_sharp = laplacian[bg_mask].var() if bg_mask.sum() > 0 else 0.0

    else:
        # LOW CONFIDENCE: soft attention-weighted sharpness
        laplacian_sq = laplacian**2

        weights = saliency_full.astype(np.float64)
        weights = weights / (weights.sum() + 1e-10)

        weighted_mean = np.sum(weights * laplacian)
        weighted_sq_mean = np.sum(weights * laplacian_sq)
        subj_sharp = weighted_sq_mean - weighted_mean**2

        inv_weights = 1.0 - (saliency_full / (saliency_full.max() + 1e-10))
        inv_weights = inv_weights / (inv_weights.sum() + 1e-10)
        bg_weighted_mean = np.sum(inv_weights * laplacian)
        bg_weighted_sq_mean = np.sum(inv_weights * laplacian_sq)
        bg_sharp = bg_weighted_sq_mean - bg_weighted_mean**2

    isolation = subj_sharp / (bg_sharp + 1e-6)
    return subj_sharp, bg_sharp, isolation, confidence


def _fallback_sharpness(
    shm_index: int, saliency_map: np.ndarray | None = None
) -> dict:
    """
    Tiered fallback when no face is detected.
    If saliency_map available, use confidence-aware dual-path analysis.
    Otherwise fall back to center-crop (legacy).

    Uses MID-RES for Laplacian to avoid 134MB full-res decode per worker.
    """
    try:
        jpeg_bytes = _worker_get_jpeg_bytes(shm_index)
        img = ImageOps.exif_transpose(Image.open(io.BytesIO(jpeg_bytes)))
        ratio = MID_LONGEST_SIDE / max(img.size)
        if ratio < 1.0:
            img = img.resize(
                (int(img.size[0] * ratio), int(img.size[1] * ratio)),
                Image.LANCZOS,
            )
        mid_rgb = np.array(img)
        del img, jpeg_bytes
        mid_gray = cv2.cvtColor(mid_rgb, cv2.COLOR_RGB2GRAY)
        h, w = mid_gray.shape
        del mid_rgb

        if saliency_map is not None:
            subj_sharp, bg_sharp, isolation, conf = _saliency_sharpness(
                mid_gray, saliency_map, h, w
            )
            det_method = "dino_attn"
        else:
            # Legacy: center 33% of frame
            cx, cy = w // 2, h // 2
            rw, rh = w // 3, h // 3
            subject = mid_gray[cy - rh // 2 : cy + rh // 2, cx - rw // 2 : cx + rw // 2]
            subj_sharp = cv2.Laplacian(subject, cv2.CV_64F).var()
            bg_mask = np.ones_like(mid_gray, dtype=bool)
            bg_mask[cy - rh // 2 : cy + rh // 2, cx - rw // 2 : cx + rw // 2] = False
            bg_sharp = cv2.Laplacian(mid_gray, cv2.CV_64F)[bg_mask].var()
            isolation = subj_sharp / (bg_sharp + 1e-6)
            conf = 0.0
            det_method = "center"

        del mid_gray

        return {
            "has_face": False,
            "n_faces": 0,
            "subject_det_method": det_method,
            "sharp_near_eye": subj_sharp,
            "sharp_far_eye": subj_sharp,
            "sharpness_subject": subj_sharp,
            "sharpness_background": bg_sharp,
            "isolation_ratio": isolation,
            "raw_min_ear": 1.0,
            "eye_ratio_raw": 0.0,
            "blink_detected": False,
            "saliency_confidence": conf,
        }
    except Exception as e:
        logger.warning(f"_fallback_sharpness failed for shm_index={shm_index}: {e}")
        return _default_face_result()


# ══════════════════════════════════════════════════════════════════
# Blink calibration + burst harmonization
# ══════════════════════════════════════════════════════════════════


def calibrate_blinks_per_burst(frames: list[Frame]):
    """
    Per-burst blink threshold using the burst's own EAR distribution.

    Fixed threshold (0.18) has demographic bias. Solution: blink =
    EAR dropped below 50% of this subject's open-eye baseline.
    """
    burst_ears = defaultdict(list)
    for f in frames:
        if f.has_face:
            burst_ears[f.burst_id].append((f, f.raw_min_ear))

    for burst_id, frame_ears in burst_ears.items():
        ears = [ear for _, ear in frame_ears]

        if len(ears) == 1:
            frame, ear = frame_ears[0]
            frame.blink_detected = ear < EAR_ABSOLUTE_FLOOR
            continue

        max_ear = max(ears)

        if max_ear < EAR_SQUINT_SUSPECT and len(ears) <= 3:
            for frame, ear in frame_ears:
                frame.blink_detected = ear < EAR_ABSOLUTE_FLOOR
        else:
            threshold = max_ear * EAR_CALIBRATION_FACTOR
            for frame, ear in frame_ears:
                frame.blink_detected = ear < threshold


def harmonize_mixed_bursts(frames: list[Frame]):
    """
    If a burst has 20–80% non-face frames, force all to center-crop
    sharpness for consistent within-burst ranking.
    """
    bursts = defaultdict(list)
    for f in frames:
        bursts[f.burst_id].append(f)

    for burst_id, burst_frames in bursts.items():
        valid = [f for f in burst_frames if f.subject_det_method != "error"]
        if len(valid) < 2:
            continue

        n_no_face = sum(1 for f in valid if not f.has_face)
        non_face_frac = n_no_face / len(valid)

        if HARMONIZE_MIN_FACE_RATE < non_face_frac < HARMONIZE_MAX_FACE_RATE:
            for f in valid:
                if f.has_face:
                    f._pre_harmonize_sharp_near = f.sharp_near_eye
                    f._pre_harmonize_sharp_far = f.sharp_far_eye
                    f._pre_harmonize_eye_ratio = f.eye_ratio_raw
                    f.sharp_near_eye = f.sharpness_subject
                    f.sharp_far_eye = f.sharpness_subject
                    f.eye_ratio_raw = 0.0


def _analyze_wrapper(shm_index: int) -> dict:
    """Wrapper for ProcessPoolExecutor.map."""
    return analyze_image(shm_index)


def _saliency_wrapper(args: tuple) -> dict:
    """Wrapper for saliency sharpness in ProcessPoolExecutor."""
    shm_index, saliency_map = args
    return _fallback_sharpness(shm_index, saliency_map=saliency_map)


def _init_saliency_worker(shm_name: str):
    """
    Lightweight initializer for saliency sharpness workers.
    Only maps shared memory — does NOT load FaceLandmarker.
    """
    import atexit

    cv2.setNumThreads(1)
    global _worker_shm
    _worker_shm = shm.SharedMemory(name=shm_name)

    def _cleanup():
        try:
            _worker_shm.close()
        except Exception:
            pass
    atexit.register(_cleanup)


# ══════════════════════════════════════════════════════════════════
# 2F. Stage 2 Orchestration
# ══════════════════════════════════════════════════════════════════


def run_face_analysis(frames: list[Frame], full_store):
    """
    Parallel face analysis with per-worker initializer.
    Workers decode mid-res from shared memory for face detection
    and Laplacian sharpness (no full-res decode).
    """
    from localcull.constants import check_memory
    check_memory(f"face analysis ({FACE_ANALYSIS_WORKERS} workers)")

    model_path = _ensure_face_model()
    args = [f.full_shm_index for f in frames]

    with ProcessPoolExecutor(
        max_workers=FACE_ANALYSIS_WORKERS,
        initializer=_init_face_worker,
        initargs=(full_store.name, model_path),
    ) as pool:
        results = list(
            tqdm(
                pool.map(_analyze_wrapper, args),
                total=len(args),
                desc="Face analysis + sharpness",
            )
        )

    for f, r in zip(frames, results):
        for k, v in r.items():
            if hasattr(f, k):
                setattr(f, k, v)

    harmonize_mixed_bursts(frames)
    calibrate_blinks_per_burst(frames)


def _run_saliency_sharpness_pass(frames: list[Frame], full_store):
    """
    Phase 2: replace center-crop sharpness with confidence-aware
    saliency analysis for non-face frames.

    Only processes frames where subject_det_method == 'center'.
    Cost: ~100ms per frame, 28-way parallel.
    """
    non_face = [
        (f.full_shm_index, f.global_index)
        for f in frames
        if f.subject_det_method == "center"
    ]

    if not non_face:
        return

    from localcull.constants import check_memory
    check_memory("saliency sharpness pass")

    logger.info(f"Saliency sharpness pass: {len(non_face)} non-face frames")

    saliency_args = [
        (shm_idx, frames[global_idx].saliency_map)
        for shm_idx, global_idx in non_face
    ]

    with ProcessPoolExecutor(
        max_workers=FACE_ANALYSIS_WORKERS,
        initializer=_init_saliency_worker,
        initargs=(full_store.name,),
    ) as pool:
        results = list(pool.map(_saliency_wrapper, saliency_args))

    for (_, global_idx), result in zip(non_face, results):
        f = frames[global_idx]
        for k, v in result.items():
            if hasattr(f, k):
                setattr(f, k, v)


def _save_mid_arrays_to_disk(mid_arrays: list[np.ndarray], shoot_id: str) -> str:
    """
    Save mid_arrays to a temporary numpy file for GPU subprocesses.

    If all arrays have the same shape (typical: same camera, same
    orientation), saves as a single stacked .npy that subprocesses
    can mmap with zero copy.

    If shapes vary (mixed orientation/camera), saves as .npz.

    Returns path to the temp file.
    """
    import tempfile

    tmpdir = tempfile.gettempdir()
    path = os.path.join(tmpdir, f"localcull_{shoot_id}_mid.npy")

    shapes = {arr.shape for arr in mid_arrays}
    if len(shapes) == 1:
        # All same shape — write incrementally to memmap (zero temp copy)
        shape = (len(mid_arrays),) + mid_arrays[0].shape
        fp = np.lib.format.open_memmap(
            path, mode="w+", dtype=mid_arrays[0].dtype, shape=shape
        )
        for i, arr in enumerate(mid_arrays):
            fp[i] = arr
        del fp
    else:
        # Mixed shapes — save individually (no mmap, but avoids re-decode)
        path = os.path.join(tmpdir, f"localcull_{shoot_id}_mid.npz")
        np.savez(path, *mid_arrays)

    return path


def _load_mid_arrays_from_disk(path: str) -> list[np.ndarray]:
    """
    Load mid_arrays from disk. Uses mmap for .npy (zero-copy),
    regular load for .npz.
    """
    if path.endswith(".npz"):
        data = np.load(path)
        return [data[k] for k in sorted(data.files, key=lambda x: int(x.replace("arr_", "")))]
    else:
        stacked = np.load(path, mmap_mode="r")
        return [stacked[i] for i in range(len(stacked))]


def _gpu_subprocess_worker(model_name, mid_path, shoot_id, data_hash):
    """
    Run a single GPU model in an isolated subprocess.

    Loads mid_arrays from disk (mmap when possible), runs inference,
    saves checkpoint, then EXITS — forcing OS to reclaim all
    Metal/MPS/MLX GPU buffer allocations.
    """
    import warnings

    os.environ["TRANSFORMERS_VERBOSITY"] = "error"
    warnings.filterwarnings("ignore", message=".*do_sample.*")
    warnings.filterwarnings("ignore", message=".*temperature.*")
    warnings.filterwarnings("ignore", message=".*top_p.*")
    warnings.filterwarnings("ignore", message=".*layer_idx.*")
    warnings.filterwarnings("ignore", message=".*use_fast.*")

    mid_arrays = _load_mid_arrays_from_disk(mid_path)

    if model_name == "dinov2":
        result = dinov2_embed_batch(mid_arrays)
    else:
        # All other models dispatch through the scorer registry
        from localcull.scorers import SCORER_REGISTRY

        if model_name not in SCORER_REGISTRY:
            raise ValueError(
                f"Unknown model: {model_name}. "
                f"Available: dinov2, {', '.join(sorted(SCORER_REGISTRY.keys()))}"
            )
        spec = SCORER_REGISTRY[model_name]
        result = spec.score_fn(mid_arrays)

    save_checkpoint(shoot_id, model_name, result, data_hash)


def _run_gpu_model(model_name, mid_path, shoot_id, data_hash):
    """
    Run GPU model with subprocess isolation for Metal memory safety.

    Returns cached checkpoint if available, otherwise spawns a
    subprocess that runs inference + saves checkpoint, then exits
    (freeing all Metal allocations). Parent reads checkpoint from disk.
    """
    import multiprocessing

    cached = load_checkpoint(shoot_id, model_name, data_hash)
    if cached is not None:
        return cached

    logger.info(f"Running {model_name} in isolated subprocess")
    p = multiprocessing.Process(
        target=_gpu_subprocess_worker,
        args=(model_name, mid_path, shoot_id, data_hash),
    )
    p.start()
    p.join()

    if p.exitcode != 0:
        raise RuntimeError(
            f"GPU subprocess for {model_name} failed "
            f"(exit code {p.exitcode})"
        )

    result = load_checkpoint(shoot_id, model_name, data_hash)
    if result is None:
        raise RuntimeError(
            f"GPU subprocess for {model_name} completed but "
            f"no checkpoint found"
        )
    return result


def run_stage2(
    frames: list[Frame],
    mid_arrays: list[np.ndarray],
    full_store,
    shoot_id: str,
    data_hash: str = "",
    enabled_scorers: list[str] | None = None,
) -> np.ndarray:
    """
    Orchestrate Stage 2: face analysis + GPU scoring + saliency.

    Memory lifecycle (330 images):
      1. Save mid_arrays to disk temp file (~6GB write, ~2s)
      2. Free mid_arrays from parent → parent drops by ~6GB
      3. Face analysis reads from full_store shm (doesn't need mid_arrays)
      4. Each GPU subprocess mmaps mid_arrays from disk (zero-copy)
         + loads one model → runs inference → exits (OS reclaims all Metal)
      5. Parent reads results from checkpoint files on disk

    Peak memory: max(face_analysis_peak, single_gpu_model_peak)
    instead of: face + all_gpu_models stacked.

    Args:
        enabled_scorers: list of scorer names to run, or None for defaults.
            Special value ["all"] runs everything in the registry.

    Returns DINOv2 embeddings array.
    """
    import gc

    from localcull.scorers import SCORER_REGISTRY, get_enabled_scorers

    scorers = get_enabled_scorers(enabled_scorers)
    scorer_names = [s.name for s in scorers]
    logger.info(
        f"Stage 2: Feature extraction "
        f"(scorers: {', '.join(s.display_name for s in scorers)})"
    )

    # ── Save mid_arrays to disk for GPU subprocesses ──
    mid_path = _save_mid_arrays_to_disk(mid_arrays, shoot_id)
    logger.info(f"Mid-res arrays saved to {mid_path}")

    # ── Free mid_arrays from parent process ──
    # Face analysis reads from full_store shm, not mid_arrays.
    # GPU subprocesses will read from disk file.
    for f in frames:
        f.mid_array = None
    mid_arrays.clear()
    gc.collect()
    logger.info("Freed mid_arrays from parent (GPU reads from disk)")

    # ── Phase 1: Face analysis (CPU parallel) ──
    logger.info("Phase 1: Face analysis (CPU, %d workers)", FACE_ANALYSIS_WORKERS)
    run_face_analysis(frames, full_store)
    gc.collect()
    logger.info("Face analysis complete")

    # ── Phase 2: GPU scoring (each model in isolated subprocess) ──
    logger.info("Phase 2: GPU scoring (subprocess-isolated)")

    # DINOv2 always runs (needed for clustering)
    dinov2_result = _run_gpu_model(
        "dinov2", mid_path, shoot_id, data_hash
    )
    embeddings, saliency_list = dinov2_result
    logger.info(
        f"DINOv2 embeddings: shape={embeddings.shape} "
        f"(expected dim=3072 for diversity, 1024 means CLS-only fallback)"
    )

    # Run all enabled scorers sequentially
    scorer_results: dict[str, np.ndarray] = {}
    failed_scorers: list[str] = []
    for spec in scorers:
        try:
            scorer_results[spec.name] = _run_gpu_model(
                spec.name, mid_path, shoot_id, data_hash
            )
        except RuntimeError as e:
            logger.warning(
                f"Scorer {spec.display_name} failed, skipping: {e}"
            )
            failed_scorers.append(spec.name)

    if failed_scorers:
        logger.warning(
            f"Skipped {len(failed_scorers)} scorer(s): "
            f"{', '.join(failed_scorers)}. "
            f"Pipeline continues with remaining scorers."
        )

    # Clean up temp file
    try:
        os.remove(mid_path)
    except OSError:
        pass

    # Apply GPU results to frames
    for i, f in enumerate(frames):
        f.dinov2_embedding = embeddings[i]
        f.saliency_map = saliency_list[i]
        if f.saliency_map is not None:
            f.saliency_confidence = saliency_confidence(f.saliency_map)
        else:
            f.saliency_confidence = 0.0

        # Dynamically assign scorer results to frame fields
        for spec in scorers:
            if spec.name in scorer_results:
                scores = scorer_results[spec.name]
                setattr(f, spec.frame_field, float(scores[i]))

    del scorer_results, saliency_list
    gc.collect()

    logger.info("GPU scoring complete")

    # ── Phase 3: Saliency sharpness for non-face frames ──
    _run_saliency_sharpness_pass(frames, full_store)

    logger.info("Stage 2 complete")
    return embeddings