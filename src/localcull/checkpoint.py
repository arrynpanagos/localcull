"""
Checkpoint save/load for crash recovery.

GPU inference is the longest phase. Checkpoints ensure a crash
during MUSIQ scoring doesn't lose completed TOPIQ and DINOv2
results.

Resume limitation: shared memory (Stage 1's CompressedImageStore)
is volatile — it does not survive crashes. On resume, Stage 1
always re-executes to rebuild shared memory + mid-res arrays (~45s).
GPU stages skip via checkpoint hits. Net resume cost: ~45s fixed
overhead + only the incomplete GPU model.
"""

import hashlib
import logging
import os
import pickle

from localcull.constants import CHECKPOINT_DIR, VERSION_HASH

logger = logging.getLogger(__name__)


def compute_data_hash(sorted_paths: list[str]) -> str:
    """Hash sorted path list to detect reordering between crash and resume."""
    content = "\n".join(sorted_paths)
    return hashlib.md5(content.encode()).hexdigest()[:8]


def save_checkpoint(shoot_id: str, stage: str, data, data_hash: str = ""):
    """Save intermediate results for crash recovery."""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    path = os.path.join(
        CHECKPOINT_DIR,
        f"{shoot_id}_{stage}_{VERSION_HASH}_{data_hash}.pkl",
    )
    with open(path, "wb") as f:
        pickle.dump(data, f)
    logger.debug(f"Checkpoint saved: {path}")


def load_checkpoint(shoot_id: str, stage: str, data_hash: str = ""):
    """Load checkpoint if it exists and version matches. Returns None if miss."""
    path = os.path.join(
        CHECKPOINT_DIR,
        f"{shoot_id}_{stage}_{VERSION_HASH}_{data_hash}.pkl",
    )
    if os.path.exists(path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        logger.info(f"Checkpoint hit: {stage}")
        return data
    return None


def clear_cache(shoot_id: str = None):
    """
    Clear checkpoint cache. If shoot_id given, only clear that shoot's
    checkpoints. Otherwise, clear all.
    """
    if not os.path.exists(CHECKPOINT_DIR):
        return
    for fname in os.listdir(CHECKPOINT_DIR):
        if shoot_id is None or fname.startswith(f"{shoot_id}_"):
            path = os.path.join(CHECKPOINT_DIR, fname)
            os.remove(path)
            logger.info(f"Cleared checkpoint: {fname}")
