"""
Pluggable scorer registry for aesthetic / quality models.

Each scorer is a function:  (mid_arrays: list[np.ndarray]) -> np.ndarray
Returns a 1-D array of scores, one per image.

Models are loaded one at a time in isolated subprocesses (Metal memory safety).
The registry stores metadata for each scorer so the pipeline can dispatch,
checkpoint, and report results generically.

Available scorers:
  pyiqa-based (existing):
    qalign       — Q-Align 7B MLLM, [1,5] MOS, pyiqa (2023)
    qualiclip    — QualiCLIP+ self-supervised CLIP, [0,1], pyiqa (2025)
    topiq        — TOPIQ ResNet50 NR-IQA, [0,1], pyiqa (2023)
    musiq        — MUSIQ multi-scale transformer, [0,100], pyiqa (2021)

  HuggingFace-based (new frontier):
    deqa_score   — DeQA-Score distribution-based MLLM, [1,5], CVPR 2025
    q_scorer     — Q-Scorer MLP regression MLLM, [1,5], AAAI 2025
    q_insight    — Q-Insight RL-based Qwen2.5-VL-7B, [1,5], NeurIPS 2025
"""

from __future__ import annotations

import gc
import logging
import os
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════
# Registry data structure
# ══════════════════════════════════════════════════════════════════


@dataclass
class ScorerSpec:
    """Metadata for a registered scorer."""

    name: str  # unique key, used in checkpoints and CLI
    display_name: str  # human-readable name for logging
    score_fn: Callable[[list[np.ndarray]], np.ndarray]  # inference function
    output_range: tuple[float, float]  # (min, max) of output scores
    frame_field: str = ""  # Frame attribute to store scores (e.g. "qalign_score")
    batch_size: int = 1  # default batch size
    is_aesthetic: bool = True  # True=aesthetic, False=technical
    default_enabled: bool = False  # included in default run?
    include_in_consensus: bool = True  # included in z-score consensus ranking?
    requires: str = "pyiqa"  # "pyiqa" | "transformers" | "vllm"
    model_id: str = ""  # HuggingFace model ID or pyiqa key
    description: str = ""


# Global registry — populated at module level below
SCORER_REGISTRY: dict[str, ScorerSpec] = {}


def register_scorer(spec: ScorerSpec):
    """Register a scorer in the global registry."""
    SCORER_REGISTRY[spec.name] = spec


def get_enabled_scorers(
    requested: list[str] | None = None,
) -> list[ScorerSpec]:
    """
    Get list of scorers to run.

    If requested is None, returns all default_enabled scorers.
    If requested is a list of names, returns those specific scorers.
    Special value "all" enables everything.
    """
    if requested is None:
        return [s for s in SCORER_REGISTRY.values() if s.default_enabled]
    if requested == ["all"]:
        return list(SCORER_REGISTRY.values())
    result = []
    for name in requested:
        if name not in SCORER_REGISTRY:
            available = ", ".join(sorted(SCORER_REGISTRY.keys()))
            raise ValueError(
                f"Unknown scorer '{name}'. Available: {available}"
            )
        result.append(SCORER_REGISTRY[name])
    return result


# ══════════════════════════════════════════════════════════════════
# pyiqa-based scorers (existing)
# ══════════════════════════════════════════════════════════════════


def score_topiq(mid_arrays: list[np.ndarray]) -> np.ndarray:
    """TOPIQ NR-IQA scoring on MPS. ResNet50 backbone, [0,1] output."""
    import pyiqa
    import torch
    from torchvision import transforms

    from localcull.constants import TOPIQ_BATCH_SIZE, TOPIQ_SIZE

    topiq = pyiqa.create_metric("topiq_nr", device="mps")
    topiq_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(TOPIQ_SIZE),
        transforms.ToTensor(),
    ])

    all_scores = []
    for i in tqdm(range(0, len(mid_arrays), TOPIQ_BATCH_SIZE), desc="TOPIQ"):
        batch_imgs = mid_arrays[i : i + TOPIQ_BATCH_SIZE]
        batch = torch.stack(
            [topiq_transform(img) for img in batch_imgs]
        ).to("mps")
        with torch.no_grad():
            scores = topiq(batch)
        all_scores.append(scores.cpu())
        del batch
    result = torch.cat(all_scores).squeeze().numpy()

    del topiq, all_scores
    gc.collect()
    torch.mps.empty_cache()

    logger.info(
        f"[TOPIQ] Complete: {len(result)} images, "
        f"mean={result.mean():.3f}, std={result.std():.3f}, "
        f"range=[{result.min():.3f}, {result.max():.3f}]"
    )
    return result


def score_musiq(mid_arrays: list[np.ndarray]) -> np.ndarray:
    """MUSIQ multi-scale quality scoring on MPS. [0,100] output."""
    import pyiqa
    import torch
    from torchvision import transforms

    musiq = pyiqa.create_metric("musiq", device="mps")
    musiq_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
    ])

    all_scores = []
    for i in tqdm(range(len(mid_arrays)), desc="MUSIQ"):
        img = mid_arrays[i]
        tensor = musiq_transform(img).unsqueeze(0).to("mps")
        with torch.no_grad():
            score = musiq(tensor)
        all_scores.append(score.cpu().item())
        del tensor
    result = np.array(all_scores, dtype=np.float32)

    del musiq, all_scores
    gc.collect()
    torch.mps.empty_cache()

    logger.info(
        f"[MUSIQ] Complete: {len(result)} images, "
        f"mean={result.mean():.3f}, std={result.std():.3f}, "
        f"range=[{result.min():.3f}, {result.max():.3f}]"
    )
    return result


def score_qalign(mid_arrays: list[np.ndarray]) -> np.ndarray:
    """
    Q-Align aesthetic scoring on MPS.

    7B MLLM (mPLUG-Owl2) that predicts quality on [1, 5] MOS scale.
    Uses vendored model code (src/localcull/vendor/mplug_owl2_qalign/).

    Weights: q-future/one-align (HuggingFace, same backbone as DeQA-Score).
    """
    import torch
    from PIL import Image

    from localcull.constants import QALIGN_BATCH_SIZE, check_memory
    from localcull.vendor.mplug_owl2_qalign.modeling_mplug_owl2 import (
        MPLUGOwl2LlamaForCausalLM,
    )

    check_memory("Q-Align model load")

    logger.info(
        f"[Q-Align] Loading model (q-future/one-align), "
        f"batch_size={QALIGN_BATCH_SIZE}..."
    )
    model = MPLUGOwl2LlamaForCausalLM.from_pretrained(
        "q-future/one-align",
        attn_implementation="eager",
        torch_dtype=torch.float16,
        device_map="mps",
    )
    model.eval()

    all_scores = []
    n = len(mid_arrays)
    for start in tqdm(range(0, n, QALIGN_BATCH_SIZE), desc="Q-Align"):
        end = min(start + QALIGN_BATCH_SIZE, n)
        batch_pil = [Image.fromarray(mid_arrays[i]) for i in range(start, end)]
        with torch.no_grad():
            scores = model.score(batch_pil, task_="quality", input_="image")
        if hasattr(scores, "tolist"):
            all_scores.extend([float(s) for s in scores.tolist()])
        elif isinstance(scores, (list, tuple)):
            all_scores.extend([float(s) for s in scores])
        else:
            all_scores.append(float(scores))
        del batch_pil

    result = np.array(all_scores, dtype=np.float32)

    del model, all_scores
    gc.collect()
    if hasattr(torch, "mps"):
        torch.mps.empty_cache()

    logger.info(
        f"[Q-Align] Complete: {len(result)} images, "
        f"mean={result.mean():.3f}, std={result.std():.3f}, "
        f"range=[{result.min():.3f}, {result.max():.3f}]"
    )
    return result


def score_qualiclip(mid_arrays: list[np.ndarray]) -> np.ndarray:
    """
    QualiCLIP+ aesthetic scoring on MPS.

    Self-supervised CLIP model — good generalization to professional
    photography since it doesn't rely on biased human opinion labels.
    [0, 1] output.
    """
    import pyiqa
    import torch
    from torchvision import transforms

    from localcull.constants import QUALICLIP_BATCH_SIZE, check_memory

    check_memory("QualiCLIP+ model load")

    qualiclip = pyiqa.create_metric("qualiclip+", device="mps")
    qualiclip_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    all_scores = []
    for i in tqdm(
        range(0, len(mid_arrays), QUALICLIP_BATCH_SIZE), desc="QualiCLIP+"
    ):
        batch_imgs = mid_arrays[i : i + QUALICLIP_BATCH_SIZE]
        batch = torch.stack(
            [qualiclip_transform(img) for img in batch_imgs]
        ).to("mps")
        with torch.no_grad():
            scores = qualiclip(batch)
        all_scores.append(scores.cpu())
        del batch
    result = torch.cat(all_scores).squeeze().numpy()

    del qualiclip, all_scores
    gc.collect()
    torch.mps.empty_cache()

    logger.info(
        f"[QualiCLIP+] Complete: {len(result)} images, "
        f"mean={result.mean():.3f}, std={result.std():.3f}, "
        f"range=[{result.min():.3f}, {result.max():.3f}]"
    )
    return result


def score_nima(mid_arrays: list[np.ndarray]) -> np.ndarray:
    """
    NIMA aesthetic scoring on MPS.

    Trained on AVA dataset (255K photos rated by photography enthusiasts
    on aesthetic appeal, NOT technical quality). InceptionResNetV2 backbone.
    Output is mean of predicted distribution over [1, 10].

    This is the only scorer trained on genuine aesthetic preference data
    rather than KonIQ-10k technical quality MOS.
    """
    import pyiqa
    import torch
    from torchvision import transforms

    from localcull.constants import NIMA_BATCH_SIZE, check_memory

    check_memory("NIMA model load")

    nima = pyiqa.create_metric("nima", device="mps")
    nima_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    all_scores = []
    for i in tqdm(range(0, len(mid_arrays), NIMA_BATCH_SIZE), desc="NIMA"):
        batch_imgs = mid_arrays[i : i + NIMA_BATCH_SIZE]
        batch = torch.stack(
            [nima_transform(img) for img in batch_imgs]
        ).to("mps")
        with torch.no_grad():
            scores = nima(batch)
        all_scores.append(scores.cpu())
        del batch
    result = torch.cat(all_scores).squeeze().numpy()

    del nima, all_scores
    gc.collect()
    torch.mps.empty_cache()

    logger.info(
        f"[NIMA] Complete: {len(result)} images, "
        f"mean={result.mean():.3f}, std={result.std():.3f}, "
        f"range=[{result.min():.3f}, {result.max():.3f}]"
    )
    return result


# ══════════════════════════════════════════════════════════════════
# HuggingFace-based scorers (frontier models)
# ══════════════════════════════════════════════════════════════════


def score_deqa(mid_arrays: list[np.ndarray]) -> np.ndarray:
    """
    DeQA-Score aesthetic scoring on MPS.

    CVPR 2025. Distribution-based soft-label training on mPLUG-Owl2.
    Fixes Q-Align's discretization error — predicts continuous quality
    distributions rather than 5-bucket one-hot, yielding much finer
    score granularity (solves star compression).

    Output: [1, 5] MOS scale.
    Uses vendored model code (src/localcull/vendor/mplug_owl2_deqa/).
    Weights: zhiyuanyou/DeQA-Score-Mix3 (HuggingFace).
    """
    import torch
    from PIL import Image

    from localcull.constants import DEQA_BATCH_SIZE, check_memory
    from localcull.vendor.mplug_owl2_deqa.modeling_mplug_owl2_huggingface import (
        MPLUGOwl2LlamaForCausalLM,
    )

    check_memory("DeQA-Score model load")

    logger.info(
        f"[DeQA-Score] Loading model (zhiyuanyou/DeQA-Score-Mix3), "
        f"batch_size={DEQA_BATCH_SIZE}..."
    )
    model = MPLUGOwl2LlamaForCausalLM.from_pretrained(
        "zhiyuanyou/DeQA-Score-Mix3",
        attn_implementation="eager",
        torch_dtype=torch.float16,
        device_map="mps",
    )
    model.eval()

    all_scores = []
    n = len(mid_arrays)
    for start in tqdm(range(0, n, DEQA_BATCH_SIZE), desc="DeQA-Score"):
        end = min(start + DEQA_BATCH_SIZE, n)
        batch_pil = [Image.fromarray(mid_arrays[i]) for i in range(start, end)]
        with torch.no_grad():
            scores = model.score(batch_pil)
        # model.score() returns a tensor for batched input
        if hasattr(scores, "tolist"):
            all_scores.extend([float(s) for s in scores.tolist()])
        elif isinstance(scores, (list, tuple)):
            all_scores.extend([float(s) for s in scores])
        else:
            all_scores.append(float(scores))
        del batch_pil

    result = np.array(all_scores, dtype=np.float32)

    del model, all_scores
    gc.collect()
    if hasattr(torch, "mps"):
        torch.mps.empty_cache()

    logger.info(
        f"[DeQA-Score] Complete: {len(result)} images, "
        f"mean={result.mean():.3f}, std={result.std():.3f}, "
        f"range=[{result.min():.3f}, {result.max():.3f}]"
    )
    return result


def score_q_scorer(mid_arrays: list[np.ndarray]) -> np.ndarray:
    """
    Q-Scorer aesthetic scoring on MPS.

    AAAI 2025. Fixes Q-Align's token-to-score conversion errors with
    IQA-specific score tokens + MLP regression module. Same mPLUG-Owl2
    backbone but more accurate continuous score prediction.

    Output: [1, 5] MOS scale. ~1.3s/image (same backbone as Q-Align).

    NOTE: Q-Scorer weights may need to be loaded from the paper's
    official repo. Check: https://github.com/TangZhenchen/Q-Scorer
    If unavailable, falls back to the paper's LoRA variant on mPLUG-Owl2.

    Install:
        pip install transformers torch
        # Clone Q-Scorer repo for architecture code if needed
    """
    import torch
    from PIL import Image

    from localcull.constants import check_memory

    check_memory("Q-Scorer model load")

    # Q-Scorer weights are not publicly available (TangZhenchen/Q-Scorer = 404)
    # AAAI 2026 paper only — no code or weights released as of March 2026.
    try:
        from transformers import AutoModelForCausalLM

        logger.info("[Q-Scorer] Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            "TangZhenchen/Q-Scorer",  # expected HF repo
            trust_remote_code=True,
            attn_implementation="eager",
            torch_dtype=torch.float16,
            device_map="mps",
        )
        model.eval()
    except Exception as e:
        raise RuntimeError(
            f"[Q-Scorer] Could not load model: {e}\n"
            "Q-Scorer weights may not be publicly available yet.\n"
            "Check: https://github.com/TangZhenchen/Q-Scorer\n"
            "or https://arxiv.org/abs/2511.07812 for release status."
        ) from e

    all_scores = []
    for i in tqdm(range(len(mid_arrays)), desc="Q-Scorer"):
        pil_img = Image.fromarray(mid_arrays[i])
        with torch.no_grad():
            score = model.score([pil_img])
        if hasattr(score, "item"):
            all_scores.append(float(score.item()))
        elif isinstance(score, (list, tuple)):
            all_scores.append(float(score[0]))
        else:
            all_scores.append(float(score))
        del pil_img

    result = np.array(all_scores, dtype=np.float32)

    del model, all_scores
    gc.collect()
    if hasattr(torch, "mps"):
        torch.mps.empty_cache()

    logger.info(
        f"[Q-Scorer] Complete: {len(result)} images, "
        f"mean={result.mean():.3f}, std={result.std():.3f}, "
        f"range=[{result.min():.3f}, {result.max():.3f}]"
    )
    return result


def score_q_insight(mid_arrays: list[np.ndarray]) -> np.ndarray:
    """
    Q-Insight quality scoring on MPS.

    NeurIPS 2025 Spotlight. RL-based (GRPO) model built on Qwen2.5-VL-7B.
    Stronger OOD generalization than DeQA-Score. Generative model that
    produces reasoning + JSON score in <think>/<answer> tags.

    Output: [1, 5] MOS scale. Generative (~2-4s/image depending on reasoning).
    Weights: ByteDance/Q-Insight (subfolder: score_degradation)

    Install:
        pip install transformers==4.51.3 qwen-vl-utils torch
    """
    import json
    import re

    import torch
    from PIL import Image

    from localcull.constants import Q_INSIGHT_BATCH_SIZE, check_memory

    check_memory("Q-Insight model load")

    try:
        from transformers import (
            GenerationConfig,
            Qwen2_5_VLForConditionalGeneration,
            AutoProcessor,
        )
        from qwen_vl_utils import process_vision_info

        model_id = "ByteDance/Q-Insight"
        subfolder = "score_degradation"

        logger.info(
            f"[Q-Insight] Loading {model_id} (subfolder={subfolder}), "
            f"batch_size={Q_INSIGHT_BATCH_SIZE}..."
        )
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",  # MPS: no flash_attention_2
            device_map="mps",
            subfolder=subfolder,
        )
        processor = AutoProcessor.from_pretrained(
            model_id, subfolder=subfolder,
            # Limit visual tokens for speed. Each 28x28 patch = 1 token.
            # Default max_pixels = 1280*28*28 → 1280 tokens → ~40s/image on MPS.
            # 256*28*28 ≈ 448x448 → 256 tokens → ~6-8s/image.
            # Our images are already mid-res (~800px); IQA doesn't need full res.
            min_pixels=128 * 28 * 28,
            max_pixels=256 * 28 * 28,
        )
        processor.tokenizer.padding_side = "left"  # for batched generation
        model.eval()
    except Exception as e:
        raise RuntimeError(
            f"[Q-Insight] Could not load model: {e}\n"
            "Install: pip install transformers==4.51.3 qwen-vl-utils\n"
            "Repo: https://github.com/bytedance/Q-Insight\n"
            "Weights: https://huggingface.co/ByteDance/Q-Insight"
        ) from e

    # Score-only prompts — skip <think> reasoning for speed.
    # The official demo uses a CoT system prompt that generates 100-500
    # tokens of reasoning before the score, costing ~45s/image on MPS.
    # By asking for the answer directly we get ~2-3s/image instead.
    SYSTEM_PROMPT = (
        "You are an image quality assessment expert. "
        "Respond with ONLY a JSON object, no explanation."
    )
    SCORE_PROMPT = (
        "Rate the overall quality of this image as a float between 1 and 5, "
        "rounded to two decimal places (1=very poor, 5=excellent). "
        'Respond with only: {"rating": <score>}'
    )

    gen_config = GenerationConfig(
        do_sample=False,          # greedy — deterministic + faster
        max_new_tokens=32,        # {"rating": 3.45} is ~10 tokens
    )

    all_scores = []
    parse_failures = 0
    n = len(mid_arrays)
    n_batches = (n + Q_INSIGHT_BATCH_SIZE - 1) // Q_INSIGHT_BATCH_SIZE

    for start in tqdm(range(0, n, Q_INSIGHT_BATCH_SIZE), desc="Q-Insight",
                      total=n_batches):
        end = min(start + Q_INSIGHT_BATCH_SIZE, n)
        batch_pil = [Image.fromarray(mid_arrays[i]) for i in range(start, end)]

        # Build one message per image (all share same system prompt + score prompt)
        batch_messages = []
        for pil_img in batch_pil:
            msg = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": SYSTEM_PROMPT}],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": SCORE_PROMPT},
                        {"type": "image", "image": pil_img},
                    ],
                },
            ]
            batch_messages.append(msg)

        # Tokenize batch
        batch_text = [
            processor.apply_chat_template(
                msg, tokenize=False, add_generation_prompt=True
            )
            for msg in batch_messages
        ]
        batch_image_inputs = []
        for msg in batch_messages:
            img_inputs, _ = process_vision_info([msg])
            batch_image_inputs.extend(img_inputs)

        inputs = processor(
            text=batch_text,
            images=batch_image_inputs,
            padding=True,
            return_tensors="pt",
        ).to("mps")

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                generation_config=gen_config,
                use_cache=True,
            )

        # Trim prompts and decode each response
        for j, (in_ids, out_ids) in enumerate(
            zip(inputs.input_ids, output_ids)
        ):
            trimmed = out_ids[len(in_ids):]
            output_text = processor.batch_decode(
                [trimmed],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]

            score = _extract_q_insight_score(output_text, default=3.0)
            if score == 3.0 and "rating" not in output_text.lower():
                parse_failures += 1
            all_scores.append(score)

        del batch_pil, inputs, output_ids

    result = np.array(all_scores, dtype=np.float32)

    del model, processor, all_scores
    gc.collect()
    if hasattr(torch, "mps"):
        torch.mps.empty_cache()

    logger.info(
        f"[Q-Insight] Complete: {len(result)} images, "
        f"mean={result.mean():.3f}, std={result.std():.3f}, "
        f"range=[{result.min():.3f}, {result.max():.3f}]"
        + (f", {parse_failures} parse failures" if parse_failures else "")
    )
    return result


def _extract_q_insight_score(text: str, default: float = 3.0) -> float:
    """
    Extract rating from Q-Insight output.

    Expected format: {"rating": X.XX}
    Also handles <answer>{"rating": X.XX}</answer> (CoT mode).
    Falls back to scanning for any number in [1, 5] range.
    """
    import json
    import re

    # Try direct JSON parse first (score-only mode output)
    text_stripped = text.strip()
    try:
        data = json.loads(text_stripped)
        if isinstance(data, dict) and "rating" in data:
            val = float(data["rating"])
            return max(1.0, min(5.0, val))
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    # Try to extract from <answer> tags (CoT mode output)
    answer_match = re.search(
        r"<answer>\s*(.*?)\s*</answer>", text, re.DOTALL
    )
    if answer_match:
        answer_text = answer_match.group(1).strip()
        # Try JSON parse
        try:
            data = json.loads(answer_text)
            if isinstance(data, dict) and "rating" in data:
                val = float(data["rating"])
                return max(1.0, min(5.0, val))
        except (json.JSONDecodeError, ValueError, TypeError):
            pass
        # Fallback: find number in answer block
        nums = re.findall(r"(\d+\.?\d*)", answer_text)
        for n in nums:
            val = float(n)
            if 0.5 <= val <= 5.5:
                return max(1.0, min(5.0, val))

    # Fallback: scan whole text for reasonable score
    matches = re.findall(r"(\d+\.?\d*)", text)
    for m in matches:
        val = float(m)
        if 0.5 <= val <= 5.5:
            return max(1.0, min(5.0, val))

    logger.warning(f"[Q-Insight parse] No score found in: {text[:200]!r}")
    return default


def _extract_score(text: str, default: float = 5.0, max_score: float = 10.0) -> float:
    """Extract a numeric score from VLM text response.

    Handles formats like:
        "7.5"
        "Score: 7.5"
        "The aesthetic score is 7.5 out of 10"
        "Aesthetics score : 85" (0-100 scale, will be clamped to max_score)
    """
    import re

    if not text or not text.strip():
        return default

    text = text.strip()

    # Try to find "score" followed by a number
    patterns = [
        r"(?:score|rating)\s*[:=]?\s*(\d+\.?\d*)",
        r"(\d+\.?\d*)\s*(?:out of|/)\s*\d+",
        r"^(\d+\.?\d*)\s*$",  # Just a number
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            val = float(match.group(1))
            # If value > max_score, assume 0-100 scale
            if val > max_score:
                val = val / (100.0 / max_score)
            return max(0.0, min(max_score, val))

    # Fallback: find any reasonable number
    matches = re.findall(r"(\d+\.?\d*)", text)
    for m in matches:
        val = float(m)
        if 0 <= val <= max_score:
            return val
        elif val <= 100:
            return val / (100.0 / max_score)

    logger.warning(f"[_extract_score] No score found in: {text[:200]!r}")
    return default


# ══════════════════════════════════════════════════════════════════
# ArtiMuse — artistic aesthetic scoring (InternVL-3 7B, CVPR 2026)
# ══════════════════════════════════════════════════════════════════


def score_artimuse(mid_arrays: list[np.ndarray]) -> np.ndarray:
    """
    ArtiMuse artistic aesthetic scoring (AVA photography checkpoint).

    InternVL-3 7B fine-tuned on AVA (250K photography competition images
    rated by photographers). Uses Token-as-Score: single forward pass
    extracts logit probabilities over number tokens for continuous scores.

    Output range: [0, 10].
    Weights: Thunderbolt215215/ArtiMuse_AVA (HuggingFace), Apache 2.0
    """
    import math
    import torch
    import torchvision.transforms as T
    from PIL import Image
    from torchvision.transforms.functional import InterpolationMode
    from transformers import AutoModel, AutoTokenizer

    from localcull.constants import ARTIMUSE_BATCH_SIZE, check_memory

    check_memory("ArtiMuse model load")

    # ── InternVL dynamic tiling preprocessing ──
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)
    IMG_SIZE = 448

    def build_transform():
        return T.Compose([
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((IMG_SIZE, IMG_SIZE), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height):
        best_ratio_diff = float("inf")
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * IMG_SIZE * IMG_SIZE * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio

    def dynamic_preprocess(image, min_num=1, max_num=6, use_thumbnail=True):
        """Split image into InternVL-style 448x448 tiles."""
        orig_w, orig_h = image.size
        aspect = orig_w / orig_h

        target_ratios = set()
        for n in range(min_num, max_num + 1):
            for i in range(1, n + 1):
                for j in range(1, n + 1):
                    if i * j <= max_num and i * j >= min_num:
                        target_ratios.add((i, j))
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        best_ratio = find_closest_aspect_ratio(
            aspect, target_ratios, orig_w, orig_h
        )

        target_w = best_ratio[0] * IMG_SIZE
        target_h = best_ratio[1] * IMG_SIZE
        blocks = best_ratio[0] * best_ratio[1]

        resized = image.resize((target_w, target_h), Image.LANCZOS)
        processed_images = []
        for i in range(best_ratio[1]):
            for j in range(best_ratio[0]):
                box = (
                    j * IMG_SIZE, i * IMG_SIZE,
                    (j + 1) * IMG_SIZE, (i + 1) * IMG_SIZE,
                )
                processed_images.append(resized.crop(box))

        if use_thumbnail and blocks > 1:
            thumbnail = image.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
            processed_images.append(thumbnail)

        return processed_images

    def preprocess_image(pil_img, max_num=6):
        """Full InternVL preprocessing: dynamic tile + transform + stack."""
        transform = build_transform()
        tiles = dynamic_preprocess(pil_img, max_num=max_num, use_thumbnail=True)
        pixel_values = torch.stack([transform(t) for t in tiles])
        return pixel_values.to(torch.float16).to("mps")

    # ── Load model (AVA photography checkpoint) ──
    model_name = "Thunderbolt215215/ArtiMuse_AVA"
    logger.info(f"[ArtiMuse] Loading model ({model_name})...")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True, use_fast=False
    )
    model = AutoModel.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).eval().to("mps")

    # ── Build Token-as-Score lookup ──
    # Map number tokens to their token IDs for logit extraction
    score_values = list(range(0, 11))  # 0 through 10
    score_token_ids = []
    for v in score_values:
        token_str = str(v)
        ids = tokenizer.encode(token_str, add_special_tokens=False)
        # Use first token ID (handles multi-digit as single token)
        score_token_ids.append(ids[0])
    score_token_ids = torch.tensor(score_token_ids, device="mps")
    score_weights = torch.tensor(score_values, dtype=torch.float32, device="mps")

    logger.info(
        f"[ArtiMuse] Token-as-Score mapped: "
        f"{dict(zip(score_values, score_token_ids.tolist()))}"
    )

    # ── Prepare prompt template (same as model.chat internals) ──
    # Import conversation template from model's custom code
    try:
        import sys
        # The model's conversation.py was loaded via trust_remote_code
        conv_module = None
        for mod_name, mod in sys.modules.items():
            if "conversation" in mod_name and hasattr(mod, "get_conv_template"):
                conv_module = mod
                break
        if conv_module is None:
            raise ImportError("conversation module not found")
        get_conv_template = conv_module.get_conv_template
    except Exception as e:
        logger.warning(f"[ArtiMuse] Could not import conversation template: {e}")
        logger.warning("[ArtiMuse] Falling back to text generation mode")
        get_conv_template = None

    score_prompt = "Rate the overall aesthetic quality of this photograph on a scale from 0 to 10."
    IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"
    IMG_START_TOKEN = "<img>"
    IMG_END_TOKEN = "</img>"

    img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    model.img_context_token_id = img_context_token_id

    all_scores = []
    n = len(mid_arrays)
    for i in tqdm(range(n), desc="ArtiMuse"):
        pil_img = Image.fromarray(mid_arrays[i])
        try:
            with torch.no_grad():
                pixel_values = preprocess_image(pil_img, max_num=6)
                num_patches = pixel_values.shape[0]

                if get_conv_template is not None:
                    # ── Token-as-Score: single forward pass ──
                    # Build prompt with image tokens (same as model.chat)
                    question = "<image>\n" + score_prompt
                    template = get_conv_template(model.template)
                    template.system_message = model.system_message
                    template.append_message(template.roles[0], question)
                    template.append_message(template.roles[1], None)
                    query = template.get_prompt()

                    # Replace <image> with actual image token sequence
                    image_tokens = (
                        IMG_START_TOKEN
                        + IMG_CONTEXT_TOKEN * model.num_image_token * num_patches
                        + IMG_END_TOKEN
                    )
                    query = query.replace("<image>", image_tokens, 1)

                    model_inputs = tokenizer(query, return_tensors="pt")
                    input_ids = model_inputs["input_ids"].to("mps")
                    attention_mask = model_inputs["attention_mask"].to("mps")

                    # Forward pass (not generate) to get logits
                    # image_flags required by InternVL forward()
                    image_flags = torch.ones(
                        num_patches, dtype=torch.long, device="mps"
                    )
                    outputs = model(
                        pixel_values=pixel_values,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        image_flags=image_flags,
                    )

                    # Extract logits at last position for score tokens
                    last_logits = outputs.logits[0, -1, :]
                    number_logits = last_logits[score_token_ids]
                    probs = torch.softmax(number_logits, dim=0)
                    score = (probs * score_weights).sum().item()

                    all_scores.append(score)
                else:
                    # Fallback: generate text and parse
                    generation_config = dict(max_new_tokens=10, do_sample=False)
                    response = model.chat(
                        tokenizer, pixel_values, score_prompt, generation_config
                    )
                    score = _extract_score(response, default=5.0)
                    all_scores.append(score)

                if i < 5:
                    logger.info(
                        f"[ArtiMuse] Image {i}: score={all_scores[-1]:.3f}"
                    )
        except Exception as e:
            logger.warning(f"[ArtiMuse] Failed on image {i}: {e}")
            all_scores.append(0.0)
        del pil_img

    result = np.array(all_scores, dtype=np.float32)

    del model, tokenizer, all_scores
    gc.collect()
    if hasattr(torch, "mps"):
        torch.mps.empty_cache()

    logger.info(
        f"[ArtiMuse] Complete: {len(result)} images, "
        f"mean={result.mean():.3f}, std={result.std():.3f}, "
        f"range=[{result.min():.3f}, {result.max():.3f}]"
    )
    return result

    result = np.array(all_scores, dtype=np.float32)

    del model, tokenizer, all_scores
    gc.collect()
    if hasattr(torch, "mps"):
        torch.mps.empty_cache()

    logger.info(
        f"[ArtiMuse] Complete: {len(result)} images, "
        f"mean={result.mean():.3f}, std={result.std():.3f}, "
        f"range=[{result.min():.3f}, {result.max():.3f}]"
    )
    return result

    result = np.array(all_scores, dtype=np.float32)

    del model, tokenizer, all_scores
    gc.collect()
    if hasattr(torch, "mps"):
        torch.mps.empty_cache()

    logger.info(
        f"[ArtiMuse] Complete: {len(result)} images, "
        f"mean={result.mean():.3f}, std={result.std():.3f}, "
        f"range=[{result.min():.3f}, {result.max():.3f}]"
    )
    return result


# ══════════════════════════════════════════════════════════════════
# UniPercept — unified perceptual scoring (InternVL 7B, 2025)
# ══════════════════════════════════════════════════════════════════


def score_unipercept(mid_arrays: list[np.ndarray]) -> np.ndarray:
    """
    UniPercept unified perceptual scoring.

    InternVL 7B fine-tuned across IAA (aesthetics), IQA (quality),
    and ISTA (structure/texture). Extends ArtiMuse with broader
    perceptual coverage.

    pip package: `pip install unipercept-reward`
    Returns IAA (aesthetics) score. Also has IQA and ISTA.
    Output range: [0, 100] (scaled to [0, 10] for consistency).

    Weights: Thunderbolt215215/UniPercept (HuggingFace, gated — request access)
    """
    import tempfile
    import torch
    from PIL import Image

    from localcull.constants import UNIPERCEPT_BATCH_SIZE, check_memory

    check_memory("UniPercept model load")

    logger.info("[UniPercept] Loading model...")

    # Use pip package (requires: pip install unipercept-reward)
    try:
        from unipercept_reward import UniPerceptRewardInferencer

        inferencer = UniPerceptRewardInferencer(device="mps")
        logger.info("[UniPercept] Loaded via unipercept-reward package")

        all_scores = []
        n = len(mid_arrays)

        # UniPercept takes file paths, so we save to temp files
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in tqdm(range(n), desc="UniPercept"):
                pil_img = Image.fromarray(mid_arrays[i])
                tmp_path = os.path.join(tmpdir, f"img_{i:04d}.jpg")
                pil_img.save(tmp_path, quality=95)
                del pil_img

                try:
                    rewards = inferencer.reward(image_paths=[tmp_path])
                    if rewards and rewards[0]:
                        # Use IAA (aesthetics) score, scale from 0-100 to 0-10
                        iaa = rewards[0].get("iaa", 50.0)
                        all_scores.append(float(iaa) / 10.0)
                    else:
                        all_scores.append(0.0)
                except Exception as e:
                    logger.warning(f"[UniPercept] Failed on image {i}: {e}")
                    all_scores.append(0.0)

                # Clean up temp file to save disk
                os.unlink(tmp_path)

        del inferencer

    except ImportError:
        logger.error(
            "[UniPercept] unipercept-reward package not found. "
            "Install with: pip install unipercept-reward"
        )
        logger.error(
            "[UniPercept] Also requires access approval at "
            "https://huggingface.co/Thunderbolt215215/UniPercept "
            "and huggingface-cli login"
        )
        return np.zeros(len(mid_arrays), dtype=np.float32)

    result = np.array(all_scores, dtype=np.float32)

    del all_scores
    gc.collect()
    if hasattr(torch, "mps"):
        torch.mps.empty_cache()

    logger.info(
        f"[UniPercept] Complete: {len(result)} images, "
        f"mean={result.mean():.3f}, std={result.std():.3f}, "
        f"range=[{result.min():.3f}, {result.max():.3f}]"
    )
    return result


# ══════════════════════════════════════════════════════════════════
# Register all scorers
# ══════════════════════════════════════════════════════════════════

from localcull.constants import (
    TOPIQ_BATCH_SIZE,
    MUSIQ_BATCH_SIZE,
    QUALICLIP_BATCH_SIZE,
    QALIGN_BATCH_SIZE,
    DEQA_BATCH_SIZE,
    Q_SCORER_BATCH_SIZE,
    Q_INSIGHT_BATCH_SIZE,
    NIMA_BATCH_SIZE,
    ARTIMUSE_BATCH_SIZE,
    UNIPERCEPT_BATCH_SIZE,
)

register_scorer(ScorerSpec(
    name="topiq",
    display_name="TOPIQ",
    score_fn=score_topiq,
    output_range=(0.0, 1.0),
    frame_field="topiq_score",
    batch_size=TOPIQ_BATCH_SIZE,
    is_aesthetic=False,
    default_enabled=True,
    requires="pyiqa",
    model_id="topiq_nr",
    description="ResNet50 NR-IQA, semantics→distortions, [0,1] (2023)",
))

register_scorer(ScorerSpec(
    name="musiq",
    display_name="MUSIQ",
    score_fn=score_musiq,
    output_range=(0.0, 100.0),
    frame_field="musiq_score",
    batch_size=MUSIQ_BATCH_SIZE,
    is_aesthetic=False,
    default_enabled=True,
    requires="pyiqa",
    model_id="musiq",
    description="Multi-scale transformer NR-IQA, [0,100] (2021)",
))

register_scorer(ScorerSpec(
    name="qalign",
    display_name="Q-Align",
    score_fn=score_qalign,
    output_range=(1.0, 5.0),
    frame_field="qalign_score",
    batch_size=QALIGN_BATCH_SIZE,
    is_aesthetic=True,
    default_enabled=True,
    requires="transformers",
    model_id="q-future/one-align",
    description="Discrete-level MLLM (mPLUG-Owl2), MOS [1,5], ICML 2024",
))

register_scorer(ScorerSpec(
    name="qualiclip",
    display_name="QualiCLIP+",
    score_fn=score_qualiclip,
    output_range=(0.0, 1.0),
    frame_field="qualiclip_score",
    batch_size=QUALICLIP_BATCH_SIZE,
    is_aesthetic=False,
    default_enabled=True,
    requires="pyiqa",
    model_id="qualiclip+",
    description="Self-supervised CLIP NR-IQA, [0,1] (2025)",
))

register_scorer(ScorerSpec(
    name="nima",
    display_name="NIMA",
    score_fn=score_nima,
    output_range=(1.0, 10.0),
    frame_field="nima_score",
    batch_size=NIMA_BATCH_SIZE,
    is_aesthetic=True,
    default_enabled=True,
    include_in_consensus=False,  # metadata only — AVA bias doesn't match pro preferences
    requires="pyiqa",
    model_id="nima",
    description="AVA-trained aesthetic predictor, InceptionResNetV2, [1,10] (2018)",
))

register_scorer(ScorerSpec(
    name="deqa_score",
    display_name="DeQA-Score",
    score_fn=score_deqa,
    output_range=(1.0, 5.0),
    frame_field="deqa_score",
    batch_size=DEQA_BATCH_SIZE,
    is_aesthetic=True,
    default_enabled=True,
    requires="transformers",
    model_id="zhiyuanyou/DeQA-Score-Mix3",
    description="Distribution-based MLLM, MOS [1,5], CVPR 2025",
))

register_scorer(ScorerSpec(
    name="q_scorer",
    display_name="Q-Scorer",
    score_fn=score_q_scorer,
    output_range=(1.0, 5.0),
    frame_field="q_scorer_score",
    batch_size=Q_SCORER_BATCH_SIZE,
    is_aesthetic=True,
    default_enabled=False,  # TangZhenchen/Q-Scorer repo does not exist on HuggingFace
    requires="transformers",
    model_id="TangZhenchen/Q-Scorer",
    description="MLP-regression MLLM, MOS [1,5], AAAI 2025",
))

register_scorer(ScorerSpec(
    name="q_insight",
    display_name="Q-Insight",
    score_fn=score_q_insight,
    output_range=(1.0, 5.0),
    frame_field="q_insight_score",
    batch_size=Q_INSIGHT_BATCH_SIZE,
    is_aesthetic=True,
    default_enabled=False,  # ~40s/image on MPS (1280 visual tokens, no flash attn). Enable with --scorers
    requires="transformers>=4.45,qwen-vl-utils",
    model_id="ByteDance/Q-Insight",
    description="RL-based Qwen2.5-VL-7B (subfolder: score_degradation), generative MOS [1,5], NeurIPS 2025 Spotlight",
))

register_scorer(ScorerSpec(
    name="artimuse",
    display_name="ArtiMuse",
    score_fn=score_artimuse,
    output_range=(0.0, 10.0),
    frame_field="artimuse_score",
    batch_size=ARTIMUSE_BATCH_SIZE,
    is_aesthetic=True,
    default_enabled=False,
    include_in_consensus=False,  # evaluate independently first, then decide weight
    requires="transformers",
    model_id="Thunderbolt215215/ArtiMuse_AVA",
    description="Photography aesthetic scorer (AVA dataset), InternVL-3 7B, Token-as-Score, Apache 2.0",
))

register_scorer(ScorerSpec(
    name="unipercept",
    display_name="UniPercept",
    score_fn=score_unipercept,
    output_range=(0.0, 10.0),
    frame_field="unipercept_score",
    batch_size=UNIPERCEPT_BATCH_SIZE,
    is_aesthetic=True,
    default_enabled=False,  # gated repo — requires HuggingFace access approval
    include_in_consensus=False,  # evaluate independently first, then decide weight
    requires="unipercept-reward",
    model_id="Thunderbolt215215/UniPercept",
    description="Unified perceptual scorer (IAA+IQA+ISTA), InternVL 7B, 2025",
))