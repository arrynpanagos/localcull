#!/usr/bin/env python3
"""
One-time vendor script for mPLUG-Owl2 model code.

Downloads custom modeling code from HuggingFace repos and applies
compatibility patches for transformers >= 4.45. Run once, commit the
result, and localcull uses vendored code directly — no trust_remote_code,
no cache patching, no runtime hacks.

Usage:
    python scripts/vendor_mplug_owl2.py

Creates:
    src/localcull/vendor/__init__.py
    src/localcull/vendor/mplug_owl2_qalign/   (from q-future/one-align)
    src/localcull/vendor/mplug_owl2_deqa/      (from zhiyuanyou/DeQA-Score-Mix3)
"""

import re
import sys
import urllib.request
from pathlib import Path

# ──────────────────────────────────────────────────────────────
# Repos and files to vendor
# ──────────────────────────────────────────────────────────────

REPOS = {
    "mplug_owl2_qalign": {
        "repo_id": "q-future/one-align",
        "files": [
            "configuration_mplug_owl2.py",
            "modeling_attn_mask_utils.py",
            "modeling_llama2.py",
            "modeling_mplug_owl2.py",
            "visual_encoder.py",
        ],
    },
    "mplug_owl2_deqa": {
        "repo_id": "zhiyuanyou/DeQA-Score-Mix3",
        "files": [
            "configuration_mplug_owl2.py",
            "modeling_attn_mask_utils.py",
            "modeling_llama2.py",
            "modeling_mplug_owl2_huggingface.py",
            "visual_encoder.py",
        ],
    },
}

# ──────────────────────────────────────────────────────────────
# Compatibility shim inserted into modeling_llama2.py
# ──────────────────────────────────────────────────────────────
#
# The original file does:
#   from transformers.models.llama.modeling_llama import *
#
# In transformers >= 4.45, that wildcard import brings in new-API
# versions of LlamaRotaryEmbedding (takes config instead of dim),
# and doesn't export Cache, logger, etc. The code below shadows
# those symbols with old-API implementations that match what the
# rest of modeling_llama2.py expects.

COMPAT_SHIM = '''
# ═══ localcull compatibility shim for transformers >= 4.45 ═══
# The wildcard import above brings in new-API symbols that are
# incompatible with this file's code. We shadow them here.

from transformers.cache_utils import Cache
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

logger = logging.get_logger(__name__)


class LlamaRotaryEmbedding(nn.Module):
    """Old-API rotary embeddings (transformers <= 4.44 style).
    New API takes a config object; this file passes dim, max_position_embeddings."""
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.scaling_factor = scaling_factor
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._set_cos_sin_cache(seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype())

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        t = t / self.scaling_factor
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        if seq_len is not None and seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)
        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


class LlamaLinearScalingRotaryEmbedding(LlamaRotaryEmbedding):
    pass


class LlamaDynamicNTKScalingRotaryEmbedding(LlamaRotaryEmbedding):
    def _set_cos_sin_cache(self, seq_len, device, dtype):
        if seq_len > self.max_position_embeddings:
            base = self.base * (
                (self.scaling_factor * seq_len / self.max_position_embeddings)
                - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))
            self.register_buffer("inv_freq", inv_freq, persistent=False)
        super()._set_cos_sin_cache(seq_len, device, dtype)


def repeat_kv(hidden_states, n_rep):
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)  # [bs, 1, seq_len, dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


try:
    from transformers.models.llama.modeling_llama import LlamaRMSNorm
except ImportError:
    class LlamaRMSNorm(nn.Module):
        def __init__(self, hidden_size, eps=1e-6):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.variance_epsilon = eps
        def forward(self, hidden_states):
            variance = hidden_states.pow(2).mean(-1, keepdim=True)
            hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
            return self.weight * hidden_states

class LlamaMLP(nn.Module):
    """Old-API LlamaMLP (transformers <= 4.44 style).
    New API reads config.mlp_bias; old mPLUG-Owl2 configs don't have that."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj

# ═══ end localcull compatibility shim ═══

'''


# ──────────────────────────────────────────────────────────────
# Patch functions
# ──────────────────────────────────────────────────────────────

def patch_modeling_llama2(content: str) -> str:
    """Insert compatibility shim after imports, before class definitions.
    Also fix removed attributes from newer transformers."""
    # The anchor: last import line before the first class definition.
    # Both repos have: from .configuration_mplug_owl2 import LlamaConfig
    # right before class MultiwayNetwork or other class definitions.
    anchor = "from .configuration_mplug_owl2 import LlamaConfig"
    if anchor in content:
        idx = content.index(anchor)
        line_end = content.index("\n", idx)
        content = content[:line_end + 1] + COMPAT_SHIM + content[line_end + 1:]
    else:
        print(f"    WARNING: anchor not found, prepending shim")
        content = COMPAT_SHIM + content

    # In transformers >= 4.45, _use_flash_attention_2 and _use_sdpa were
    # removed from PreTrainedModel. The old code checks these attributes
    # in the forward() method. Replace with safe getattr (always False
    # since we use attn_implementation="eager").
    content = content.replace(
        "self._use_flash_attention_2",
        'getattr(self, "_use_flash_attention_2", False)',
    )
    content = content.replace(
        "self._use_sdpa",
        'getattr(self, "_use_sdpa", False)',
    )

    return content


def patch_visual_encoder(content: str) -> str:
    """Fix find_pruneable → find_prunable for transformers >= 5.0."""
    # Check which name the installed transformers uses
    old = "find_pruneable_heads_and_indices"
    new = "find_prunable_heads_and_indices"

    try:
        from transformers.pytorch_utils import find_pruneable_heads_and_indices  # noqa: F401
        return content  # old name still works, no patch needed
    except ImportError:
        pass

    if old in content:
        content = content.replace(old, new)
        print(f"    Fixed {old} → {new}")
    return content


# ──────────────────────────────────────────────────────────────
# Download helper
# ──────────────────────────────────────────────────────────────

def download_raw(repo_id: str, filename: str) -> str:
    """Download a file directly from HuggingFace (bypasses HF cache)."""
    url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
    try:
        resp = urllib.request.urlopen(url)
        return resp.read().decode("utf-8")
    except Exception as e:
        print(f"    ERROR downloading {url}: {e}")
        sys.exit(1)


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    project_root = Path(__file__).resolve().parent.parent
    vendor_root = project_root / "src" / "localcull" / "vendor"

    # Create vendor root
    vendor_root.mkdir(parents=True, exist_ok=True)
    (vendor_root / "__init__.py").write_text("# Vendored model code\n")

    for pkg_name, repo_info in REPOS.items():
        repo_id = repo_info["repo_id"]
        pkg_dir = vendor_root / pkg_name

        print(f"\n{'=' * 60}")
        print(f"  Vendoring: {repo_id}")
        print(f"  Target:    {pkg_dir}")
        print(f"{'=' * 60}")

        pkg_dir.mkdir(parents=True, exist_ok=True)

        for filename in repo_info["files"]:
            print(f"  Downloading {filename}...")
            content = download_raw(repo_id, filename)

            # Apply patches
            patched = False
            if filename == "modeling_llama2.py":
                content = patch_modeling_llama2(content)
                patched = True
            elif filename == "visual_encoder.py":
                orig = content
                content = patch_visual_encoder(content)
                patched = content != orig

            (pkg_dir / filename).write_text(content)
            status = "PATCHED" if patched else "ok"
            print(f"    ✓ {filename} [{status}] ({len(content)} bytes)")

        # Write __init__.py that exports the model class
        (pkg_dir / "__init__.py").write_text(
            f"# Vendored from {repo_id}\n"
        )
        print(f"    ✓ __init__.py")

    # Summary
    print(f"\n{'=' * 60}")
    print(f"  Vendor complete!")
    print(f"  Files written to: {vendor_root}/")
    print()
    print("  Next steps:")
    print("  1. Clean corrupted HF cache:")
    print("     rm -rf ~/.cache/huggingface/modules/transformers_modules/q-future")
    print("     rm -rf ~/.cache/huggingface/modules/transformers_modules/zhiyuanyou")
    print("  2. Run localcull as normal")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()