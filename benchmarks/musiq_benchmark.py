"""
MUSIQ Batch Benchmark (Pre-Implementation Checklist Item #4)

Compares ranking between:
  - Unbatched (batch_size=1, native aspect ratio) — current pipeline default
  - Batched (batch_size=32, resized to 384×384) — 4 minutes faster

If Spearman ρ > 0.98, adopt batching.

Usage:
    python benchmarks/musiq_benchmark.py /path/to/folder/of/cr3s
    python benchmarks/musiq_benchmark.py /path/to/folder/of/jpgs  # also works
"""

import argparse
import glob
import io
import os
import subprocess
import sys
import tempfile
import time

import numpy as np
from PIL import Image, ImageOps
from scipy.stats import spearmanr


def load_mid_res(path: str, longest_side: int = 2500) -> np.ndarray:
    img = ImageOps.exif_transpose(Image.open(path))
    ratio = longest_side / max(img.size)
    if ratio < 1.0:
        img = img.resize(
            (int(img.size[0] * ratio), int(img.size[1] * ratio)),
            Image.LANCZOS,
        )
    return np.array(img)


def extract_jpegs_from_cr3s(cr3_paths: list[str]) -> list[str]:
    """Extract embedded JPEGs, return paths to the extracted files."""
    # Check which ones need extracting
    need_extract = []
    already_done = []
    for p in cr3_paths:
        emb = p.rsplit(".", 1)[0] + "_embedded.jpg"
        if os.path.exists(emb) and os.path.getsize(emb) > 10_000:
            already_done.append(emb)
        else:
            need_extract.append(p)

    if already_done:
        print(f"  {len(already_done)} embedded JPEGs already extracted, reusing")

    if need_extract:
        print(f"  Extracting {len(need_extract)} embedded JPEGs...")
        subprocess.run(
            ["exiftool", "-b", "-JpgFromRaw", "-w", "%d%f_embedded.jpg"] + need_extract,
            check=False,  # some may fail
        )

    jpeg_paths = []
    for p in cr3_paths:
        emb = p.rsplit(".", 1)[0] + "_embedded.jpg"
        if os.path.exists(emb) and os.path.getsize(emb) > 10_000:
            jpeg_paths.append(emb)
        else:
            print(f"  WARNING: No embedded JPEG for {os.path.basename(p)}, skipping")
    return jpeg_paths


def score_unbatched(mid_arrays: list[np.ndarray]) -> np.ndarray:
    """MUSIQ at batch_size=1, native aspect ratio (current pipeline default)."""
    import pyiqa
    import torch
    from torchvision import transforms

    musiq = pyiqa.create_metric("musiq", device="mps")
    to_tensor = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])

    scores = []
    for img in mid_arrays:
        tensor = to_tensor(img).unsqueeze(0).to("mps")
        with torch.no_grad():
            s = musiq(tensor)
        scores.append(s.cpu().item())
    return np.array(scores)


def score_batched(mid_arrays: list[np.ndarray], batch_size: int = 32) -> np.ndarray:
    """MUSIQ at batch_size=32, resized to uniform 384×384."""
    import pyiqa
    import torch
    from torchvision import transforms

    musiq = pyiqa.create_metric("musiq", device="mps")
    to_tensor = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
    ])

    tensors = [to_tensor(img) for img in mid_arrays]
    scores = []
    for i in range(0, len(tensors), batch_size):
        batch = torch.stack(tensors[i : i + batch_size]).to("mps")
        with torch.no_grad():
            s = musiq(batch)
        scores.append(s.cpu())
    return torch.cat(scores).squeeze().numpy()


def main():
    parser = argparse.ArgumentParser(description="MUSIQ batch benchmark")
    parser.add_argument("input_dir", help="Folder of CR3 or JPEG files")
    args = parser.parse_args()

    # Find images
    cr3s = sorted(
        glob.glob(os.path.join(args.input_dir, "*.CR3"))
        + glob.glob(os.path.join(args.input_dir, "*.cr3"))
    )
    jpgs = sorted(
        glob.glob(os.path.join(args.input_dir, "*.jpg"))
        + glob.glob(os.path.join(args.input_dir, "*.jpeg"))
        + glob.glob(os.path.join(args.input_dir, "*.JPG"))
        + glob.glob(os.path.join(args.input_dir, "*.JPEG"))
    )

    if cr3s:
        jpeg_paths = extract_jpegs_from_cr3s(cr3s)
    elif jpgs:
        jpeg_paths = jpgs
    else:
        print("No CR3 or JPEG files found.")
        sys.exit(1)

    print(f"\n{len(jpeg_paths)} images loaded")

    # Load mid-res
    print("Decoding mid-res variants...")
    mid_arrays = [load_mid_res(p) for p in jpeg_paths]
    aspects = [f"{img.shape[1]}x{img.shape[0]}" for img in mid_arrays]
    print(f"  Aspect ratios: {set(aspects)}")

    # Unbatched (native)
    print("\nScoring unbatched (native aspect, batch_size=1)...")
    t0 = time.time()
    scores_native = score_unbatched(mid_arrays)
    t_native = time.time() - t0

    # Batched (384×384)
    print("Scoring batched (384×384, batch_size=32)...")
    t0 = time.time()
    scores_batched = score_batched(mid_arrays)
    t_batched = time.time() - t0

    # Compare
    rho, pvalue = spearmanr(scores_native, scores_batched)

    print(f"\n{'='*50}")
    print(f"MUSIQ Batch Benchmark Results")
    print(f"{'='*50}")
    print(f"Images:          {len(mid_arrays)}")
    print(f"Unbatched time:  {t_native:.1f}s ({t_native/len(mid_arrays)*1000:.0f}ms/img)")
    print(f"Batched time:    {t_batched:.1f}s ({t_batched/len(mid_arrays)*1000:.0f}ms/img)")
    print(f"Speedup:         {t_native/t_batched:.1f}x")
    print(f"Spearman ρ:      {rho:.4f} (p={pvalue:.2e})")
    print(f"{'='*50}")

    if rho > 0.98:
        print(f"✅ ρ > 0.98 — ADOPT BATCHING. Save ~{(t_native-t_batched)/len(mid_arrays)*3000:.0f}s on 3000 images.")
    elif rho > 0.95:
        print(f"⚠️  ρ = {rho:.4f} — borderline. Inspect rank disagreements before deciding.")
    else:
        print(f"❌ ρ < 0.95 — KEEP UNBATCHED. Aspect ratio matters for this content mix.")

    # Show biggest rank disagreements
    rank_native = np.argsort(np.argsort(-scores_native))
    rank_batched = np.argsort(np.argsort(-scores_batched))
    rank_diff = np.abs(rank_native - rank_batched)
    worst = np.argsort(-rank_diff)[:5]

    print(f"\nBiggest rank disagreements:")
    for idx in worst:
        fname = os.path.basename(jpeg_paths[idx])
        print(
            f"  {fname}: native=#{rank_native[idx]+1} "
            f"batched=#{rank_batched[idx]+1} "
            f"(Δ{rank_diff[idx]})"
        )


if __name__ == "__main__":
    main()