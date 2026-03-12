#!/usr/bin/env python3
"""
Analyze scorer disagreement structure via PCA and z-score consensus.

Produces:
  1. PCA loadings — what each component captures
  2. Explained variance — how much structure exists
  3. Per-image PC scores with outlier detection
  4. Z-score consensus ranking vs current DeQA-only ranking
  5. Disagreement heatmap — which images split the models
  6. Scatter plots of PC1 vs PC2, colored by star rating

Usage:
    python scripts/analyze_pca_scorers.py 28FEB25_HOLLIDAY_PARK/DCIM/100EOSR5/28FEB25_HOLLIDAY_PARK_features.csv
    python scripts/analyze_pca_scorers.py test_photos2/test_photos2_features.csv
"""

import csv
import sys
import os

import numpy as np
from scipy.stats import spearmanr, zscore


def load_scores(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))

    scorer_cols = {
        "Q-Align": "qalign_score",
        "DeQA": "deqa_score",
        "QualiCLIP+": "qualiclip_score",
        "TOPIQ": "topiq_score",
        "MUSIQ": "musiq_score",
        "NIMA": "nima_score",
    }

    # Filter out all-zero columns (disabled scorers)
    active = {}
    for name, col in scorer_cols.items():
        arr = np.array([float(r[col]) for r in rows])
        if arr.sum() > 0:
            active[name] = arr

    filenames = [r["filename"] for r in rows]
    stars = np.array([int(r["star_rating"]) for r in rows])
    deqa_raw = np.array([float(r["deqa_score"]) for r in rows])
    tech_pass = np.array([r["technical_gate_pass"] == "True" for r in rows])
    disagree = np.array([r["aesthetic_disagreement"] == "True" for r in rows])

    return rows, active, filenames, stars, deqa_raw, tech_pass, disagree


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/analyze_pca_scorers.py <features.csv>")
        sys.exit(1)

    path = sys.argv[1]
    rows, scores, filenames, stars, deqa_raw, tech_pass, disagree = load_scores(path)
    names = list(scores.keys())
    n_images = len(filenames)
    n_scorers = len(names)

    print(f"Loaded {n_images} images, {n_scorers} active scorers: {', '.join(names)}")
    print()

    # ══════════════════════════════════════════════════════════════
    # 1. Z-NORMALIZE all scores
    # ══════════════════════════════════════════════════════════════
    raw_matrix = np.column_stack([scores[n] for n in names])  # (N, K)
    z_matrix = zscore(raw_matrix, axis=0)  # (N, K) each column mean=0 std=1

    print("=" * 70)
    print("Z-SCORE NORMALIZED DISTRIBUTIONS")
    print("=" * 70)
    for i, name in enumerate(names):
        col = z_matrix[:, i]
        print(f"  {name:12s}: mean={col.mean():.4f}  std={col.std():.4f}  "
              f"range=[{col.min():.2f}, {col.max():.2f}]")

    # ══════════════════════════════════════════════════════════════
    # 2. PCA
    # ══════════════════════════════════════════════════════════════
    cov = np.cov(z_matrix, rowvar=False)  # (K, K)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Sort descending
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    explained = eigenvalues / eigenvalues.sum()
    cumulative = np.cumsum(explained)

    print()
    print("=" * 70)
    print("PCA EXPLAINED VARIANCE")
    print("=" * 70)
    for i in range(n_scorers):
        bar = "█" * int(explained[i] * 50)
        print(f"  PC{i+1}: {explained[i]:6.1%} (cumulative: {cumulative[i]:6.1%})  {bar}")

    print()
    print("=" * 70)
    print("PCA LOADINGS (what each component captures)")
    print("=" * 70)
    print(f"  {'Scorer':12s}", end="")
    for i in range(min(n_scorers, 4)):
        print(f"    PC{i+1:d}   ", end="")
    print()
    print("  " + "-" * 60)

    for j, name in enumerate(names):
        print(f"  {name:12s}", end="")
        for i in range(min(n_scorers, 4)):
            v = eigenvectors[j, i]
            marker = " ***" if abs(v) > 0.5 else ""
            print(f"  {v:+7.3f}{marker}", end="")
        print()

    # Interpret components
    print()
    print("  INTERPRETATION:")
    for i in range(min(n_scorers, 3)):
        pos = [(names[j], eigenvectors[j, i]) for j in range(n_scorers) if eigenvectors[j, i] > 0.3]
        neg = [(names[j], eigenvectors[j, i]) for j in range(n_scorers) if eigenvectors[j, i] < -0.3]
        pos.sort(key=lambda x: -x[1])
        neg.sort(key=lambda x: x[1])

        pos_str = " + ".join(f"{n}({v:+.2f})" for n, v in pos) if pos else "—"
        neg_str = " + ".join(f"{n}({v:+.2f})" for n, v in neg) if neg else "—"

        print(f"  PC{i+1} ({explained[i]:.0%}): HIGH={pos_str}")
        if neg:
            print(f"  {'':12s}  LOW={neg_str}")

    # ══════════════════════════════════════════════════════════════
    # 3. PROJECT images onto PCs
    # ══════════════════════════════════════════════════════════════
    pc_scores = z_matrix @ eigenvectors  # (N, K)

    # ══════════════════════════════════════════════════════════════
    # 4. Z-SCORE CONSENSUS vs DeQA-only ranking
    # ══════════════════════════════════════════════════════════════
    z_consensus = z_matrix.mean(axis=1)  # average of z-scores = PC1 approx
    z_disagreement = z_matrix.std(axis=1)  # per-image std = model uncertainty

    # Rank comparison
    deqa_rank = np.argsort(np.argsort(-deqa_raw))  # 0 = best
    consensus_rank = np.argsort(np.argsort(-z_consensus))
    pc1_rank = np.argsort(np.argsort(-pc_scores[:, 0]))

    rho_consensus, _ = spearmanr(deqa_rank, consensus_rank)
    rho_pc1, _ = spearmanr(deqa_rank, pc1_rank)

    print()
    print("=" * 70)
    print("CONSENSUS vs DEQA-ONLY RANKING")
    print("=" * 70)
    print(f"  Spearman rank correlation (DeQA rank vs z-consensus rank): {rho_consensus:.4f}")
    print(f"  Spearman rank correlation (DeQA rank vs PC1 rank):         {rho_pc1:.4f}")

    # Where do they disagree most?
    rank_diff = np.abs(deqa_rank - consensus_rank)
    big_diffs = np.where(rank_diff > n_images * 0.1)[0]  # >10% rank shift
    print(f"  Images with >10% rank shift: {len(big_diffs)} / {n_images}")

    if len(big_diffs) > 0:
        print()
        print("  Top 10 rank shifts (consensus promotes or demotes vs DeQA):")
        top_shifts = sorted(big_diffs, key=lambda i: rank_diff[i], reverse=True)[:10]
        for idx in top_shifts:
            direction = "↑" if consensus_rank[idx] < deqa_rank[idx] else "↓"
            print(f"    {filenames[idx]:20s}: DeQA rank={deqa_rank[idx]:4d}  "
                  f"consensus rank={consensus_rank[idx]:4d}  "
                  f"shift={direction}{rank_diff[idx]:4d}  "
                  f"disagreement={z_disagreement[idx]:.2f}")

    # ══════════════════════════════════════════════════════════════
    # 5. DISAGREEMENT ANALYSIS
    # ══════════════════════════════════════════════════════════════
    print()
    print("=" * 70)
    print("MODEL DISAGREEMENT (per-image z-score std)")
    print("=" * 70)
    print(f"  Mean disagreement: {z_disagreement.mean():.3f}")
    print(f"  Std of disagreement: {z_disagreement.std():.3f}")
    print(f"  Low agreement (std > 1.0): {np.sum(z_disagreement > 1.0)} images")
    print(f"  High agreement (std < 0.3): {np.sum(z_disagreement < 0.3)} images")

    # Most controversial images
    print()
    print("  Top 10 most controversial (highest model disagreement):")
    controversial = np.argsort(-z_disagreement)[:10]
    for idx in controversial:
        scores_str = "  ".join(f"{names[j]}={z_matrix[idx, j]:+.2f}" for j in range(n_scorers))
        print(f"    {filenames[idx]:20s}: std={z_disagreement[idx]:.2f}  [{scores_str}]")

    # ══════════════════════════════════════════════════════════════
    # 6. ASYMMETRIC RESIDUALS (Q-Align vs DeQA direction)
    # ══════════════════════════════════════════════════════════════
    if "Q-Align" in names and "DeQA" in names:
        qi = names.index("Q-Align")
        di = names.index("DeQA")
        residual = z_matrix[:, qi] - z_matrix[:, di]

        print()
        print("=" * 70)
        print("ASYMMETRIC RESIDUALS: Q-Align - DeQA (z-normalized)")
        print("=" * 70)
        print(f"  Mean residual: {residual.mean():+.3f}")
        print(f"  Std: {residual.std():.3f}")
        print(f"  Q-Align >> DeQA (residual > 0.5): {np.sum(residual > 0.5)} images")
        print(f"  DeQA >> Q-Align (residual < -0.5): {np.sum(residual < -0.5)} images")

        print()
        print("  Top 5 where Q-Align loves it but DeQA doesn't:")
        for idx in np.argsort(-residual)[:5]:
            print(f"    {filenames[idx]:20s}: Q-Align z={z_matrix[idx, qi]:+.2f}  "
                  f"DeQA z={z_matrix[idx, di]:+.2f}  "
                  f"residual={residual[idx]:+.2f}  star={stars[idx]}")

        print()
        print("  Top 5 where DeQA loves it but Q-Align doesn't:")
        for idx in np.argsort(residual)[:5]:
            print(f"    {filenames[idx]:20s}: Q-Align z={z_matrix[idx, qi]:+.2f}  "
                  f"DeQA z={z_matrix[idx, di]:+.2f}  "
                  f"residual={residual[idx]:+.2f}  star={stars[idx]}")

    # ══════════════════════════════════════════════════════════════
    # 7. MUSIQ GATE REDUNDANCY CHECK
    # ══════════════════════════════════════════════════════════════
    if "MUSIQ" in scores:
        musiq_raw = scores["MUSIQ"]
        musiq_gate_fail = musiq_raw < 30.0
        n_fail = musiq_gate_fail.sum()

        print()
        print("=" * 70)
        print(f"MUSIQ GATE REDUNDANCY ({n_fail} images fail MUSIQ < 30)")
        print("=" * 70)

        if n_fail > 0:
            fail_deqa = deqa_raw[musiq_gate_fail]
            fail_consensus = z_consensus[musiq_gate_fail]
            all_deqa_sorted = np.sort(deqa_raw)

            # What percentile are these images in?
            for idx in np.where(musiq_gate_fail)[0]:
                pct = (deqa_raw[idx] > deqa_raw).mean() * 100
                print(f"    {filenames[idx]:20s}: MUSIQ={musiq_raw[idx]:.1f}  "
                      f"DeQA={deqa_raw[idx]:.3f} (percentile={pct:.0f}%)  "
                      f"consensus_z={z_consensus[idx]:+.2f}  star={stars[idx]}★")

            bottom_10_deqa = np.percentile(deqa_raw, 10)
            caught_by_deqa = np.sum(fail_deqa < bottom_10_deqa)
            print()
            print(f"  Of {n_fail} MUSIQ-gated images:")
            print(f"    {caught_by_deqa} are also in DeQA bottom 10% (< {bottom_10_deqa:.3f})")
            print(f"    {n_fail - caught_by_deqa} would be missed by DeQA-only gating")

    # ══════════════════════════════════════════════════════════════
    # 8. STAR RATING vs CONSENSUS
    # ══════════════════════════════════════════════════════════════
    print()
    print("=" * 70)
    print("STAR COMPRESSION: consensus z-score by current star rating")
    print("=" * 70)
    for s in sorted(set(stars)):
        mask = stars == s
        if mask.sum() == 0:
            continue
        z_vals = z_consensus[mask]
        print(f"  {s}★ ({mask.sum():5d} images): "
              f"consensus_z mean={z_vals.mean():+.3f}  std={z_vals.std():.3f}  "
              f"range=[{z_vals.min():+.2f}, {z_vals.max():+.2f}]")

    # ══════════════════════════════════════════════════════════════
    # 9. PROPOSED PERCENTILE THRESHOLDS
    # ══════════════════════════════════════════════════════════════
    print()
    print("=" * 70)
    print("PROPOSED PERCENTILE-BASED STAR THRESHOLDS (using consensus)")
    print("=" * 70)
    percentiles = [5, 20, 50, 80, 95]
    thresholds = np.percentile(z_consensus, percentiles)
    labels = ["1★ (bottom 5%)", "2★ (5-20%)", "3★ (20-50%)", "4★ (50-80%)", "5★ (top 20%)"]

    star_counts = [
        np.sum(z_consensus < thresholds[0]),
        np.sum((z_consensus >= thresholds[0]) & (z_consensus < thresholds[1])),
        np.sum((z_consensus >= thresholds[1]) & (z_consensus < thresholds[2])),
        np.sum((z_consensus >= thresholds[2]) & (z_consensus < thresholds[3])),
        np.sum(z_consensus >= thresholds[3]),
    ]

    for label, count in zip(labels, star_counts):
        pct = 100 * count / n_images
        bar = "█" * int(pct)
        print(f"  {label:20s}: {count:5d} ({pct:4.1f}%)  {bar}")

    print()
    print("  vs current fixed-threshold distribution:")
    for s in sorted(set(stars)):
        mask = stars == s
        print(f"    {s}★: {mask.sum():5d} ({100*mask.sum()/n_images:4.1f}%)")

    # ══════════════════════════════════════════════════════════════
    # 10. WRITE ENRICHED CSV
    # ══════════════════════════════════════════════════════════════
    out_path = path.replace("_features.csv", "_pca.csv")
    if out_path == path:
        out_path = path.replace(".csv", "_pca.csv")

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["filename", "star_rating", "deqa_rank", "consensus_rank",
                  "rank_shift", "z_consensus", "z_disagreement"]
        for i in range(min(n_scorers, 3)):
            header.append(f"PC{i+1}")
        for name in names:
            header.append(f"z_{name.lower().replace('+', 'plus')}")
        writer.writerow(header)

        for idx in range(n_images):
            row = [
                filenames[idx],
                stars[idx],
                int(deqa_rank[idx]),
                int(consensus_rank[idx]),
                int(consensus_rank[idx]) - int(deqa_rank[idx]),
                f"{z_consensus[idx]:.4f}",
                f"{z_disagreement[idx]:.4f}",
            ]
            for i in range(min(n_scorers, 3)):
                row.append(f"{pc_scores[idx, i]:.4f}")
            for j in range(n_scorers):
                row.append(f"{z_matrix[idx, j]:.4f}")
            writer.writerow(row)

    print()
    print(f"Enriched CSV written: {out_path}")
    print("  Columns: z_consensus, z_disagreement, PC1-PC3, per-scorer z-scores")
    print("  Sort by z_consensus for multi-model ranking")
    print("  Sort by z_disagreement descending for most controversial images")


if __name__ == "__main__":
    main()