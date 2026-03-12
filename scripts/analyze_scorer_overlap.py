#!/usr/bin/env python3
"""
Analyze scorer overlap and redundancy from a localcull features CSV.

Usage:
    python scripts/analyze_scorer_overlap.py test_photos2/test_photos2_features.csv
"""

import csv
import sys
from collections import defaultdict

import numpy as np
from scipy.stats import spearmanr


def main():
    if len(sys.argv) < 2:
        path = "test_photos2/test_photos2_features.csv"
    else:
        path = sys.argv[1]

    with open(path) as f:
        rows = list(csv.DictReader(f))

    print(f"Loaded {len(rows)} images from {path}\n")

    # Extract score arrays (skip scorers that are all zeros)
    scorer_cols = {
        "Q-Align": "qalign_score",
        "DeQA": "deqa_score",
        "QualiCLIP+": "qualiclip_score",
        "TOPIQ": "topiq_score",
        "MUSIQ": "musiq_score",
    }
    arrays = {}
    for name, col in scorer_cols.items():
        arr = np.array([float(r[col]) for r in rows])
        if arr.sum() == 0:
            print(f"  {name}: all zeros, skipping")
            continue
        arrays[name] = arr

    names = list(arrays.keys())

    # --- Distributions ---
    print("=" * 60)
    print("SCORE DISTRIBUTIONS")
    print("=" * 60)
    for name in names:
        a = arrays[name]
        print(f"  {name:12s}: mean={a.mean():.3f}  std={a.std():.3f}  "
              f"range=[{a.min():.3f}, {a.max():.3f}]")

    # --- Pearson ---
    print()
    print("=" * 60)
    print("PEARSON CORRELATIONS")
    print("=" * 60)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            r = np.corrcoef(arrays[names[i]], arrays[names[j]])[0, 1]
            flag = " *** REDUNDANT" if abs(r) > 0.90 else ""
            print(f"  {names[i]:12s} vs {names[j]:12s}: r={r:.4f}{flag}")

    # --- Spearman ---
    print()
    print("=" * 60)
    print("SPEARMAN RANK CORRELATIONS")
    print("=" * 60)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            rho, p = spearmanr(arrays[names[i]], arrays[names[j]])
            flag = " *** REDUNDANT" if abs(rho) > 0.90 else ""
            print(f"  {names[i]:12s} vs {names[j]:12s}: rho={rho:.4f}  p={p:.2e}{flag}")

    # --- Q-Align vs DeQA deep dive ---
    if "Q-Align" in arrays and "DeQA" in arrays:
        print()
        print("=" * 60)
        print("Q-ALIGN vs DeQA DISAGREEMENT ANALYSIS")
        print("=" * 60)
        qa = arrays["Q-Align"]
        dq = arrays["DeQA"]
        diffs = qa - dq
        print(f"  Mean diff (qalign - deqa): {diffs.mean():+.3f}")
        print(f"  Std of diffs:              {diffs.std():.3f}")
        print(f"  Max |diff|:                {np.abs(diffs).max():.3f}")
        print(f"  |diff| > 0.3: {np.sum(np.abs(diffs) > 0.3):4d} / {len(rows)} "
              f"({100 * np.sum(np.abs(diffs) > 0.3) / len(rows):.1f}%)")
        print(f"  |diff| > 0.5: {np.sum(np.abs(diffs) > 0.5):4d} / {len(rows)} "
              f"({100 * np.sum(np.abs(diffs) > 0.5) / len(rows):.1f}%)")
        print(f"  |diff| > 1.0: {np.sum(np.abs(diffs) > 1.0):4d} / {len(rows)} "
              f"({100 * np.sum(np.abs(diffs) > 1.0) / len(rows):.1f}%)")

    # --- Within-scene rank flips for all pairs ---
    print()
    print("=" * 60)
    print("WITHIN-SCENE RANK FLIPS (does scorer A rank img1 > img2")
    print("while scorer B ranks img1 < img2, within the same scene?)")
    print("=" * 60)

    scenes = defaultdict(list)
    for i, r in enumerate(rows):
        scenes[r["scene_id"]].append(i)

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a_arr = arrays[names[i]]
            b_arr = arrays[names[j]]
            flips = 0
            total = 0
            for sid, indices in scenes.items():
                if len(indices) < 2:
                    continue
                for a in range(len(indices)):
                    for b in range(a + 1, len(indices)):
                        ii, jj = indices[a], indices[b]
                        a_order = a_arr[ii] - a_arr[jj]
                        b_order = b_arr[ii] - b_arr[jj]
                        total += 1
                        if a_order * b_order < 0:
                            flips += 1
            pct = 100 * flips / total if total > 0 else 0
            print(f"  {names[i]:12s} vs {names[j]:12s}: "
                  f"{flips:4d} / {total:4d} flips ({pct:.1f}%)")

    # --- TOPIQ vs MUSIQ redundancy check ---
    if "TOPIQ" in arrays and "MUSIQ" in arrays:
        print()
        print("=" * 60)
        print("TECH GATE REDUNDANCY: TOPIQ vs MUSIQ")
        print("=" * 60)
        topiq = arrays["TOPIQ"]
        musiq = arrays["MUSIQ"]
        r = np.corrcoef(topiq, musiq)[0, 1]
        rho, _ = spearmanr(topiq, musiq)
        print(f"  Pearson r={r:.4f}, Spearman rho={rho:.4f}")

        # Check: does dropping one change tech gate outcomes?
        topiq_thresh = 0.35
        musiq_thresh = 30.0
        topiq_fail = topiq < topiq_thresh
        musiq_fail = musiq < musiq_thresh
        both_fail = topiq_fail & musiq_fail
        topiq_only = topiq_fail & ~musiq_fail
        musiq_only = musiq_fail & ~topiq_fail

        print(f"  TOPIQ < {topiq_thresh}: {topiq_fail.sum()} images")
        print(f"  MUSIQ < {musiq_thresh}: {musiq_fail.sum()} images")
        print(f"  Both fail:         {both_fail.sum()} images")
        print(f"  TOPIQ-only fail:   {topiq_only.sum()} images")
        print(f"  MUSIQ-only fail:   {musiq_only.sum()} images")
        print()
        if topiq_only.sum() == 0 and musiq_only.sum() == 0:
            print("  -> They gate the EXACT same images. One is fully redundant.")
        elif topiq_only.sum() + musiq_only.sum() < 3:
            print("  -> Near-identical gating. Minimal info loss from dropping one.")
        else:
            print("  -> They gate some different images. Both contribute to the gate.")

    # --- Summary ---
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("  Scorers ranked by Pearson correlation with DeQA (primary):")
    if "DeQA" in arrays:
        corrs = []
        for name in names:
            if name == "DeQA":
                continue
            r = np.corrcoef(arrays[name], arrays["DeQA"])[0, 1]
            corrs.append((name, r))
        corrs.sort(key=lambda x: -abs(x[1]))
        for name, r in corrs:
            verdict = "REDUNDANT?" if abs(r) > 0.90 else "orthogonal" if abs(r) < 0.50 else "correlated"
            print(f"    {name:12s}: r={r:.4f}  ({verdict})")


if __name__ == "__main__":
    main()