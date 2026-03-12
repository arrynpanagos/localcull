"""
CLI entry point for localcull.

Usage:
    localcull /path/to/cr3s --shoot-id wedding_2026 --target 100
    localcull /path/to/cr3s --shoot-id session_01
    localcull --cleanup-shm
    localcull --clear-cache
    localcull --clear-cache --shoot-id session_01
"""

# Suppress C++ noise from TensorFlow Lite / MediaPipe / glog
# Must be set BEFORE any imports that trigger these libraries
import os as _os

_os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
_os.environ["GLOG_minloglevel"] = "3"

import argparse
import glob
import logging
import os
import sys
import time


def main():
    parser = argparse.ArgumentParser(
        prog="localcull",
        description="AI-powered photo culling pipeline for RAW + JPEG workflows.",
    )
    parser.add_argument(
        "input_dir",
        nargs="?",
        help="Directory containing image files (scanned recursively)",
    )
    parser.add_argument(
        "--shoot-id",
        required=False,
        help="Unique shoot identifier for caching. Defaults to directory name.",
    )
    parser.add_argument(
        "--target",
        type=int,
        default=None,
        help="Approximate number of images to select. "
        "Defaults to ~3-5%% of input.",
    )
    parser.add_argument(
        "--pu-head",
        default=None,
        help="Path to saved PUPersonalizationHead (Phase 2).",
    )
    parser.add_argument(
        "--cleanup-shm",
        action="store_true",
        help="Clean up orphaned shared memory blocks and exit.",
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear checkpoint cache and exit. Use with --shoot-id "
        "to clear a specific shoot.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress all output except errors.",
    )
    parser.add_argument(
        "--scorers",
        nargs="+",
        default=None,
        metavar="NAME",
        help="Scorers to run (default: topiq musiq qalign qualiclip deqa_score). "
        "Use 'all' to run every registered scorer. "
        "Available: topiq, musiq, qalign, qualiclip, deqa_score, "
        "q_scorer, q_insight.",
    )
    parser.add_argument(
        "--list-scorers",
        action="store_true",
        help="List all registered scorers and exit.",
    )

    args = parser.parse_args()

    # ── Logging setup ──
    if args.quiet:
        level = logging.ERROR
    elif args.verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Silence noisy third-party loggers even in verbose mode
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("PIL.TiffImagePlugin").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("timm").setLevel(logging.WARNING)

    # ── Maintenance commands ──
    if args.cleanup_shm:
        from localcull.memory import cleanup_orphaned_shm

        cleanup_orphaned_shm()
        print("Shared memory cleanup complete.")
        return

    if args.clear_cache:
        from localcull.checkpoint import clear_cache

        clear_cache(args.shoot_id)
        print("Cache cleared.")
        return

    if args.list_scorers:
        from localcull.scorers import SCORER_REGISTRY

        print("Registered scorers:")
        for name, spec in sorted(SCORER_REGISTRY.items()):
            default = " [DEFAULT]" if spec.default_enabled else ""
            print(
                f"  {name:15s} {spec.display_name:15s} "
                f"{spec.output_range[0]:.0f}-{spec.output_range[1]:.0f}  "
                f"{spec.requires:12s} {spec.description}{default}"
            )
        return

    # ── Validate input ──
    if not args.input_dir:
        parser.error("input_dir is required (unless using --cleanup-shm or --clear-cache)")

    input_dir = os.path.abspath(args.input_dir)
    if not os.path.isdir(input_dir):
        parser.error(f"Not a directory: {input_dir}")

    # Find image files (case-insensitive, recursive)
    # Supported: Canon CR3, Nikon NEF, Sony ARW, Fuji RAF, Adobe DNG,
    # Olympus ORF, Pentax PEF, Panasonic RW2, plus JPEG/TIFF.
    SUPPORTED_EXTENSIONS = {
        ".cr3", ".cr2", ".nef", ".nrw", ".arw", ".srf", ".sr2",
        ".raf", ".dng", ".orf", ".pef", ".rw2", ".rwl",
        ".3fr", ".fff", ".iiq", ".mrw", ".x3f", ".srw",
        ".jpg", ".jpeg", ".tif", ".tiff",
    }

    all_files = []
    for root, dirs, files in os.walk(input_dir):
        # Skip ranked output folders and hidden directories
        dirs[:] = [d for d in dirs if not d.startswith(".") and "_ranked_" not in d]
        for fname in files:
            # Skip embedded JPEGs from previous runs
            if fname.lower().endswith("_embedded.jpg"):
                continue
            ext = os.path.splitext(fname)[1].lower()
            if ext in SUPPORTED_EXTENSIONS:
                all_files.append(os.path.join(root, fname))

    image_paths = sorted(set(os.path.abspath(p) for p in all_files))

    if not image_paths:
        parser.error(f"No supported image files found in {input_dir}")

    shoot_id = args.shoot_id or os.path.basename(input_dir)

    # Summarize formats found
    from collections import Counter
    ext_counts = Counter(os.path.splitext(p)[1].upper() for p in image_paths)
    fmt_str = ", ".join(f"{cnt} {ext}" for ext, cnt in ext_counts.most_common())
    print(f"localcull: {len(image_paths)} images in {input_dir} ({fmt_str})")
    print(f"Shoot ID: {shoot_id}")
    if args.target:
        print(f"Target selection: ~{args.target} images")

    # ── Load optional PU head ──
    pu_head = None
    if args.pu_head:
        from localcull.personalization import load_pu_head

        pu_head = load_pu_head(args.pu_head)

    # ── Run pipeline ──
    t0 = time.time()

    from localcull.pipeline import run_pipeline

    frames, ratings, selected_indices = run_pipeline(
        image_paths,
        shoot_id=shoot_id,
        pu_head=pu_head,
        target_total=args.target,
        enabled_scorers=args.scorers,
    )

    elapsed = time.time() - t0
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)

    n_selected = len(selected_indices)
    print(f"\nComplete: {len(frames)} rated, {n_selected} selected for edit")
    print(f"Time: {minutes}m {seconds}s")

    # Star distribution summary
    from collections import Counter

    dist = Counter(ratings.values())
    for star in sorted(dist.keys(), reverse=True):
        print(f"  {star}★: {dist[star]} images")
    print(f"  Selected for edit: {n_selected} images")


if __name__ == "__main__":
    main()