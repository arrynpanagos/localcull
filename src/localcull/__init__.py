# Copyright 2025 Arryn Panagos. All rights reserved.
# Licensed under the Elastic License 2.0 (ELv2).

"""
localcull — AI-assisted local photo culling for RAW and JPEG workflows.

Processes photo shoots through a multi-stage pipeline: temporal
clustering, multi-model IQA scoring, face/subject analysis,
visual cluster selection, and star rating. Outputs XMP sidecar
files compatible with Lightroom / Capture One / darktable.

All processing runs locally on Apple Silicon. No images leave
your computer.
"""

__version__ = "0.6.0"

from localcull.pipeline import run_pipeline

__all__ = ["run_pipeline", "__version__"]
