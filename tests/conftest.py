"""
Test fixtures and configuration for localcull test suite.

Test CR3 fixtures must be assembled manually from real shoots.
See tests/README.md for requirements and directory structure.
"""

import os

import numpy as np
import pytest

from localcull.types import Frame

# ── Fixture directory ──
FIXTURE_DIR = os.environ.get("LOCALCULL_TEST_FIXTURES", os.path.join(
    os.path.dirname(__file__), "fixtures"
))

# Expected subdirectory structure within fixtures/
FIXTURE_STRUCTURE = {
    "portrait": "1+ portrait CR3 with clear face, eyes open",
    "landscape": "1+ landscape CR3 with no faces",
    "pet": "1+ pet/wildlife CR3 (clear single subject, no face)",
    "blink": "1+ CR3 with known blink (eyes clearly closed)",
    "burst": "5+ sequential CR3 from a continuous burst",
    "mixed_burst": "10+ CR3 burst with some frames where subject turns away",
    "portrait_orientation": "1+ CR3 shot in portrait (vertical) orientation",
    "second_camera": "1+ CR3 from a different camera body",
}


def _fixtures_available() -> bool:
    """Check if the fixture directory is populated."""
    if not os.path.exists(FIXTURE_DIR):
        return False
    cr3s = []
    for root, _, files in os.walk(FIXTURE_DIR):
        cr3s.extend(
            os.path.join(root, f)
            for f in files
            if f.lower().endswith(".cr3")
        )
    return len(cr3s) >= 5  # minimum viable test set


requires_fixtures = pytest.mark.skipif(
    not _fixtures_available(),
    reason=(
        f"Test fixtures not populated at {FIXTURE_DIR}. "
        f"See tests/README.md for setup instructions."
    ),
)


@pytest.fixture
def fixture_cr3_paths():
    """All CR3 paths in the fixture directory."""
    if not _fixtures_available():
        pytest.skip("Fixtures not available")
    paths = []
    for root, _, files in os.walk(FIXTURE_DIR):
        paths.extend(
            os.path.join(root, f)
            for f in sorted(files)
            if f.lower().endswith(".cr3")
        )
    return sorted(paths)


@pytest.fixture
def fixture_portrait_paths():
    """CR3 paths from the portrait subdirectory."""
    d = os.path.join(FIXTURE_DIR, "portrait")
    if not os.path.isdir(d):
        pytest.skip("portrait fixture dir missing")
    return sorted(
        os.path.join(d, f) for f in os.listdir(d) if f.lower().endswith(".cr3")
    )


@pytest.fixture
def fixture_landscape_paths():
    """CR3 paths from the landscape subdirectory."""
    d = os.path.join(FIXTURE_DIR, "landscape")
    if not os.path.isdir(d):
        pytest.skip("landscape fixture dir missing")
    return sorted(
        os.path.join(d, f) for f in os.listdir(d) if f.lower().endswith(".cr3")
    )


# ── Synthetic frame factory ──


def make_synthetic_frame(
    global_index: int = 0,
    scene_id: int = 0,
    burst_id: int = 0,
    frame_index: int = 0,
    burst_length: int = 30,
    camera_body: str = "Canon EOS R5 m2",
    has_face: bool = True,
    sharp_near_eye: float = 100.0,
    sharp_far_eye: float = 50.0,
    sharpness_subject: float = 80.0,
    sharpness_background: float = 20.0,
    topiq_score: float = 0.5,
    musiq_score: float = 0.5,
    blink_detected: bool = False,
    raw_min_ear: float = 0.30,
    subject_det_method: str = "face_mesh",
    rendering_path: str = "embedded_jpeg",
    saliency_confidence: float = 1.0,
    **kwargs,
) -> Frame:
    """Create a synthetic Frame for unit testing."""
    f = Frame(
        path=f"/fake/IMG_{global_index:04d}.CR3",
        global_index=global_index,
        scene_id=scene_id,
        burst_id=burst_id,
        frame_index=frame_index,
        burst_length=burst_length,
        camera_body=camera_body,
    )
    f.has_face = has_face
    f.sharp_near_eye = sharp_near_eye
    f.sharp_far_eye = sharp_far_eye
    f.sharpness_subject = sharpness_subject
    f.sharpness_background = sharpness_background
    f.topiq_score = topiq_score
    f.musiq_score = musiq_score
    f.blink_detected = blink_detected
    f.raw_min_ear = raw_min_ear
    f.subject_det_method = subject_det_method
    f.rendering_path = rendering_path
    f.saliency_confidence = saliency_confidence
    f.isolation_ratio = sharpness_subject / (sharpness_background + 1e-6)
    f.eye_ratio_raw = (
        np.clip(sharp_near_eye / (sharp_far_eye + 1e-6) - 1.0, 0, 3.0) * 0.1
    )
    f.n_faces = 1 if has_face else 0

    for k, v in kwargs.items():
        if hasattr(f, k):
            setattr(f, k, v)
    return f


def make_synthetic_burst(
    n: int = 30,
    n_no_face: int = 0,
    burst_id: int = 0,
    scene_id: int = 0,
    start_index: int = 0,
) -> list[Frame]:
    """
    Create a synthetic burst for testing harmonization, blink
    calibration, and burst reduction.
    """
    frames = []
    for i in range(n):
        has_face = i >= n_no_face  # first n_no_face frames are no-face
        f = make_synthetic_frame(
            global_index=start_index + i,
            scene_id=scene_id,
            burst_id=burst_id,
            frame_index=i,
            burst_length=n,
            has_face=has_face,
            sharp_near_eye=100.0 + np.random.randn() * 10 if has_face else 0.0,
            sharp_far_eye=50.0 + np.random.randn() * 5 if has_face else 0.0,
            sharpness_subject=80.0 + np.random.randn() * 8,
            topiq_score=0.5 + np.random.randn() * 0.02,
            musiq_score=0.5 + np.random.randn() * 0.02,
            raw_min_ear=0.28 + np.random.randn() * 0.03,
            subject_det_method="face_mesh" if has_face else "center",
        )
        frames.append(f)
    return frames
