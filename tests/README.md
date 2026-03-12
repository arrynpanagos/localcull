# Test Fixtures Setup

The integration tests require real CR3 files from Canon cameras. These
are not distributed with the package — assemble them from your own shoots.

## Directory Structure

```
tests/fixtures/
├── portrait/           # 1+ portrait CR3 with clear face, eyes open
├── landscape/          # 1+ landscape CR3 with no faces
├── pet/                # 1+ pet/wildlife CR3 (clear subject, no face)
├── blink/              # 1+ CR3 with known blink (eyes clearly closed)
├── burst/              # 5+ sequential CR3 from a continuous burst
├── mixed_burst/        # 10+ CR3 burst where subject turns away mid-sequence
├── portrait_orientation/  # 1+ CR3 shot in portrait (vertical) orientation
└── second_camera/      # 1+ CR3 from a different camera body (e.g., EOS R)
```

## Requirements

- **Minimum**: 20 CR3 files across the above categories
- **Recommended**: 50 CR3 files for thorough coverage
- **Two camera bodies**: At least one file from a second body (e.g., EOS R
  alongside R5 Mark II) to test per-camera z-normalization

## Content Guidelines

| Category | What to include | Why |
|----------|----------------|-----|
| portrait | Well-lit face, both eyes visible and sharp | Face Mesh detection baseline |
| landscape | No people, wide scene | Saliency confidence < 0.3 expected |
| pet | Single animal, clear subject | High saliency confidence without face |
| blink | Eyes clearly closed | Blink gate validation |
| burst | 5+ frames from 30fps continuous | Burst boundary detection |
| mixed_burst | Subject turns 90°+ mid-burst | Harmonization trigger (20-80% non-face) |
| portrait_orientation | Camera rotated 90° | EXIF transpose validation |
| second_camera | Different body, any content | Per-camera z-norm validation |

## Running Tests

```bash
# Unit tests only (no fixtures needed)
pytest tests/ -v -m "not fixtures"

# Integration tests (requires fixtures)
export LOCALCULL_TEST_FIXTURES=tests/fixtures
pytest tests/ -v

# Skip slow tests
pytest tests/ -v -m "not slow"
```

## Custom Fixture Location

Set `LOCALCULL_TEST_FIXTURES` to use a different directory:

```bash
export LOCALCULL_TEST_FIXTURES=/path/to/my/test/images
pytest tests/ -v
```

This is useful if your test images live outside the repo (e.g., on an
external drive or a shared network volume).
