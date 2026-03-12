# localcull

Silicon Native AI-assisted local photo culling for RAW and JPEG workflows. Entirely local.

Supports Canon CR3/CR2, Nikon NEF, Sony ARW, Fuji RAF, DNG, Olympus ORF, Panasonic RW2, and most other RAW formats, plus JPEG and TIFF.

---

Hello there, I am a mathematician, statistician and hobbyist photographer and I have developed this AI photo culling app. This is its first public version.

The purpose of this tool is not to replace culling and inspection, but rather to speed the process up. To help photographers judge photos more quickly and more completely. Many know the adage "what I think looks good and what my client/others think looks good aren't always the same thing." This tool accelerates your culling by surfacing the strongest images and offering perspectives you might not have considered. Will those perspectives always be right? No, but neither is a human grader. This tool is meant to provide greater diversity in understanding.

localcull runs six neural image quality models and treats their disagreements as signal, meaningful information, not noise to average away. When models disagree on the best image in a group, both picks are kept: the consensus choice (Red label) and the alternative (Purple label). It surfaces images you might otherwise scroll past.

Beyond scoring, localcull groups your images by visual similarity using DINOv2 embeddings, not just timestamps or burst detection, but what the images actually look like. A 1,800-image shoot can contain 58 distinct visual moments, especially shooting high fps. Seeing that structure helps you understand the shape of a shoot before you start editing: how many unique compositions you captured, where you spent the most frames, and which moments have real variety versus repetitive bursts.

All processing runs locally on your Apple Silicon Mac. No images leave your computer. Each scoring model runs in an isolated subprocess; Metal GPU memory is fully reclaimed between models, preventing the memory leaks that plague MPS inference pipelines. All results are cached, so you can tweak settings without re-running models.

## Pipeline

1. **Ingest** — Read EXIF, sort chronologically, detect scene and burst boundaries from time gaps, focal length changes, and exposure shifts.
2. **Prepare** — Extract embedded JPEGs from RAW files (or use native JPEGs), decode to mid-resolution arrays in shared memory.
3. **Score** — Run six neural image quality models, face detection with eye-level sharpness measurement, and per-burst blink calibration.
4. **Select** — Group visually similar images using DINOv2 embeddings + agglomerative clustering, then pick the best from each cluster using a weighted consensus of model scores.
5. **Output** — Write XMP sidecar files with star ratings (1–5) and color labels. Generate ranked browsing folders and a detailed features CSV.

## Scoring ensemble

| Model | Role | Consensus weight |
|---|---|---|
| Q-Align (mPLUG-Owl2 7B) | Scene-level aesthetics, content understanding | 36% |
| DeQA-Score (mPLUG-Owl2 7B) | Holistic quality assessment | 36% |
| MUSIQ | Spatial structure at native resolution | 18% |
| QualiCLIP+ | CLIP-based perceptual quality; disagreement scorer | 9% |
| TOPIQ | Technical quality (sharpness, noise) | Gate only |
| NIMA | Aesthetic prediction | Collected, excluded from consensus |

The two VLM-scale models (Q-Align and DeQA-Score) dominate because they understand image *content*, not just pixels. After the technical gate filters out flawed images, semantic understanding matters more than low-level signal.

## Color labels

- **Red** — Best image in each visual cluster (consensus pick)
- **Purple** — Alternative pick where the disagreement scorer chose differently
- **Green** — Technical gate failure (soft focus, blink, or otherwise technically flawed)

## Limitations

localcull does not consider cropping or zoom potential. An unprocessed wildlife photo that you'd crop in on 4x may be rated more harshly than it would if read in as already cropped. Wildlife and sports shooters especially should keep this in mind.

The scoring models were trained on conventional image quality preferences. Creatively unconventional images (dramatic lighting, intentional underexposure, heavy mood) will tend to score lower than cleaner, more conventional shots. The models reward what the training raters rewarded.

The models are also biased toward faces and human subjects. If you run a mixed shoot (e.g., portraits + landscapes + macro shots) together, landscapes and macros may tend to score lower. Not because they're worse, but because the models favor human subjects. See the User Guide for recommendations on handling mixed shoots.

## Requirements

- macOS with Apple Silicon (tested on M3 Ultra)
- Python 3.11+
- exiftool (`brew install exiftool`)
- **64GB unified memory minimum**

This runs multiple 7B-parameter neural networks locally. Model weights are downloaded on first run and cached (~30GB disk space).

## Quick start

```bash
pip install git+https://github.com/arrynpanagos/localcull.git
localcull /path/to/your/photos
```

Output appears alongside your images: XMP sidecars, a features CSV with per-image scores from all models, and ranked browsing folders.

## darktable integration

localcull includes a Lua plugin that groups images in darktable's lighttable by visual cluster, with the Red pick as group leader. See `localcull_grouper.lua`.

## Documentation

See [USER_GUIDE.md](docs/USER_GUIDE.md) for details on how the ratings work, what the models are trained on, and best practices for different types of shoots.

## License

Elastic License 2.0 (ELv2). Free for personal, educational, research, and commercial photography use. You may not offer localcull itself as a hosted service. See [LICENSE.txt](LICENSE.txt).

## Acknowledgments

localcull builds on work from several research groups:

- **Q-Align**: Wu et al., ICML 2024
- **TOPIQ**: Chen et al., IEEE TIP 2024
- **MUSIQ**: Ke et al., ICCV 2021
- **QualiCLIP+**: Agnolucci et al., 2024
- **DINOv2**: Oquab et al., 2023
- **pyiqa**: Chen & Mo, 2022
- **MediaPipe**: Lugaresi et al., 2019