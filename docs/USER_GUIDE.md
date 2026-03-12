# localcull — User Guide

## Understanding the ratings

### A note on star ratings

This star rating is not how you would rate your own shoots. The purpose isn't to provide final ratings, it's to accelerate your workflow by making differences between images visible faster, so you can make better decisions in less time. You can use the ratings to learn more about your photos and to quickly identify which images the models thought were strongest and weakest within the set.

Stars are relative to the shoot. They tell you where each image falls within the range of scores for this particular batch. A 1★ image in a shoot full of excellent photos might still be a perfectly good image. It just scored lower than the others in this set.

### Color labels

Stars tell you relative quality. Color labels tell you the *role* of each image in the selection process, they work together:

- **Red**: Best image in each visual cluster. This is your edit set. If you only have time to process a subset of the shoot, start with Reds.
- **Purple**: Alternative pick where a second scoring model disagreed with the consensus. Worth reviewing, sometimes the alternative is actually better for your intent.
- **Green**: Technical failure detected (soft focus, blink, or otherwise technically flawed). Usually safe to skip, but check a few to make sure the gate isn't being too aggressive for your style.
- **No label**: Not selected, but not technically flawed. The models ranked other images from the same cluster higher.

### The features CSV

Every run produces a `*_features.csv` alongside your images. This contains the raw scores from all six models, cluster assignments, face detection results, and sharpness measurements. If you disagree with a rating, the CSV tells you exactly why the models scored it that way.

---

## How the models work

localcull uses an ensemble of six neural image quality models. Understanding what they're actually trained on helps set realistic expectations.

### The primary models: Q-Align and DeQA-Score

These are the 'heaviest hitters'. They are both built on mPLUG-Owl2, a 7-billion parameter vision-language model (the same class of model as ChatGPT, but trained to look at images). They contribute 72% of the consensus score.

Q-Align was trained on three datasets: KonIQ-10k (10,073 real-world photos rated by ~1,460 crowd workers who produced over 1.2 million quality judgments), SPAQ (11,125 smartphone photos scored in a controlled lab), and KADID-10k (synthetic distortions). For aesthetic scoring, it was additionally trained on the AVA dataset, 236,000 images rated for aesthetic appeal.

The workers who rated these images were screened for reliability and instructed to evaluate "technical image quality", considering sharpness, noise, exposure, color, and artifacts. Each image received at least 120 independent ratings. The final score is the mean opinion.

Because these models are built on a vision-language foundation, they understand image *content*, not just pixel-level statistics. They can distinguish between a sharp portrait with good eye contact and a sharp photo of an empty wall. This is why they're weighted heavily in the consensus.

DeQA-Score uses the same mPLUG-Owl2 backbone with different training, providing a second VLM-scale opinion.

### Supporting models

**MUSIQ** (18% weight): A multi-scale transformer that processes images at their native resolution and aspect ratio. Good at catching compositional and structural issues that the VLMs sometimes miss. Trained on KonIQ-10k.

**QualiCLIP+** (9% weight, also the "disagreement scorer"): Built on OpenAI's CLIP model with a learned quality projection head. Trained on KonIQ-10k. It sees images differently from the VLMs, it's the model most likely to disagree with the consensus, which is why localcull uses it to generate alternative picks (Purple labels) when it picks a different image than the consensus.

**TOPIQ** (gate only, 0% consensus weight): A ResNet50-based model that's very good at detecting technical flaws, soft focus, noise, artifacts. It doesn't contribute to ranking but is used as a pass/fail technical gate. Images below its threshold get flagged Green.

**NIMA** (collected, 0% weight): An older aesthetic predictor. Scores are recorded in the CSV for reference but don't influence ratings or selection.

### What this means in practice

The scores reflect what ~1,500 human raters, across multiple studies, considered to be good image quality and aesthetics. The models have internalized those preferences at scale. They're good at the things those raters were good at. This includes sharpness, exposure, color, composition, expression. They're blind to the things those raters weren't asked about, e.g., narrative significance, client preferences, or your personal style.

---

## Best practices

### Keep your analysis sets consistent

localcull works best when the images it's comparing are trying to do the same thing. The scoring models have different strengths:

- **Portraits**: Models excel at comparing eye sharpness, expression, and face quality within a burst.
- **Landscapes**: The VLM-scale models (Q-Align, DeQA-Score) understand composition, light, and scene aesthetics well. Where they fall short is comparing a landscape against a portrait, they're good within a category, less reliable across categories.
- **Candids/event**: Models catch blinks and soft focus reliably, but can't distinguish "important moment" from "random hallway shot."

**Recommendation:** If your shoot mixes very different content (e.g., ceremony + reception + macro shots + portraits at a wedding), consider placing photos of different types into separate folders and running localcull on each folder independently. This gives each category its own rating scale, so your landscape shots aren't competing against your portraits for stars.

If you run everything together, landscapes and macro shots will tend to score lower than portraits, not because they're worse, but because the models are biased toward faces and human subjects.

### Trust the clusters, audit the ratings

The visual clustering (grouping similar images together) and the Red/Purple picks are the most reliable part of localcull. The models are very good at identifying which frame in a burst has the sharpest focus, best expression, and cleanest composition.

The star ratings are useful for *sorting within the edit set* but less reliable for absolute quality judgments. Use stars to prioritize your editing order, not to decide what to keep or delete.

### Check the Greens

Images flagged Green (technical failure) are caught by thresholds on sharpness, focus quality, and blink detection. These thresholds are conservative and they aim to catch obviously flawed images rather than borderline ones.

Occasionally, intentionally creative images (motion blur, shallow depth of field on non-face subjects, dramatic lighting) may trigger the technical gate. If your style involves a lot of intentional "imperfection," review the Green-labeled images to make sure you're not losing keepers.

### Review the Purple alternatives

When the consensus scoring model and the disagreement scorer (QualiCLIP+) pick different images as best in a cluster, both are kept, Red for the consensus pick, Purple for the alternative.

Purple images are worth reviewing because they represent a genuinely different aesthetic perspective. The consensus favors semantic content (expression, moment, composition). QualiCLIP+ favors technical photographic quality (sharpness, color, exposure). When they disagree, you can review if that disagreement highlights anything notable.

### Large shoots benefit most

On a 36-image set where you've already hand-picked diverse compositions, localcull won't dramatically reduce your workload, most images are already unique.

The real value shows on 500+ image shoots with lots of bursts and repetitive compositions. A 2,000-image wedding that reduces to 200 Red picks with clear cluster grouping saves hours of scrolling.

### Your photos never leave your computer

localcull runs entirely on your machine. No images are uploaded to any server. The AI models run locally on your Apple Silicon GPU. Your client work stays private.

This does mean you need sufficient hardware, the models are large (multiple 7B-parameter neural networks). 64GB unified memory is a good minimum.

---

## Workflow ideas

localcull produces XMP sidecars (star ratings + color labels), a features CSV with full per-image scoring detail, and ranked browsing folders with symlinks. These work with any editor that reads XMP, Lightroom, Capture One, darktable, and others. How you integrate them is up to you.

A few approaches that can help:

- **Start with Red labels** to see the edit set, then explore from there.
- **Browse the ranked folders** in Finder for a quick visual overview before importing into your editor.
- **Expand the clusters in darktable** (via the Lua plugin) to see what each pick was chosen from and swap in alternatives if you prefer.
- **Check a few Greens** if your style involves intentional motion blur, shallow DOF, or other creative "imperfections" that might trigger the technical gate.
- **Glance at the Purple alternatives** when two models disagree on the best image, sometimes the alternative is the one you'd actually pick.
