# SDD Object Detection Design

## Goal

Add Scale Decoupled Distillation (SDD) as an independent object-detection
method. The implementation must not change the behavior or configuration of
any existing distillation method.

## Confirmed settings

- Method type: `SDD`
- Region scales: `[1, 2]`
- Temperature: `1.0`, matching the project's KD setting
- Distillation loss weight: `1.0`, matching the project's KD setting
- Complementary-term weight: `2.0`, matching the official SDD implementation
- Architectures: R18-R101, R50-R101, and MV2-R50

## Detection adaptation

The official SDD implementation applies the same classifier to global and
spatially pooled local image features. Faster R-CNN has no image-level
classifier at that point, so SDD will operate on each sampled proposal instead.

For every proposal, scale 1 contributes the original proposal and scale 2
contributes its four non-overlapping quadrants. All five regions are processed
by each network's existing ROI pooler, box head, and box predictor. Student and
teacher therefore use the same proposal regions while retaining their own
model parameters. No backbone, ROI-head, or existing KD implementation is
modified.

The generated logits are arranged as `[proposal, class, region]`, with the
global region first. Proposal ground-truth classes are repeated across the
five regions.

## Loss

For each proposal and region, calculate temperature-scaled KL divergence from
teacher to student and multiply by `T^2`. A local term is complementary when
the teacher's global and local predictions differ in correctness relative to
the proposal label. Complementary local terms receive weight `2.0`; consistent
terms receive weight `1.0`. The global term receives weight `1.0`. The final
SDD loss is the mean over proposals and regions, multiplied by the configured
loss weight.

## Code boundaries

- Add `model/sdd.py` for region construction, logit reshaping, and SDD loss.
- Add only an `SDD` import, ROI forwarding helper, and route in
  `model/rcnn.py`.
- Add only `KD.SDD` defaults in `model/config.py`.
- Add three standalone files under `configs/SDD/`.
- Do not change existing methods or their configuration values.

## Verification

- Unit-test exact region coordinates and ordering.
- Unit-test the loss against a direct implementation of the official formula.
- Check finite gradients and the confirmed configuration defaults.
- Check that the SDD route uses the same sampled proposals for student and
  teacher and adds only `loss_sdd`.
- Check all three architecture configurations.
- Run the new tests, syntax compilation, and the full existing test suite;
  report pre-existing failures separately from new regressions.
