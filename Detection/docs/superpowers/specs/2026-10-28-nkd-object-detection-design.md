# NKD Object Detection Design

## Goal

Add Normalized Knowledge Distillation (NKD) as an independent object-detection
method. The implementation must not change the behavior or parameter values of
any existing distillation method.

## Confirmed settings

- Method type: `NKD`
- Temperature: `1.0`, matching the project's KD setting and the paper's
  ImageNet setting
- Gamma: `1.5`
- Distillation loss weight: `1.0`, matching the project's KD weight
- Architectures: R18-R101, R50-R101, and MV2-R50
- Scope: bounding-box detection only; no Mask R-CNN or ReviewKD combination

## Detection adaptation

The provided reference implementation applies NKD to image-classification
logits. For Faster R-CNN, NKD will instead operate on the classification logits
of sampled ROI proposals. Student and teacher will process the same
`sampled_proposals` through their existing ROI pooler, box head, and box
predictor. The proposal `gt_classes` values, including the background class,
will be used as NKD target labels.

The detector's existing supervised ROI classification, box regression, and RPN
losses remain unchanged. NKD adds only `loss_nkd`.

## Loss

The target-class term follows the provided `NKD.py`: the teacher target-class
probability weights the negative student target-class log probability. The
target term is calculated without temperature scaling.

For the non-target term, the target class is removed from both logits. Student
and teacher non-target logits are independently normalized with softmax using
temperature `T`. Their cross-entropy is multiplied by `GAMMA * T^2`. The final
NKD loss is the sum of target and non-target terms, then multiplied by
`LOSS_WEIGHT`.

Teacher logits are produced under `torch.no_grad()`. Only the student receives
gradients from `loss_nkd`.

## Code boundaries

- Add `model/nkd.py` containing the standalone tensor-only NKD loss.
- Add only an `NKD` import and independent route in `model/rcnn.py`.
- Reuse the existing `forward_pure_roi_head`; do not change network structure.
- Add only `KD.NKD` defaults in `model/config.py`.
- Add three standalone files under `configs/NKD/`.
- Do not modify existing method branches or configuration values.

## Configuration

Each NKD configuration will contain:

```yaml
KD:
  TYPE: "NKD"
  NKD:
    T: 1.0
    GAMMA: 1.5
    LOSS_WEIGHT: 1.0
```

Training schedules, pretrained weights, student backbones, and teacher
backbones will follow the corresponding existing LSKD/SDD architecture
configuration.

## Verification

- Unit-test the loss against a direct calculation of the provided reference
  formula.
- Verify both integer class labels and one-hot labels.
- Verify finite student gradients and no teacher gradients.
- Check that the NKD route uses identical sampled proposals for student and
  teacher and adds only `loss_nkd`.
- Check the confirmed default values and all three architecture configurations.
- Compile changed Python files and run the complete test suite.
- Treat the four previously identified baseline expectation mismatches as
  pre-existing failures; any additional failure is a regression.

## Run outputs

The three configurations will write to separate directories:

- `output/NKD-R18-R101`
- `output/NKD-R50-R101`
- `output/NKD-MV2-R50`
