# Mask SDKD and ReviewSDKD Design

## Goal

Add instance-segmentation configurations for SDKD and ReviewSDKD while preserving every existing source file and configuration. The student keeps the standard Mask R-CNN `loss_mask`; SDKD distills only ROI classification logits, and ReviewSDKD additionally applies the existing ReviewKD feature loss.

## Selected approach

Reuse the existing `RCNNKD` routes selected by `KD.TYPE: "SDKD"` and `KD.TYPE: "ReviewSDKD"`. Enable instance segmentation through `MODEL.MASK_ON: True` in six new YAML files under `configs/SDKD/`.

This approach is intentionally configuration-only for production behavior:

- `MODEL.MASK_ON` controls whether Detectron2 builds and trains the student mask head.
- `KD.TYPE` controls which existing distillation losses are added after the normal ROI-head losses.
- Mask and detection experiments remain separate through `-Mask.yaml` filenames and distinct `OUTPUT_DIR` values.
- No mask-logit distillation is introduced.

Creating new `SDKDMask` or `ReviewSDKDMask` method types is rejected because it would require editing existing routing code without changing the underlying loss behavior. Creating a second RCNN meta-architecture is rejected because it would duplicate the training path.

## Training data flow

### SDKD-Mask

1. The student backbone, RPN, box head, and mask head run through the existing `RCNNKD` training path.
2. The student ROI heads produce the ordinary detection losses and `loss_mask`.
3. The frozen teacher backbone and teacher ROI box predictor run on the same sampled proposals.
4. The existing `sdkd_loss` adds ROI-classification-logit losses using `KD.TYPE: "SDKD"`.
5. Detectron2 sums the detector, proposal, mask, and SDKD losses.

The teacher mask head is not used because mask logits are not distilled.

### ReviewSDKD-Mask

1. The student produces the ordinary detection losses and `loss_mask` as above.
2. The existing ReviewSDKD route applies `sdkd_loss` to student and teacher ROI classification logits.
3. The existing ReviewKD adapter transforms student FPN features, and `hcl` compares them with teacher FPN features.
4. Detectron2 sums detector, proposal, mask, SDKD, and ReviewKD losses.

## New configuration files

All six files are created under `configs/SDKD/`:

| File | `KD.TYPE` | Student/teacher | Pretrained weights | Output directory |
|---|---|---|---|---|
| `SDKD-R18-R101-Mask.yaml` | `SDKD` | R18/R101 | `pretrained/r18-r101mask.pth` | `output/SDKD-R18-R101-Mask` |
| `SDKD-R50-R101-Mask.yaml` | `SDKD` | R50/R101 | `pretrained/r50-r101mask.pth` | `output/SDKD-R50-R101-Mask` |
| `SDKD-MV2-R50-Mask.yaml` | `SDKD` | MV2/R50 | `pretrained/mv2-r50mask.pth` | `output/SDKD-MV2-R50-Mask` |
| `ReviewSDKD-R18-R101-Mask.yaml` | `ReviewSDKD` | R18/R101 | `pretrained/r18-r101mask.pth` | `output/ReviewSDKD-R18-R101-Mask` |
| `ReviewSDKD-R50-R101-Mask.yaml` | `ReviewSDKD` | R50/R101 | `pretrained/r50-r101mask.pth` | `output/ReviewSDKD-R50-R101-Mask` |
| `ReviewSDKD-MV2-R50-Mask.yaml` | `ReviewSDKD` | MV2/R50 | `pretrained/mv2-r50mask.pth` | `output/ReviewSDKD-MV2-R50-Mask` |

Each file inherits `../Base-Distillation.yaml`, sets `MODEL.MASK_ON: True`, and copies the corresponding student, teacher, and solver structure already used by the matching SDKD and ReviewKD-Mask configurations.

## Distillation parameters

All SDKD-Mask files use the existing SDKD detection parameters:

```yaml
KD:
  TYPE: "SDKD"
  SDKD:
    INARANK_WEIGHT: 0.1
    T: 1.0
    HEAD_WEIGHT: 1.0
    TAIL_WEIGHT: 0.25
```

All ReviewSDKD-Mask files use the existing ReviewSDKD SDKD parameters:

```yaml
SDKD:
  INARANK_WEIGHT: 0.5
  T: 1.0
  HEAD_WEIGHT: 1.0
  TAIL_WEIGHT: 0.25
```

Their ReviewKD feature-loss weights come from the matching existing ReviewKD-Mask experiments:

| Student/teacher | `REVIEWKD.LOSS_WEIGHT` |
|---|---:|
| R18/R101 | 1.5 |
| R50/R101 | 0.8 |
| MV2/R50 | 1.0 |

## File boundaries

Create only:

- The six YAML files listed above.
- `tests/test_mask_sdkd_configs.py`, a focused configuration regression test.
- This design document and the subsequent implementation plan.

Do not modify:

- `model/rcnn.py`
- `model/sdkd.py`
- `model/reviewkd.py`
- `model/config.py`
- Any existing YAML or test file

## Validation and failure handling

The new test parses all six YAML files and checks:

- The file exists and inherits `../Base-Distillation.yaml`.
- `MODEL.MASK_ON` is `True`.
- `KD.TYPE`, SDKD parameters, and optional ReviewKD weight are exact.
- Student backbone/depth, teacher depth, mask checkpoint, and output directory match the architecture pair.
- Every new output directory is unique and ends in `-Mask`.
- Existing `RCNNKD` contains the reused SDKD and ReviewSDKD loss routes, without modifying them.

Verification consists of the focused unit test, the existing unit-test suite, Python compilation for the new test, YAML parsing for all six configurations, and a final file inventory confirming that no pre-existing file changed. Full COCO training is outside local verification because it requires Detectron2, the dataset, pretrained checkpoints, and GPUs; successful static and unit checks do not claim training accuracy.
