# ReviewSDKD Design

## Goal

Add a new `ReviewSDKD` distillation method that combines the existing ReviewKD feature loss with the existing SDKD logit losses, without changing the behavior of `ReviewKD`, `SDKD`, or other methods.

## Selected approach

Follow the existing `ReviewDKD` composition pattern with surgical additions:

- Add `ReviewSDKD` to the set of method types that initialize `self.kd_trans`.
- Add one independent `ReviewSDKD` forward branch alongside the existing branches.
- Reuse `sdkd_loss`, `build_kd_trans`, and `hcl` without changing their implementations.
- Add explicit ReviewSDKD configuration files for every non-Mask architecture pair already supported by SDKD.

This avoids refactoring shared helpers or creating another Detectron2 meta-architecture, either of which would touch more existing behavior.

## Data flow and losses

For `KD.TYPE: "ReviewSDKD"`:

1. Run the student backbone and normal detector path exactly as before.
2. Run the frozen teacher backbone once.
3. Use student and teacher ROI predictions from the same sampled proposals to compute:
   - `loss_ina_rank = 0.5 * raw_ina_rank`
   - `loss_ahtkd = 1.0 * head_loss + 0.25 * tail_loss`
4. Transform student FPN features with the existing ReviewKD adapter and compute:
   - `loss_reviewkd = REVIEWKD.LOSS_WEIGHT * hcl(student_features, teacher_features)`
5. Return these three distillation losses together with the unchanged detector and proposal losses. Detectron2 sums them for backpropagation.

## Configurations

The supported combinations and ReviewKD feature-loss weights are:

| Student | Teacher | ReviewKD weight | Output directory |
|---|---|---:|---|
| R18 | R101 | 1.2 | `output/ReviewSDKD-R18-R101` |
| R50 | R101 | 1.0 | `output/ReviewSDKD-R50-R101` |
| MV2 | R50 | 2.0 | `output/ReviewSDKD-MV2-R50` |

Each configuration uses:

```yaml
KD:
  TYPE: "ReviewSDKD"
  REVIEWKD:
    LOSS_WEIGHT: 1.2  # R18→R101; use 1.0 for R50→R101 and 2.0 for MV2→R50
  SDKD:
    INARANK_WEIGHT: 0.5
    T: 1.0
    HEAD_WEIGHT: 1.0
    TAIL_WEIGHT: 0.25
```

Each file copies the model, teacher, solver, dataset, and checkpoint settings from the corresponding existing SDKD and ReviewKD configurations. Mask R-CNN variants are out of scope because SDKD has no corresponding Mask configuration.

## Files and compatibility

- Modify `model/rcnn.py` only to add the new initialization route and forward branch.
- Create `configs/SDKD/ReviewSDKD-R18-R101.yaml`, `configs/SDKD/ReviewSDKD-R50-R101.yaml`, and `configs/SDKD/ReviewSDKD-MV2-R50.yaml`.
- Create a focused regression test that verifies routing, three-loss composition, and all three configuration files.
- Do not modify `model/sdkd.py`, `model/reviewkd.py`, existing YAML files, or existing method branches.

## Verification

- Extend the focused regression test to cover all three architecture configurations and confirm it fails because the two additional YAML files are absent.
- Add the two configuration files, then rerun the focused test.
- Compile the touched Python files.
- Parse all three YAML files and verify the architecture, selected type, output directory, ReviewKD weight, and shared SDKD weights.
- Inspect the final diff to ensure every production change belongs only to `ReviewSDKD`.
