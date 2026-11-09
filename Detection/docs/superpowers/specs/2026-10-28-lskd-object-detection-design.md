# LSKD Object Detection Design

## Goal

Add Logit Standardization Knowledge Distillation (LSKD) to the object-detection
training path as a new, isolated method. The implementation follows the paper
and official `KD + Ours` code: standardize teacher and student classification
logits independently, then compute the ordinary temperature-scaled KD loss.
Existing distillation methods must remain unchanged.

## Confirmed interpretation

`LSKD` means the detector's existing supervised losses plus one standardized KD
loss. It does not add both an unstandardized KD loss and a standardized KD loss.

For every sampled ROI, let `z_s` and `z_t` be the student and teacher class-logit
vectors. Each vector is standardized independently across its class dimension:

```text
Z(z) = (z - mean(z)) / (std(z) + 1e-7)
```

The new loss is:

```text
loss_lskd = LOSS_WEIGHT * T^2
            * KL(softmax(Z(z_t) / T) || softmax(Z(z_s) / T))
```

The implementation follows the official code's standard-deviation behavior and
`1e-7` numerical stabilizer. Teacher probabilities are targets; gradients flow
through the student side only under the project's existing frozen-teacher setup.

## Architecture and isolation

Create `model/lskd.py` as the sole owner of LSKD math. It exposes a focused
normalization helper and `lskd_loss(student_logits, teacher_logits, temperature)`.
The function returns the unweighted temperature-scaled KL loss so the detector
route applies the configured method weight explicitly.

Add an `LSKD` configuration node in `model/config.py`:

```yaml
KD:
  LSKD:
    T: 1.0
    LOSS_WEIGHT: 1.0
```

These values intentionally match the current project's KD defaults rather than
the classification-only hyperparameters shown in the official LSKD examples.

Add one new `KD.TYPE == "LSKD"` branch in `RCNNKD.forward`. The branch:

1. Runs the teacher backbone using the project's existing teacher path.
2. Sends the same `sampled_proposals` through student and teacher ROI box heads.
3. Extracts the aligned student and teacher classification logits.
4. Computes `loss_lskd` from `model/lskd.py`.
5. Adds only `loss_lskd` to the detector losses.

The existing RPN, ROI classification, and box-regression losses remain active.
No existing KD branch, loss implementation, or experiment configuration is
modified.

## Experiment configurations

Create three object-detection configurations under `configs/LSKD/`:

- `LSKD-R18-R101.yaml`
- `LSKD-R50-R101.yaml`
- `LSKD-MV2-R50.yaml`

They mirror the corresponding architecture, weights, optimizer, schedule, and
dataset settings already used by this project. Their only method-specific block
is:

```yaml
KD:
  TYPE: "LSKD"
  LSKD:
    T: 1.0
    LOSS_WEIGHT: 1.0
```

Each configuration has a unique `output/LSKD-<architecture>` directory. These
are detection-only configurations, so `MODEL.MASK_ON` remains false and the
existing best-checkpoint hook selects `model_best.pth` by `bbox/AP`.

## Numerical behavior

- Standardization is per ROI and across all ROI classifier outputs, including
  Detectron2's background class, because each ROI logit vector is the detection
  analogue of one classification sample.
- Teacher and student statistics are computed independently.
- The `1e-7` denominator stabilizer prevents division by zero for constant
  logits.
- Logits are not standardized in place.
- The loss uses the official reduction: sum KL terms across classes, then mean
  across sampled ROIs, followed by multiplication by `T^2`.

## Verification

Use test-driven development and add focused tests before production code:

1. Compare `lskd_loss` with a direct implementation of the official formula.
2. Verify invariance to per-ROI shifts and positive rescaling, within numerical
   tolerance.
3. Verify constant logits produce a finite loss and finite student gradients.
4. Verify the `LSKD` route uses identical sampled proposals for student and
   teacher and adds only `loss_lskd`.
5. Verify all three configurations contain the confirmed defaults and correct
   architecture-specific settings.
6. Run the complete existing unit-test suite and compile changed Python files.

Full COCO/CUDA training is outside local unit verification. The first real run
must confirm that `loss_lskd` is finite, evaluation emits `bbox/AP`, and
`model_best.pth` is selected by the highest `bbox/AP`.

## Out of scope

- Adding LSKD to instance segmentation.
- Combining LSKD with DKD, SDKD, ReviewKD, or other methods.
- Changing existing method defaults, loss weights, routes, or YAML files.
- Tuning LSKD hyperparameters for object detection.
- Refactoring the shared training architecture.
