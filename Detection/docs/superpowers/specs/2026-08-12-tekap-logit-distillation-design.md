# TeKAP Logit Distillation Design

## Goal

Integrate the logit-level method from *Single Teacher, Multiple Perspectives: Teacher Knowledge Augmentation for Enhanced Knowledge Distillation* into Detection as an independent `KD.TYPE: "TeKAP"` option. The integration supports the existing R18 to R101, R50 to R101, and MobileNetV2 to R50 detector pairs without changing existing distillation methods or inference behavior.

Feature-level TeKAP, combined TeKAP F+L, and the class-mean/rank-noise experimental extension found in `TeKAP-main` are outside this scope.

## Selected Approach

Add a focused, stateless TeKAP loss module and route one new distillation type through the existing `RCNNKD` training path:

- Keep TeKAP math in `model/tekap.py`, independent of Detectron2.
- Add one `TeKAP` branch in `model/rcnn.py`.
- Reuse the existing frozen teacher and the same sampled ROI proposals used by the student detector.
- Add a dedicated `KD.TEKAP` configuration node and three explicit architecture configurations.
- Leave all existing KD branches and the inference path unchanged.

This gives the method a testable boundary while keeping the Detection integration surgical.

## Algorithm and Loss

For a student ROI-classification logit tensor `z_s` and a detached teacher tensor `z_t`, TeKAP creates three synthetic teacher perspectives per training batch:

```text
noise_i     ~ Normal(0, 1)
synthetic_i = (1 - noise_weight) * z_t + noise_weight * noise_i
```

The default settings are:

```yaml
T: 1.0
NUM_AUGMENTED: 3
NOISE_WEIGHT: 0.1
AUGMENTED_WEIGHT: 0.8
LOSS_WEIGHT: 1.0
```

The loss is:

```text
KL_original  = T^2 * KL(softmax(z_t / T) || softmax(z_s / T))
KL_synthetic = sum_i T^2 * KL(softmax(synthetic_i / T) || softmax(z_s / T))
loss_tekap   = LOSS_WEIGHT * (KL_original + AUGMENTED_WEIGHT * KL_synthetic)
```

Each KL term uses PyTorch `batchmean` reduction. This intentionally follows the supplied author implementation's additive form: the original teacher term plus `0.8` times the sum of three synthetic-teacher terms. It does not reinterpret the paper's lambda notation as a normalized convex combination.

Noise is generated with `torch.randn_like`, so it is refreshed on every loss call and inherits the teacher logits' device and dtype. The teacher logits are detached before both the original and synthetic targets are built. Only student logits receive gradients.

## Detection Data Flow

For `KD.TYPE: "TeKAP"`, training proceeds as follows:

1. Run the normal student backbone, proposal generator, and ROI-head training path.
2. Preserve the existing detector classification, box-regression, and RPN losses.
3. Preprocess the same input batch for the frozen teacher and run its backbone.
4. Pass the student and teacher features through their ROI box heads using the same `sampled_proposals` selected by the student ROI heads.
5. Extract the paired ROI classification logits and compute `loss_tekap`.
6. Add only `loss_tekap` to the returned loss dictionary; Detectron2 combines it with the existing detection losses.

TeKAP is training-only. Evaluation and inference continue to use only the student detector and do not generate noise or invoke the teacher.

## Components and Files

### `model/tekap.py`

Create a stateless function with the interface:

```python
def tekap_loss(
    student_logits,
    teacher_logits,
    temperature,
    num_augmented,
    noise_weight,
    augmented_weight,
):
    ...
```

The function returns the unscaled TeKAP scalar loss. `LOSS_WEIGHT` remains an integration-level concern and is applied in `RCNNKD`, matching how existing Detection KD weights are handled.

The function validates:

- student and teacher logits have the same shape;
- temperature is greater than zero;
- `num_augmented` is a positive integer;
- `noise_weight` lies in `[0, 1]`;
- `augmented_weight` is non-negative.

No trainable parameters, persistent statistics, class labels, or Detectron2 dependencies are introduced.

### `model/config.py`

Add the following defaults beneath `cfg.KD`:

```python
cfg.KD.TEKAP = CN()
cfg.KD.TEKAP.T = 1.0
cfg.KD.TEKAP.NUM_AUGMENTED = 3
cfg.KD.TEKAP.NOISE_WEIGHT = 0.1
cfg.KD.TEKAP.AUGMENTED_WEIGHT = 0.8
cfg.KD.TEKAP.LOSS_WEIGHT = 1.0
```

Teacher weight is derived as `1 - NOISE_WEIGHT`; it is not exposed as a second independently configurable value.

### `model/rcnn.py`

- Import `tekap_loss`.
- Add one independent `elif self.kd_args.TYPE == "TeKAP"` training branch.
- Reuse `teacher_preprocess_image`, the frozen teacher backbone, `forward_pure_roi_head`, and `sampled_proposals`.
- Store the weighted result as `losses["loss_tekap"]`.
- Do not change any existing KD branch, loss name, or inference method.

### Configurations

Create:

- `configs/TeKAP/TeKAP-R18-R101.yaml`
- `configs/TeKAP/TeKAP-R50-R101.yaml`
- `configs/TeKAP/TeKAP-MV2-R50.yaml`

Each file inherits `configs/Base-Distillation.yaml`, selects `KD.TYPE: "TeKAP"`, explicitly records the confirmed TeKAP settings, and copies the corresponding architecture, pretrained-weight path, teacher, solver, and dataset behavior from the existing non-Mask configuration for that model pair. Output directories are unique:

- `output/TeKAP-R18-R101`
- `output/TeKAP-R50-R101`
- `output/TeKAP-MV2-R50`

Mask R-CNN variants are not included.

## Error Handling and Compatibility

Invalid standalone loss arguments fail early with `ValueError`; incompatible logit shapes fail with `ValueError`. Configuration keys remain typed through Detectron2/YACS.

The implementation avoids hard-coded CUDA calls and creates noise from the teacher tensor. It therefore supports CPU unit tests and inherits the existing project's single-GPU and distributed-data-parallel behavior. Each DDP rank samples its own local noise, consistent with ordinary stochastic training operations.

No new runtime dependency is required beyond the project's existing PyTorch and Detectron2 stack.

## Testing and Acceptance Criteria

### Numerical unit tests

- With a fixed PyTorch seed, compare `tekap_loss` against a direct reference computation of the original and three synthetic KL terms.
- With `NOISE_WEIGHT=0`, verify every synthetic target equals the original teacher target and the total equals `(1 + NUM_AUGMENTED * AUGMENTED_WEIGHT)` times the base KL loss.
- Verify the student receives finite gradients while the teacher receives no gradient.
- Verify invalid temperature, augmentation count, noise weight, augmented weight, and incompatible shapes are rejected.
- Exercise CPU tensors and at least the default floating-point dtype. If CUDA is available, confirm the function stays on the input device without making CUDA a required test condition.

### Detection integration tests

- Confirm `RCNNKD.forward` has an independent `TeKAP` route.
- Confirm the route obtains teacher and student ROI predictions from the same `sampled_proposals`.
- Confirm the route calls `tekap_loss`, applies `KD.TEKAP.LOSS_WEIGHT`, and adds `loss_tekap`.
- Confirm existing detection loss aggregation and other KD routes remain structurally unchanged.
- Parse all three YAML files and verify architecture identities, weights, output directories, solver settings, `KD.TYPE`, and TeKAP parameters.

### Regression verification

- Write and run focused tests before production code, observing expected failures caused by the missing feature.
- Make the minimum production changes needed for the focused tests to pass.
- Run the complete existing `unittest` suite.
- Run Python syntax compilation on every changed or added Python file.
- Parse all new YAML configurations independently.
- If Detectron2 is importable in the local environment, run a lightweight integration/import check. Full COCO training and AP reproduction are not local completion criteria because they require the dataset, pretrained checkpoints, GPU resources, and a full training run.

## Verification Boundary

Passing local tests will establish that the TeKAP formula, gradient boundary, Detection route, and configurations are implemented as designed. It will not establish an accuracy improvement. Final COCO AP and training stability must be evaluated through user-run training with the supplied checkpoints and dataset.
