# Task-Aware Best Checkpoint and Final Evaluation Design

## Goal

Select `model_best.pth` with the metric appropriate to the configured task, then automatically reload and evaluate that checkpoint after training so the final printed results describe the best model rather than merely the last iteration.

This design extends `2026-08-09-training-eval-best-checkpoint-design.md`, where instance-segmentation selection and final best-checkpoint evaluation were explicitly out of scope.

## Selected approach

Extend the existing `Trainer` in `train_net.py` without introducing a second trainer, an external evaluation subprocess, or new configuration keys.

- Detection configurations keep `bbox/AP` as the best-checkpoint metric.
- Configurations with `MODEL.MASK_ON: True` use `segm/AP` as the best-checkpoint metric.
- After `trainer.train()` finishes, the same distributed job reloads `OUTPUT_DIR/model_best.pth` and performs one regular COCO evaluation.

This keeps existing experiment files compatible and limits implementation changes to `train_net.py` and its focused regression test.

## Best-checkpoint selection

`Trainer.build_hooks()` retains the superclass hook list and current placement of `BestCheckpointer` after evaluation and before metric writing. It derives the monitored metric from the frozen configuration:

```python
val_metric = "segm/AP" if self.cfg.MODEL.MASK_ON else "bbox/AP"
```

The remaining behavior is unchanged:

- Evaluation runs every `cfg.TEST.EVAL_PERIOD` iterations.
- The best metric is maximized.
- The checkpoint is saved as `OUTPUT_DIR/model_best.pth`.
- Only the main process runs `BestCheckpointer` and writes the file.

The selected checkpoint for a Mask R-CNN experiment is therefore the checkpoint with the highest mask AP, not necessarily the highest box AP. Its final evaluation still reports both `bbox` and `segm` metrics through `COCOEvaluator`.

## Final best-checkpoint evaluation

Add a focused `Trainer.test_best_checkpoint(cfg, model)` class method with this flow:

1. Synchronize distributed workers after training hooks finish.
2. Resolve `OUTPUT_DIR/model_best.pth`.
3. If the file does not exist, log a warning on the main process and return `None`.
4. Load only model parameters through a new `DetectionCheckpointer(model)` so optimizer, scheduler, and trainer iteration state are not restored.
5. Log that final best-checkpoint evaluation is starting.
6. Call the existing `Trainer.test(cfg, model)` on every worker and return its results.

Update the training path in `main(args)`:

1. Store the result of `trainer.train()` as `train_results`.
2. Call `Trainer.test_best_checkpoint(cfg, trainer.model)`.
3. Return the best-checkpoint results when available; otherwise return `train_results`.

The existing `--eval-only` path remains unchanged. Existing TTA hooks also remain unchanged; the automatic best-checkpoint evaluation is the standard non-TTA COCO evaluation.

## Error and distributed behavior

- All workers synchronize before checking and loading the checkpoint, preventing evaluation from racing the main-process save.
- The current single-machine multi-GPU workflow uses a shared output directory, so every worker can load the same path.
- Missing `model_best.pth` is non-fatal. This can occur when evaluation is disabled, the evaluator does not emit the requested metric, or training stops before a valid evaluation.
- Evaluation exceptions are not swallowed; dataset, checkpoint-corruption, or model errors remain visible instead of being reported as a successful best-model evaluation.

## Files and compatibility

Modify only:

- `train_net.py`
- `tests/test_training_eval_best_checkpoint.py`

Create only this design document and the corresponding implementation plan.

Do not modify SDKD, ReviewSDKD, Mask R-CNN, evaluator, dataset, or experiment YAML files. Existing non-Mask experiments continue to select by `bbox/AP`; Mask experiments automatically switch to `segm/AP` through their existing `MODEL.MASK_ON: True` setting.

## Verification

Extend the dependency-free AST regression tests before implementation to verify:

- `build_hooks()` chooses `segm/AP` for Mask models and `bbox/AP` otherwise.
- `BestCheckpointer` receives the derived metric while retaining its current mode, filename, guards, and hook position.
- The final training path stores `trainer.train()` results, calls the best-checkpoint evaluator, and falls back when no best result is available.
- The helper synchronizes workers, targets `model_best.pth`, checks for absence, loads through `DetectionCheckpointer`, and invokes `Trainer.test`.

Run the focused tests in RED and GREEN states, then run the full local unit-test suite and compile the modified Python files. Full COCO/CUDA training remains a runtime verification boundary; the next real Mask run must confirm that logs report `segm/AP` best-checkpoint decisions followed by a final best-checkpoint evaluation block containing both `bbox` and `segm` results.
