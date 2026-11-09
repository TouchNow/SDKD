# Training-Time Evaluation and Best Checkpoint Design

## Goal

Run object-detection evaluation during training and retain the checkpoint with
the highest COCO `bbox/AP` score.

## Scope

- Evaluate the configured validation dataset every 10,000 training iterations.
- Use Detectron2 0.6's existing `COCOEvaluator`, `EvalHook`, and
  `BestCheckpointer` implementations.
- Save the best checkpoint as `model_best.pth` in `cfg.OUTPUT_DIR`.
- Preserve the existing final and periodic checkpoint behavior.
- Apply the behavior to configurations inheriting from
  `configs/Base-Distillation.yaml`.

Instance-segmentation `segm/AP`, early stopping, and deletion of existing
periodic checkpoints are outside this change.

## Configuration

Add the following setting to `configs/Base-Distillation.yaml`:

```yaml
TEST:
  EVAL_PERIOD: 10000
```

The existing `DATASETS.TEST` value remains `("coco_2017_val",)`, so each
evaluation uses the COCO validation set already configured by the project.

## Trainer Hook Design

Override `Trainer.build_hooks()` in `train_net.py`:

1. Call `super().build_hooks()` to retain Detectron2's existing training,
   checkpoint, evaluation, and writer hooks.
2. When running on the main process and `cfg.TEST.EVAL_PERIOD > 0`, insert a
   `hooks.BestCheckpointer` immediately after the existing evaluation hook and
   before the periodic writer.
3. Configure the best-checkpoint hook with:
   - `eval_period=cfg.TEST.EVAL_PERIOD`
   - `checkpointer=self.checkpointer`
   - `val_metric="bbox/AP"`
   - `mode="max"`
   - `file_prefix="model_best"`

Hook ordering is required because `BestCheckpointer` reads the latest metric
written by `EvalHook`. Only the main process may save the file in distributed
training.

## Runtime Behavior

At iterations 10,000, 20,000, and so on, Detectron2 pauses optimization, runs
inference on the configured validation set, and records COCO metrics. The first
valid `bbox/AP` result creates `model_best.pth`. A later result overwrites that
file only when its `bbox/AP` value is higher.

At the final iteration, the normal final evaluation and checkpoint behavior
remain active. `model_final.pth` is therefore independent of
`model_best.pth`. `SOLVER.CHECKPOINT_PERIOD` continues to control periodic
recovery checkpoints and is not changed by this feature.

If the evaluator does not produce `bbox/AP`, Detectron2's built-in hook logs a
warning and does not save a misleading best checkpoint.

## Verification

Add regression coverage following the repository's existing AST/YAML test
style. Tests will verify that:

- the base configuration sets `TEST.EVAL_PERIOD` to 10,000;
- `Trainer.build_hooks()` retains the superclass hooks;
- `BestCheckpointer` monitors `bbox/AP` in `max` mode;
- the best-checkpoint filename prefix is `model_best`;
- registration is restricted to the main process and enabled evaluation;
- the best hook is placed after evaluation and before metric writers.

Run the complete local unittest suite after the change. Full COCO evaluation
requires the user's Detectron2/CUDA/COCO training environment and is therefore
a separate runtime verification boundary.
