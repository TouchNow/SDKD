# Task-Aware Best Checkpoint and Final Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Select detection checkpoints by `bbox/AP`, select Mask R-CNN checkpoints by `segm/AP`, and automatically evaluate `model_best.pth` after training.

**Architecture:** Keep the existing Detectron2 `BestCheckpointer` and derive its metric from `cfg.MODEL.MASK_ON`. Add one focused trainer class method that synchronizes workers, loads only the best model parameters, runs the existing evaluator, and lets `main()` return those results with a safe fallback when no best checkpoint exists.

**Tech Stack:** Python 3, Detectron2 0.6, PyTorch distributed utilities through `detectron2.utils.comm`, standard-library `unittest` and `ast`

---

## File map and constraints

- Modify `tests/test_training_eval_best_checkpoint.py`: extend dependency-free AST coverage for task-aware metric selection and final best-checkpoint evaluation.
- Modify `train_net.py`: derive the best metric, add `Trainer.test_best_checkpoint()`, and call it after `trainer.train()`.
- Do not modify SDKD, ReviewSDKD, Mask R-CNN, evaluator, dataset, or YAML configuration files.
- Preserve the current `--eval-only`, TTA, periodic evaluation, periodic checkpoint, and distributed launch paths.
- Git commit steps are omitted because `/Users/touchnow/Documents/Pycharm/Detection` has no `.git` repository.

### Task 1: Select the best metric by task

**Files:**
- Modify: `tests/test_training_eval_best_checkpoint.py:67-93`
- Modify: `train_net.py:70-83`

- [ ] **Step 1: Replace the fixed-bbox regression with a task-aware failing test**

Replace `test_best_hook_monitors_bbox_ap_and_is_after_evaluation()` in `BestCheckpointHookTests` with:

```python
    def test_best_hook_selects_task_metric_and_is_after_evaluation(self):
        metric_assignment = next(
            node
            for node in ast.walk(self.method)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "val_metric"
                for target in node.targets
            )
        )
        self.assertIsInstance(metric_assignment.value, ast.IfExp)
        self.assertEqual(
            ast.unparse(metric_assignment.value.test),
            "self.cfg.MODEL.MASK_ON",
        )
        self.assertEqual(ast.literal_eval(metric_assignment.value.body), "segm/AP")
        self.assertEqual(ast.literal_eval(metric_assignment.value.orelse), "bbox/AP")

        insert_call = next(
            node
            for node in ast.walk(self.method)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "insert"
        )
        self.assertEqual(ast.literal_eval(insert_call.args[0]), -1)

        best_call = insert_call.args[1]
        self.assertEqual(ast.unparse(best_call.func), "hooks.BestCheckpointer")
        keywords = {keyword.arg: keyword.value for keyword in best_call.keywords}
        self.assertEqual(
            ast.unparse(keywords["eval_period"]),
            "self.cfg.TEST.EVAL_PERIOD",
        )
        self.assertEqual(
            ast.unparse(keywords["checkpointer"]),
            "self.checkpointer",
        )
        self.assertEqual(ast.unparse(keywords["val_metric"]), "val_metric")
        self.assertEqual(ast.literal_eval(keywords["mode"]), "max")
        self.assertEqual(
            ast.literal_eval(keywords["file_prefix"]),
            "model_best",
        )
```

- [ ] **Step 2: Run the hook test and verify the RED state**

Run:

```bash
python3 -m unittest \
  tests.test_training_eval_best_checkpoint.BestCheckpointHookTests.test_best_hook_selects_task_metric_and_is_after_evaluation \
  -v
```

Expected: ERROR with `StopIteration` while looking for the `val_metric` assignment, because `build_hooks()` still passes the literal `"bbox/AP"`.

- [ ] **Step 3: Implement the minimal task-aware metric selection**

Replace `Trainer.build_hooks()` in `train_net.py` with:

```python
    def build_hooks(self):
        hook_list = super().build_hooks()
        val_metric = "segm/AP" if self.cfg.MODEL.MASK_ON else "bbox/AP"
        if comm.is_main_process() and self.cfg.TEST.EVAL_PERIOD > 0:
            hook_list.insert(
                -1,
                hooks.BestCheckpointer(
                    eval_period=self.cfg.TEST.EVAL_PERIOD,
                    checkpointer=self.checkpointer,
                    val_metric=val_metric,
                    mode="max",
                    file_prefix="model_best",
                ),
            )
        return hook_list
```

- [ ] **Step 4: Run the hook class and verify the GREEN state**

Run:

```bash
python3 -m unittest \
  tests.test_training_eval_best_checkpoint.BestCheckpointHookTests \
  -v
```

Expected: 2 tests pass.

### Task 2: Automatically evaluate the best checkpoint after training

**Files:**
- Modify: `tests/test_training_eval_best_checkpoint.py`
- Modify: `train_net.py` inside `Trainer` and `main(args)`

- [ ] **Step 1: Add a top-level function AST helper to the regression test**

Add this helper immediately after `_trainer_method()` in `tests/test_training_eval_best_checkpoint.py`:

```python
def _top_level_function(function_name):
    tree = ast.parse(TRAIN_PATH.read_text())
    return next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == function_name
    )
```

- [ ] **Step 2: Add failing tests for the helper and training handoff**

Add this class before the final `if __name__ == "__main__"` block:

```python
class FinalBestCheckpointEvaluationTests(unittest.TestCase):
    def test_helper_synchronizes_loads_and_tests_best_checkpoint(self):
        method = _trainer_method("test_best_checkpoint")
        source = ast.unparse(method)

        self.assertIn("comm.synchronize()", source)
        self.assertIn(
            "best_checkpoint = os.path.join(cfg.OUTPUT_DIR, 'model_best.pth')",
            source,
        )
        self.assertIn("os.path.isfile(best_checkpoint)", source)
        self.assertIn("return None", source)
        self.assertIn(
            "DetectionCheckpointer(model).load(best_checkpoint)",
            source,
        )
        self.assertIn("return cls.test(cfg, model)", source)

    def test_training_path_returns_best_results_with_fallback(self):
        main_method = _top_level_function("main")
        source = ast.unparse(main_method)

        self.assertIn("train_results = trainer.train()", source)
        self.assertIn(
            "best_results = Trainer.test_best_checkpoint(cfg, trainer.model)",
            source,
        )
        self.assertIn(
            "return best_results if best_results is not None else train_results",
            source,
        )
```

- [ ] **Step 3: Run the new class and verify the RED state**

Run:

```bash
python3 -m unittest \
  tests.test_training_eval_best_checkpoint.FinalBestCheckpointEvaluationTests \
  -v
```

Expected: the helper test errors because `Trainer.test_best_checkpoint()` does not exist, and the training-path test fails because `main()` still directly returns `trainer.train()`.

- [ ] **Step 4: Add the minimal best-checkpoint evaluation helper**

Add this method to `Trainer` immediately before `test_with_TTA()`:

```python
    @classmethod
    def test_best_checkpoint(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        comm.synchronize()
        best_checkpoint = os.path.join(cfg.OUTPUT_DIR, "model_best.pth")
        if not os.path.isfile(best_checkpoint):
            if comm.is_main_process():
                logger.warning(
                    "Best checkpoint not found at %s; keeping final training results.",
                    best_checkpoint,
                )
            return None
        if comm.is_main_process():
            logger.info(
                "Loading best checkpoint from %s for final evaluation.",
                best_checkpoint,
            )
        DetectionCheckpointer(model).load(best_checkpoint)
        return cls.test(cfg, model)
```

- [ ] **Step 5: Replace the direct training return with the best-result handoff**

Replace the final line of the training path in `main(args)`:

```python
    return trainer.train()
```

with:

```python
    train_results = trainer.train()
    best_results = Trainer.test_best_checkpoint(cfg, trainer.model)
    return best_results if best_results is not None else train_results
```

- [ ] **Step 6: Run the new class and verify the GREEN state**

Run:

```bash
python3 -m unittest \
  tests.test_training_eval_best_checkpoint.FinalBestCheckpointEvaluationTests \
  -v
```

Expected: 2 tests pass.

### Task 3: Regression and verification

**Files:**
- Verify: `train_net.py`
- Verify: `tests/test_training_eval_best_checkpoint.py`
- Verify unchanged: SDKD, ReviewSDKD, Mask configurations, and `configs/Base-Distillation.yaml`

- [ ] **Step 1: Run the complete focused regression module**

Run:

```bash
python3 -m unittest tests.test_training_eval_best_checkpoint -v
```

Expected: 5 tests pass.

- [ ] **Step 2: Run the complete local test suite**

Run:

```bash
python3 -m unittest discover -s tests -v
```

Expected: 14 tests pass with no failures or errors.

- [ ] **Step 3: Compile the modified Python files**

Run:

```bash
python3 -m py_compile train_net.py tests/test_training_eval_best_checkpoint.py
```

Expected: exit status 0 with no output.

- [ ] **Step 4: Inspect the exact implementation surface**

Run:

```bash
rg -n \
  "val_metric|test_best_checkpoint|train_results|best_results|model_best\\.pth" \
  train_net.py tests/test_training_eval_best_checkpoint.py
```

Expected: changes are confined to the task-aware hook, final best-checkpoint evaluator, training handoff, and their focused tests.

- [ ] **Step 5: Record the runtime verification boundary**

Report local test and compilation results. Do not claim end-to-end COCO/CUDA success until a real training run confirms that a Mask experiment logs `segm/AP` best-checkpoint decisions and ends with a second evaluation block after loading `model_best.pth`.
