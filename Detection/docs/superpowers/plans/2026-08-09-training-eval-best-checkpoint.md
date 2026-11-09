# Training-Time Evaluation and Best Checkpoint Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Evaluate COCO object detection every 10,000 training iterations and retain the checkpoint with the highest `bbox/AP` as `model_best.pth`.

**Architecture:** Keep Detectron2 0.6's existing `DefaultTrainer` evaluation flow. Enable periodic evaluation in the shared base configuration, then extend `Trainer.build_hooks()` with Detectron2's built-in `BestCheckpointer` immediately after `EvalHook` and before the writer hook on the main process only.

**Tech Stack:** Python 3, Detectron2 0.6, YACS YAML configuration, standard-library `unittest` and `ast`.

---

## File Map

- Create `tests/test_training_eval_best_checkpoint.py`: dependency-free regression tests for configuration and hook wiring.
- Modify `configs/Base-Distillation.yaml`: enable evaluation every 10,000 iterations for inherited experiment configurations.
- Modify `train_net.py`: register the built-in best-checkpoint hook without replacing existing hooks.

The current workspace has no Git metadata, so this plan does not initialize a repository or create commits. If the work is later copied into a Git checkout, these three files form one focused commit.

### Task 1: Enable Periodic Evaluation

**Files:**
- Create: `tests/test_training_eval_best_checkpoint.py`
- Modify: `configs/Base-Distillation.yaml:68-78`

- [ ] **Step 1: Write the failing configuration test**

Create `tests/test_training_eval_best_checkpoint.py` with:

```python
import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "train_net.py"
BASE_CONFIG_PATH = ROOT / "configs" / "Base-Distillation.yaml"


def _top_level_yaml_block(path, block_name):
    lines = path.read_text().splitlines()
    start = lines.index(f"{block_name}:") + 1
    end = next(
        (
            index
            for index in range(start, len(lines))
            if lines[index] and not lines[index].startswith(" ")
        ),
        len(lines),
    )
    return lines[start:end]


def _trainer_method(method_name):
    tree = ast.parse(TRAIN_PATH.read_text())
    trainer = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Trainer"
    )
    return next(
        node
        for node in trainer.body
        if isinstance(node, ast.FunctionDef) and node.name == method_name
    )


class TrainingEvaluationConfigTests(unittest.TestCase):
    def test_base_config_evaluates_every_10000_iterations(self):
        test_block = _top_level_yaml_block(BASE_CONFIG_PATH, "TEST")
        self.assertIn("  EVAL_PERIOD: 10000", test_block)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the configuration test and verify it fails**

Run:

```bash
/Users/touchnow/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  -m unittest tests.test_training_eval_best_checkpoint.TrainingEvaluationConfigTests -v
```

Expected: `ERROR` with `ValueError: 'TEST:' is not in list`, because the shared base configuration does not yet contain a top-level `TEST` block.

- [ ] **Step 3: Add the minimal evaluation configuration**

In `configs/Base-Distillation.yaml`, place this block after `DATASETS` and before `SOLVER`:

```yaml
TEST:
  EVAL_PERIOD: 10000
```

- [ ] **Step 4: Run the configuration test and verify it passes**

Run:

```bash
/Users/touchnow/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  -m unittest tests.test_training_eval_best_checkpoint.TrainingEvaluationConfigTests -v
```

Expected: one test runs and reports `OK`.

### Task 2: Save the Highest bbox/AP Checkpoint

**Files:**
- Modify: `tests/test_training_eval_best_checkpoint.py`
- Modify: `train_net.py:52-125`

- [ ] **Step 1: Write the failing hook-wiring tests**

Add the following class before the `if __name__ == "__main__"` block in `tests/test_training_eval_best_checkpoint.py`:

```python
class BestCheckpointHookTests(unittest.TestCase):
    def setUp(self):
        self.method = _trainer_method("build_hooks")
        self.source = ast.unparse(self.method)

    def test_best_hook_keeps_default_hooks_and_has_runtime_guards(self):
        self.assertIn("super().build_hooks()", self.source)
        self.assertIn("comm.is_main_process()", self.source)
        self.assertIn("self.cfg.TEST.EVAL_PERIOD > 0", self.source)
        returns = [
            node
            for node in ast.walk(self.method)
            if isinstance(node, ast.Return)
        ]
        self.assertTrue(
            any(
                isinstance(node.value, ast.Name)
                and node.value.id == "hook_list"
                for node in returns
            )
        )

    def test_best_hook_monitors_bbox_ap_and_is_after_evaluation(self):
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
        self.assertEqual(ast.literal_eval(keywords["val_metric"]), "bbox/AP")
        self.assertEqual(ast.literal_eval(keywords["mode"]), "max")
        self.assertEqual(
            ast.literal_eval(keywords["file_prefix"]),
            "model_best",
        )
```

- [ ] **Step 2: Run the hook tests and verify they fail**

Run:

```bash
/Users/touchnow/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  -m unittest tests.test_training_eval_best_checkpoint.BestCheckpointHookTests -v
```

Expected: `ERROR` with `StopIteration`, because `Trainer.build_hooks()` does not yet exist.

- [ ] **Step 3: Implement the minimal hook override**

Add this method to `Trainer` in `train_net.py`, immediately before `build_evaluator`:

```python
    def build_hooks(self):
        hook_list = super().build_hooks()
        if comm.is_main_process() and self.cfg.TEST.EVAL_PERIOD > 0:
            hook_list.insert(
                -1,
                hooks.BestCheckpointer(
                    eval_period=self.cfg.TEST.EVAL_PERIOD,
                    checkpointer=self.checkpointer,
                    val_metric="bbox/AP",
                    mode="max",
                    file_prefix="model_best",
                ),
            )
        return hook_list
```

Detectron2 0.6's default main-process hook list ends with `EvalHook` followed by `PeriodicWriter`. Inserting at index `-1` places `BestCheckpointer` after evaluation and before writing, so it reads the metric produced in the same iteration.

- [ ] **Step 4: Run the hook tests and verify they pass**

Run:

```bash
/Users/touchnow/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  -m unittest tests.test_training_eval_best_checkpoint.BestCheckpointHookTests -v
```

Expected: two tests run and report `OK`.

- [ ] **Step 5: Run the complete new regression module**

Run:

```bash
/Users/touchnow/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  -m unittest tests.test_training_eval_best_checkpoint -v
```

Expected: three tests run and report `OK`.

### Task 3: Regression and Runtime Verification

**Files:**
- Verify: `tests/test_training_eval_best_checkpoint.py`
- Verify: `tests/test_review_sdkd_integration.py`
- Verify: `configs/Base-Distillation.yaml`
- Verify: `train_net.py`

- [ ] **Step 1: Compile the changed Python files**

Run:

```bash
/Users/touchnow/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  -m py_compile train_net.py tests/test_training_eval_best_checkpoint.py
```

Expected: exit code 0 with no output.

- [ ] **Step 2: Run all dependency-free new tests**

Run:

```bash
/Users/touchnow/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  -m unittest tests.test_training_eval_best_checkpoint -v
```

Expected: three tests run and report `OK`.

- [ ] **Step 3: Run the pre-existing regression module in the training environment**

Run with the Python environment used for training, which already provides PyYAML:

```bash
python3 -m unittest tests.test_review_sdkd_integration -v
```

Expected: all five existing tests report `OK`. The bundled local Python lacks PyYAML, so this check must use the user's training environment rather than being reported as locally verified.

- [ ] **Step 4: Smoke-test configuration and evaluation startup in the training environment**

Run:

```bash
python3 train_net.py \
  --config-file configs/SDKD/SDKD-R18-R101.yaml \
  --num-gpus 1 \
  --eval-only \
  MODEL.DEVICE cpu
```

Expected: configuration parsing succeeds and evaluation starts. Dataset or weight availability may stop the command later; any such stop is environment evidence, not proof that training-time best-checkpoint behavior ran.

- [ ] **Step 5: Verify the feature during a real training run**

Start training with the normal command and a unique distributed URL/output directory. After iteration 10,000, verify that the log contains a COCO `bbox/AP` value and a `Saved first model` or `Saved best model` message, and verify that `OUTPUT_DIR/model_best.pth` exists. Repeat after a later evaluation to confirm the file changes only when `bbox/AP` improves.

This is the boundary for claiming end-to-end runtime success: local AST/config tests prove the implementation wiring, while only the user's Detectron2/CUDA/COCO environment can prove evaluation cost, distributed execution, and checkpoint creation on real data.
