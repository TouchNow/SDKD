# TeKAP Logit Distillation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an independent logit-level `KD.TYPE: "TeKAP"` method for R18 to R101, R50 to R101, and MobileNetV2 to R50 object-detector distillation.

**Architecture:** Put the stateless TeKAP KL computation in a PyTorch-only module, then add one surgical route to the existing `RCNNKD` training method using paired ROI logits from the same sampled proposals. Expose the confirmed `T=1.0` and paper/source perturbation defaults through one config node and three architecture-specific YAML files while leaving all existing KD and inference paths unchanged.

**Tech Stack:** Python 3, PyTorch, Detectron2/YACS, YAML, `unittest`, Python AST

---

The workspace is not a Git repository, so the commit steps normally required after each task are unavailable. Preserve task-sized edits, run each listed verification command, and report the final changed-file set instead.

## File Map

- Create `model/tekap.py`: validate inputs and calculate the original-teacher plus synthetic-teacher KL loss.
- Create `tests/test_tekap_loss.py`: numerical, gradient, and argument-validation coverage independent of Detectron2.
- Modify `model/config.py`: register the `KD.TEKAP` defaults.
- Modify `model/rcnn.py`: connect paired ROI logits to the standalone loss only when `KD.TYPE == "TeKAP"`.
- Create `configs/TeKAP/TeKAP-R18-R101.yaml`: R18 student and R101 teacher training configuration.
- Create `configs/TeKAP/TeKAP-R50-R101.yaml`: R50 student and R101 teacher training configuration.
- Create `configs/TeKAP/TeKAP-MV2-R50.yaml`: MobileNetV2 student and R50 teacher training configuration.
- Create `tests/test_tekap_integration.py`: AST routing/default checks and YAML configuration checks.

### Task 1: Implement the standalone TeKAP loss with TDD

**Files:**
- Create: `tests/test_tekap_loss.py`
- Create: `model/tekap.py`

- [ ] **Step 1: Write the failing numerical and gradient tests**

Create `tests/test_tekap_loss.py` with the following initial content. Loading by file path avoids importing `model/__init__.py`, which imports Detectron2 and would make this PyTorch-only unit test environment-dependent.

```python
import importlib.util
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
TEKAP_PATH = ROOT / "model" / "tekap.py"


def _load_tekap_module():
    spec = importlib.util.spec_from_file_location("tekap_under_test", TEKAP_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TeKAPLossTests(unittest.TestCase):
    def test_matches_reference_computation_with_fixed_noise(self):
        tekap_loss = _load_tekap_module().tekap_loss
        student = torch.tensor(
            [[1.2, -0.3, 0.7], [-0.4, 0.9, 1.5]],
            dtype=torch.float32,
            requires_grad=True,
        )
        teacher = torch.tensor(
            [[0.8, 0.1, -0.5], [1.0, -0.2, 0.4]],
            dtype=torch.float32,
        )

        torch.manual_seed(17)
        actual = tekap_loss(
            student,
            teacher,
            temperature=1.0,
            num_augmented=3,
            noise_weight=0.1,
            augmented_weight=0.8,
        )

        torch.manual_seed(17)
        student_log_prob = F.log_softmax(student, dim=1)
        teacher_logits = teacher.detach()
        expected = F.kl_div(
            student_log_prob,
            F.softmax(teacher_logits, dim=1),
            reduction="batchmean",
        )
        for _ in range(3):
            synthetic_logits = (
                0.9 * teacher_logits
                + 0.1 * torch.randn_like(teacher_logits)
            )
            expected = expected + 0.8 * F.kl_div(
                student_log_prob,
                F.softmax(synthetic_logits, dim=1),
                reduction="batchmean",
            )

        torch.testing.assert_close(actual, expected)

    def test_zero_noise_has_expected_weight_relation(self):
        tekap_loss = _load_tekap_module().tekap_loss
        student = torch.tensor([[0.1, 0.5, -0.2]], requires_grad=True)
        teacher = torch.tensor([[0.4, -0.1, 0.2]])
        base = F.kl_div(
            F.log_softmax(student, dim=1),
            F.softmax(teacher, dim=1),
            reduction="batchmean",
        )

        actual = tekap_loss(
            student,
            teacher,
            temperature=1.0,
            num_augmented=3,
            noise_weight=0.0,
            augmented_weight=0.8,
        )

        torch.testing.assert_close(actual, base * (1 + 3 * 0.8))

    def test_gradient_flows_only_to_student(self):
        tekap_loss = _load_tekap_module().tekap_loss
        student = torch.randn(4, 5, requires_grad=True)
        teacher = torch.randn(4, 5, requires_grad=True)

        loss = tekap_loss(
            student,
            teacher,
            temperature=1.0,
            num_augmented=3,
            noise_weight=0.1,
            augmented_weight=0.8,
        )
        loss.backward()

        self.assertIsNotNone(student.grad)
        self.assertTrue(torch.isfinite(student.grad).all())
        self.assertIsNone(teacher.grad)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_cuda_inputs_produce_cuda_loss(self):
        tekap_loss = _load_tekap_module().tekap_loss
        student = torch.randn(2, 3, device="cuda", requires_grad=True)
        teacher = torch.randn(2, 3, device="cuda")

        loss = tekap_loss(
            student,
            teacher,
            temperature=1.0,
            num_augmented=3,
            noise_weight=0.1,
            augmented_weight=0.8,
        )

        self.assertEqual(loss.device.type, "cuda")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
python -m unittest tests.test_tekap_loss -v
```

Expected: the three CPU tests error with `FileNotFoundError` because `model/tekap.py` does not exist; the CUDA test has the same error when CUDA is available, or is skipped otherwise. This is the expected missing-feature failure, not an unrelated import failure.

- [ ] **Step 3: Add failing parameter-validation tests**

Add these methods to `TeKAPLossTests` before the `if __name__` block:

```python
    def test_rejects_invalid_arguments(self):
        tekap_loss = _load_tekap_module().tekap_loss
        student = torch.randn(2, 3)
        teacher = torch.randn(2, 3)
        invalid_cases = (
            {"temperature": 0.0},
            {"temperature": -1.0},
            {"num_augmented": 0},
            {"num_augmented": 1.5},
            {"num_augmented": True},
            {"noise_weight": -0.1},
            {"noise_weight": 1.1},
            {"augmented_weight": -0.1},
        )

        defaults = {
            "temperature": 1.0,
            "num_augmented": 3,
            "noise_weight": 0.1,
            "augmented_weight": 0.8,
        }
        for override in invalid_cases:
            with self.subTest(override=override):
                arguments = dict(defaults)
                arguments.update(override)
                with self.assertRaises(ValueError):
                    tekap_loss(student, teacher, **arguments)

    def test_rejects_mismatched_logit_shapes(self):
        tekap_loss = _load_tekap_module().tekap_loss
        with self.assertRaises(ValueError):
            tekap_loss(
                torch.randn(2, 3),
                torch.randn(2, 4),
                temperature=1.0,
                num_augmented=3,
                noise_weight=0.1,
                augmented_weight=0.8,
            )
```

- [ ] **Step 4: Run the expanded tests and verify they remain RED for the missing module**

Run:

```bash
python -m unittest tests.test_tekap_loss -v
```

Expected: five CPU-test errors caused by missing `model/tekap.py`; the CUDA test also errors when CUDA is available, or is skipped otherwise.

- [ ] **Step 5: Implement the minimal standalone loss**

Create `model/tekap.py`:

```python
import torch
import torch.nn.functional as F


def tekap_loss(
        student_logits,
        teacher_logits,
        temperature,
        num_augmented,
        noise_weight,
        augmented_weight):
    if student_logits.shape != teacher_logits.shape:
        raise ValueError("student and teacher logits must have the same shape")
    if temperature <= 0:
        raise ValueError("temperature must be greater than zero")
    if (
            isinstance(num_augmented, bool)
            or not isinstance(num_augmented, int)
            or num_augmented <= 0):
        raise ValueError("num_augmented must be a positive integer")
    if not 0 <= noise_weight <= 1:
        raise ValueError("noise_weight must be between zero and one")
    if augmented_weight < 0:
        raise ValueError("augmented_weight must be non-negative")

    teacher_logits = teacher_logits.detach()
    student_log_prob = F.log_softmax(student_logits / temperature, dim=1)
    teacher_prob = F.softmax(teacher_logits / temperature, dim=1)
    loss = F.kl_div(
        student_log_prob,
        teacher_prob,
        reduction="batchmean",
    ) * (temperature ** 2)

    teacher_weight = 1 - noise_weight
    for _ in range(num_augmented):
        synthetic_logits = (
            teacher_weight * teacher_logits
            + noise_weight * torch.randn_like(teacher_logits)
        )
        synthetic_prob = F.softmax(synthetic_logits / temperature, dim=1)
        loss = loss + augmented_weight * F.kl_div(
            student_log_prob,
            synthetic_prob,
            reduction="batchmean",
        ) * (temperature ** 2)

    return loss
```

- [ ] **Step 6: Run the focused tests and verify GREEN**

Run:

```bash
python -m unittest tests.test_tekap_loss -v
```

Expected: `Ran 6 tests` and `OK` (with one skip when CUDA is unavailable).

### Task 2: Register TeKAP configuration defaults with TDD

**Files:**
- Create: `tests/test_tekap_integration.py`
- Modify: `model/config.py:57-65`

- [ ] **Step 1: Write a failing AST test for exact defaults**

Create `tests/test_tekap_integration.py`:

```python
import ast
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "model" / "config.py"
RCNN_PATH = ROOT / "model" / "rcnn.py"
TEKAP_CONFIG_DIR = ROOT / "configs" / "TeKAP"


def _assigned_literals(path):
    tree = ast.parse(path.read_text())
    values = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        try:
            values[ast.unparse(node.targets[0])] = ast.literal_eval(node.value)
        except (ValueError, TypeError):
            continue
    return values


class TeKAPDefaultConfigTests(unittest.TestCase):
    def test_tekap_defaults_match_confirmed_design(self):
        assignments = _assigned_literals(CONFIG_PATH)
        expected = {
            "cfg.KD.TEKAP.T": 1.0,
            "cfg.KD.TEKAP.NUM_AUGMENTED": 3,
            "cfg.KD.TEKAP.NOISE_WEIGHT": 0.1,
            "cfg.KD.TEKAP.AUGMENTED_WEIGHT": 0.8,
            "cfg.KD.TEKAP.LOSS_WEIGHT": 1.0,
        }
        for target, value in expected.items():
            with self.subTest(target=target):
                self.assertIn(target, assignments)
                self.assertEqual(assignments[target], value)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the default-config test and verify RED**

Run:

```bash
python -m unittest tests.test_tekap_integration.TeKAPDefaultConfigTests -v
```

Expected: failure because `cfg.KD.TEKAP.*` assignments do not exist.

- [ ] **Step 3: Add the minimal config node**

In `model/config.py`, insert after the existing `KD.SDKD` defaults and before `KD.REVIEWKD`:

```python
    # TeKAP
    cfg.KD.TEKAP = CN()
    cfg.KD.TEKAP.T = 1.0
    cfg.KD.TEKAP.NUM_AUGMENTED = 3
    cfg.KD.TEKAP.NOISE_WEIGHT = 0.1
    cfg.KD.TEKAP.AUGMENTED_WEIGHT = 0.8
    cfg.KD.TEKAP.LOSS_WEIGHT = 1.0

```

- [ ] **Step 4: Run the default-config test and verify GREEN**

Run:

```bash
python -m unittest tests.test_tekap_integration.TeKAPDefaultConfigTests -v
```

Expected: `Ran 1 test` and `OK`.

### Task 3: Route paired ROI logits through TeKAP with TDD

**Files:**
- Modify: `tests/test_tekap_integration.py`
- Modify: `model/rcnn.py:18-29,341-359`

- [ ] **Step 1: Add AST helpers and failing route tests**

Add these helpers above the test classes in `tests/test_tekap_integration.py`:

```python
def _method_node(class_name, method_name):
    tree = ast.parse(RCNN_PATH.read_text())
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    return next(
        node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef) and node.name == method_name
    )


def _branch_for_type(type_name):
    forward_node = _method_node("RCNNKD", "forward")
    for node in ast.walk(forward_node):
        if not isinstance(node, ast.If) or not isinstance(node.test, ast.Compare):
            continue
        constants = {
            item.value
            for item in ast.walk(node.test)
            if isinstance(item, ast.Constant) and isinstance(item.value, str)
        }
        if type_name in constants:
            return ast.Module(body=node.body, type_ignores=[])
    raise AssertionError("missing RCNNKD route: {}".format(type_name))
```

Add this class after `TeKAPDefaultConfigTests`:

```python
class TeKAPRouteTests(unittest.TestCase):
    def setUp(self):
        self.branch = _branch_for_type("TeKAP")

    def test_route_imports_and_calls_standalone_loss(self):
        module_tree = ast.parse(RCNN_PATH.read_text())
        imports = {
            (node.module, alias.name)
            for node in module_tree.body
            if isinstance(node, ast.ImportFrom)
            for alias in node.names
        }
        self.assertIn(("tekap", "tekap_loss"), imports)

        calls = {
            ast.unparse(node.func)
            for node in ast.walk(self.branch)
            if isinstance(node, ast.Call)
        }
        self.assertIn("tekap_loss", calls)

    def test_route_uses_same_sampled_proposals_for_student_and_teacher(self):
        roi_calls = [
            node
            for node in ast.walk(self.branch)
            if isinstance(node, ast.Call)
            and ast.unparse(node.func) == "self.forward_pure_roi_head"
        ]
        self.assertEqual(len(roi_calls), 2)
        self.assertEqual(
            [ast.unparse(call.args[2]) for call in roi_calls],
            ["sampled_proposals", "sampled_proposals"],
        )
        self.assertEqual(
            {ast.unparse(call.args[0]) for call in roi_calls},
            {"self.roi_heads", "self.teacher.roi_heads"},
        )

    def test_route_adds_only_weighted_loss_tekap(self):
        assignments = [
            node
            for node in ast.walk(self.branch)
            if isinstance(node, ast.Assign)
        ]
        loss_assignment = next(
            node
            for node in assignments
            if any(ast.unparse(target) == "losses['loss_tekap']" for target in node.targets)
        )
        value_source = ast.unparse(loss_assignment.value)
        self.assertIn("tekap_loss", value_source)
        self.assertIn("self.kd_args.TEKAP.LOSS_WEIGHT", value_source)

        loss_keys = {
            node.slice.value
            for node in ast.walk(self.branch)
            if isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Name)
            and node.value.id in {"losses", "detector_losses", "proposal_losses"}
            and isinstance(node.slice, ast.Constant)
            and isinstance(node.slice.value, str)
            and node.slice.value.startswith("loss_")
        }
        self.assertEqual(loss_keys, {"loss_tekap"})
```

- [ ] **Step 2: Run the route tests and verify RED**

Run:

```bash
python -m unittest tests.test_tekap_integration.TeKAPRouteTests -v
```

Expected: errors from `_branch_for_type` because `RCNNKD.forward` has no `TeKAP` route.

- [ ] **Step 3: Import the standalone loss**

In `model/rcnn.py`, add this import immediately after the existing `sdkd_loss` import:

```python
from .tekap import tekap_loss
```

- [ ] **Step 4: Add the minimal independent training route**

In `model/rcnn.py`, insert this branch after the existing `DKD` branch and before `SDKD`:

```python
        elif self.kd_args.TYPE == "TeKAP":
            teacher_images = self.teacher_preprocess_image(batched_inputs)
            t_features = self.teacher.backbone(teacher_images.tensor)
            s_emb, stu_predictions = self.forward_pure_roi_head(
                self.roi_heads, features, sampled_proposals)
            t_emb, tea_predictions = self.forward_pure_roi_head(
                self.teacher.roi_heads, t_features, sampled_proposals)
            stu_logits, _ = stu_predictions
            tea_logits, _ = tea_predictions
            losses['loss_tekap'] = tekap_loss(
                stu_logits,
                tea_logits,
                self.kd_args.TEKAP.T,
                self.kd_args.TEKAP.NUM_AUGMENTED,
                self.kd_args.TEKAP.NOISE_WEIGHT,
                self.kd_args.TEKAP.AUGMENTED_WEIGHT,
            ) * self.kd_args.TEKAP.LOSS_WEIGHT
```

Do not edit the surrounding `DKD`, `SDKD`, or other branches.

- [ ] **Step 5: Run all current TeKAP tests and verify GREEN**

Run:

```bash
python -m unittest tests.test_tekap_loss tests.test_tekap_integration -v
```

Expected: `Ran 10 tests` and `OK` (with one skip when CUDA is unavailable).

### Task 4: Add all three architecture configurations with TDD

**Files:**
- Modify: `tests/test_tekap_integration.py`
- Create: `configs/TeKAP/TeKAP-R18-R101.yaml`
- Create: `configs/TeKAP/TeKAP-R50-R101.yaml`
- Create: `configs/TeKAP/TeKAP-MV2-R50.yaml`

- [ ] **Step 1: Add failing configuration tests**

Add these constants beneath the path constants in `tests/test_tekap_integration.py`:

```python
TEKAP_SETTINGS = {
    "T": 1.0,
    "NUM_AUGMENTED": 3,
    "NOISE_WEIGHT": 0.1,
    "AUGMENTED_WEIGHT": 0.8,
    "LOSS_WEIGHT": 1.0,
}

ARCHITECTURES = {
    "R18-R101": {
        "weights": "pretrained/r18-r101.pth",
        "backbone": "build_resnet_fpn_backbone_kd",
        "student_depth": 18,
        "teacher_depth": 101,
        "features": ["res2", "res3", "res4", "res5"],
    },
    "R50-R101": {
        "weights": "pretrained/r50-r101.pth",
        "backbone": "build_resnet_fpn_backbone_kd",
        "student_depth": 50,
        "teacher_depth": 101,
        "features": ["res2", "res3", "res4", "res5"],
    },
    "MV2-R50": {
        "weights": "pretrained/mv2-r50.pth",
        "backbone": "build_mobilenetv2_fpn_backbone",
        "teacher_depth": 50,
        "features": ["m2", "m3", "m4", "m5"],
    },
}
```

Add this class after `TeKAPRouteTests`:

```python
class TeKAPArchitectureConfigTests(unittest.TestCase):
    def _load(self, architecture):
        path = TEKAP_CONFIG_DIR / "TeKAP-{}.yaml".format(architecture)
        self.assertTrue(path.exists(), "missing configuration: {}".format(path))
        return yaml.safe_load(path.read_text())

    def test_all_architecture_configs_match_confirmed_settings(self):
        output_dirs = []
        for architecture, spec in ARCHITECTURES.items():
            with self.subTest(architecture=architecture):
                config = self._load(architecture)
                self.assertEqual(config["_BASE_"], "../Base-Distillation.yaml")
                self.assertEqual(
                    config["OUTPUT_DIR"], "output/TeKAP-{}".format(architecture))
                output_dirs.append(config["OUTPUT_DIR"])
                self.assertEqual(config["MODEL"]["WEIGHTS"], spec["weights"])
                self.assertEqual(
                    config["MODEL"]["BACKBONE"]["NAME"], spec["backbone"])
                self.assertEqual(
                    config["MODEL"]["FPN"]["IN_FEATURES"], spec["features"])
                self.assertEqual(
                    config["TEACHER"]["MODEL"]["RESNETS"]["DEPTH"],
                    spec["teacher_depth"],
                )
                self.assertEqual(config["KD"]["TYPE"], "TeKAP")
                self.assertEqual(config["KD"]["TEKAP"], TEKAP_SETTINGS)
                self.assertEqual(config["SOLVER"]["IMS_PER_BATCH"], 8)
                self.assertEqual(config["SOLVER"]["BASE_LR"], 0.01)
                self.assertEqual(config["SOLVER"]["MAX_ITER"], 180000)
                self.assertEqual(config["SOLVER"]["STEPS"], [120000, 160000])

                if "student_depth" in spec:
                    self.assertEqual(
                        config["MODEL"]["RESNETS"]["DEPTH"],
                        spec["student_depth"],
                    )
                    self.assertEqual(
                        config["MODEL"]["RESNETS"]["OUT_FEATURES"],
                        spec["features"],
                    )
                else:
                    self.assertEqual(config["MODEL"]["BACKBONE"]["FREEZE_AT"], 0)
                    self.assertEqual(
                        config["MODEL"]["MOBILENETV2"]["OUT_FEATURES"],
                        spec["features"],
                    )

        self.assertEqual(len(output_dirs), len(set(output_dirs)))
```

- [ ] **Step 2: Run the config test and verify RED**

Run:

```bash
python -m unittest tests.test_tekap_integration.TeKAPArchitectureConfigTests -v
```

Expected: three subtest failures reporting the missing YAML paths.

- [ ] **Step 3: Create the R18 to R101 configuration**

Create `configs/TeKAP/TeKAP-R18-R101.yaml`:

```yaml
_BASE_: "../Base-Distillation.yaml"
OUTPUT_DIR: output/TeKAP-R18-R101
INPUT:
  FORMAT: 'RGB'
MODEL:
  PIXEL_STD: [57.375, 57.120, 58.395]
  BACKBONE:
    NAME: "build_resnet_fpn_backbone_kd"
  WEIGHTS: "pretrained/r18-r101.pth"
  RESNETS:
    OUT_FEATURES: ["res2", "res3", "res4", "res5"]
    DEPTH: 18
    RES2_OUT_CHANNELS: 64
  FPN:
    IN_FEATURES: ["res2", "res3", "res4", "res5"]
  PROPOSAL_GENERATOR:
    NAME: "RPN"
  ROI_HEADS:
    NAME: "StandardROIHeads"

TEACHER:
  MODEL:
    RESNETS:
      DEPTH: 101
KD:
  TYPE: "TeKAP"
  TEKAP:
    T: 1.0
    NUM_AUGMENTED: 3
    NOISE_WEIGHT: 0.1
    AUGMENTED_WEIGHT: 0.8
    LOSS_WEIGHT: 1.0

SOLVER:
  IMS_PER_BATCH: 8
  BASE_LR: 0.01
  MAX_ITER: 180000
  STEPS:
    - 120000
    - 160000
```

- [ ] **Step 4: Create the R50 to R101 configuration**

Create `configs/TeKAP/TeKAP-R50-R101.yaml`:

```yaml
_BASE_: "../Base-Distillation.yaml"
OUTPUT_DIR: output/TeKAP-R50-R101
MODEL:
  BACKBONE:
    NAME: "build_resnet_fpn_backbone_kd"
  WEIGHTS: "pretrained/r50-r101.pth"
  RESNETS:
    OUT_FEATURES: ["res2", "res3", "res4", "res5"]
    DEPTH: 50
  FPN:
    IN_FEATURES: ["res2", "res3", "res4", "res5"]
  PROPOSAL_GENERATOR:
    NAME: "RPN"
  ROI_HEADS:
    NAME: "StandardROIHeads"

TEACHER:
  MODEL:
    RESNETS:
      DEPTH: 101
KD:
  TYPE: "TeKAP"
  TEKAP:
    T: 1.0
    NUM_AUGMENTED: 3
    NOISE_WEIGHT: 0.1
    AUGMENTED_WEIGHT: 0.8
    LOSS_WEIGHT: 1.0

SOLVER:
  IMS_PER_BATCH: 8
  BASE_LR: 0.01
  MAX_ITER: 180000
  STEPS:
    - 120000
    - 160000
```

- [ ] **Step 5: Create the MobileNetV2 to R50 configuration**

Create `configs/TeKAP/TeKAP-MV2-R50.yaml`:

```yaml
_BASE_: "../Base-Distillation.yaml"
OUTPUT_DIR: output/TeKAP-MV2-R50
MODEL:
  BACKBONE:
    NAME: "build_mobilenetv2_fpn_backbone"
    FREEZE_AT: 0
  WEIGHTS: "pretrained/mv2-r50.pth"
  MOBILENETV2:
    OUT_FEATURES: ["m2", "m3", "m4", "m5"]
  FPN:
    IN_FEATURES: ["m2", "m3", "m4", "m5"]
  PROPOSAL_GENERATOR:
    NAME: "RPN"
  ROI_HEADS:
    NAME: "StandardROIHeads"

TEACHER:
  MODEL:
    RESNETS:
      DEPTH: 50
KD:
  TYPE: "TeKAP"
  TEKAP:
    T: 1.0
    NUM_AUGMENTED: 3
    NOISE_WEIGHT: 0.1
    AUGMENTED_WEIGHT: 0.8
    LOSS_WEIGHT: 1.0

SOLVER:
  IMS_PER_BATCH: 8
  BASE_LR: 0.01
  MAX_ITER: 180000
  STEPS:
    - 120000
    - 160000
```

- [ ] **Step 6: Run all TeKAP tests and verify GREEN**

Run:

```bash
python -m unittest tests.test_tekap_loss tests.test_tekap_integration -v
```

Expected: `Ran 11 tests` and `OK` (with one skip when CUDA is unavailable).

### Task 5: Run complete verification and inspect scope

**Files:**
- Verify: `model/tekap.py`
- Verify: `model/config.py`
- Verify: `model/rcnn.py`
- Verify: `tests/test_tekap_loss.py`
- Verify: `tests/test_tekap_integration.py`
- Verify: `configs/TeKAP/*.yaml`

- [ ] **Step 1: Run the complete unittest suite**

Run:

```bash
python -m unittest discover -s tests -v
```

Expected: all pre-existing and new tests pass with zero failures and zero errors.

- [ ] **Step 2: Compile every touched Python file**

Run:

```bash
python -m py_compile \
    model/tekap.py \
    model/config.py \
    model/rcnn.py \
    tests/test_tekap_loss.py \
    tests/test_tekap_integration.py
```

Expected: exit code 0 with no output.

- [ ] **Step 3: Parse all new YAML files independently**

Run:

```bash
python -c "from pathlib import Path; import yaml; paths=sorted(Path('configs/TeKAP').glob('*.yaml')); assert len(paths)==3; configs=[yaml.safe_load(p.read_text()) for p in paths]; assert all(c['KD']['TYPE']=='TeKAP' and c['KD']['TEKAP']['T']==1.0 for c in configs); print([str(p) for p in paths])"
```

Expected: a list containing exactly the three TeKAP YAML paths.

- [ ] **Step 4: Check whether a live Detectron2 import verification is available**

Run:

```bash
python -c "import torch, detectron2; from model.tekap import tekap_loss; print(torch.__version__, detectron2.__file__)"
```

Expected when the training environment is installed: exit code 0 and the PyTorch version plus Detectron2 path. If this fails because Detectron2 is absent or incompatible, record it as an environment limitation; it does not invalidate the PyTorch-only loss tests or AST/YAML integration tests.

- [ ] **Step 5: Verify config merging when Detectron2 is available**

Only if Step 4 succeeds, run:

```bash
python -c "from detectron2.config import get_cfg; from model import add_distillation_cfg; paths=['configs/TeKAP/TeKAP-R18-R101.yaml','configs/TeKAP/TeKAP-R50-R101.yaml','configs/TeKAP/TeKAP-MV2-R50.yaml']; [(lambda c,p: (add_distillation_cfg(c), c.merge_from_file(p), c.freeze()))(get_cfg(), p) for p in paths]; print('merged', len(paths))"
```

Expected: `merged 3`.

- [ ] **Step 6: Inspect the exact TeKAP surface**

Run:

```bash
rg -n "TeKAP|TEKAP|tekap" \
    model/tekap.py \
    model/config.py \
    model/rcnn.py \
    configs/TeKAP \
    tests/test_tekap_loss.py \
    tests/test_tekap_integration.py
```

Expected: matches are limited to the standalone loss, one config-default block, one import and one `RCNNKD` route, three YAML files, and focused tests.

- [ ] **Step 7: Record the verification boundary**

In the final handoff, report the exact test counts and commands from Steps 1-5. State explicitly that these checks validate implementation behavior and configuration wiring, while COCO AP improvement and full training stability require the user's dataset, pretrained checkpoints, GPUs, and training run.
