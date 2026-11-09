# NKD Object Detection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add independent Normalized Knowledge Distillation (NKD) training for Faster R-CNN ROI classification logits.

**Architecture:** A tensor-only `model/nkd.py` implements the provided NKD formula. `RCNNKD` obtains student and teacher logits from the same sampled proposals and adds only `loss_nkd`; existing detector and distillation losses remain unchanged. Three standalone configurations expose R18-R101, R50-R101, and MV2-R50 training.

**Tech Stack:** Python, PyTorch, Detectron2, YAML, `unittest`

**Repository note:** This workspace is not a Git repository. Use the specified test commands as checkpoints; Git commit steps are intentionally omitted.

---

### Task 1: Specify the NKD loss with failing tests

**Files:**
- Create: `tests/test_nkd_integration.py`
- Create later: `model/nkd.py`

- [ ] **Step 1: Add the loss test module and loader**

```python
import importlib.util
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
NKD_PATH = ROOT / "model" / "nkd.py"


def _load_nkd_module():
    if not NKD_PATH.exists():
        return None
    spec = importlib.util.spec_from_file_location("nkd_under_test", NKD_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
```

- [ ] **Step 2: Add and run a failing public-API test**

```python
class NKDLossTests(unittest.TestCase):
    def test_exposes_nkd_loss(self):
        module = _load_nkd_module()
        self.assertIsNotNone(module)
        self.assertTrue(callable(module.nkd_loss))
```

Run:

```bash
python -m unittest tests.test_nkd_integration.NKDLossTests.test_exposes_nkd_loss
```

Expected: assertion failure because `model/nkd.py` does not exist.

- [ ] **Step 3: Add the smallest API implementation**

```python
def nkd_loss(student_logits, teacher_logits, targets, temperature, gamma):
    return student_logits.sum() * 0.0
```

- [ ] **Step 4: Verify the public-API test passes**

Run:

```bash
python -m unittest tests.test_nkd_integration.NKDLossTests.test_exposes_nkd_loss
```

Expected: 1 test passes.

- [ ] **Step 5: Test exact agreement with the provided formula**

```python
    def test_loss_matches_reference_formula(self):
        nkd_loss = _load_nkd_module().nkd_loss
        student = torch.tensor(
            [[1.0, 3.0, -1.0, 0.5], [2.0, -0.5, 1.5, 0.0]],
            requires_grad=True,
        )
        teacher = torch.tensor(
            [[2.5, 1.0, 0.0, -1.0], [0.5, 3.0, 1.0, -0.5]])
        targets = torch.tensor([1, 2])
        temperature = 1.0
        gamma = 1.5

        labels = targets.reshape(-1, 1)
        student_log_probability = F.log_softmax(student, dim=1)
        teacher_probability = F.softmax(teacher, dim=1)
        target_loss = -(
            teacher_probability.gather(1, labels)
            * student_log_probability.gather(1, labels)
        ).mean()

        mask = torch.ones_like(student, dtype=torch.bool)
        mask.scatter_(1, labels, False)
        student_non_target = student[mask].reshape(student.shape[0], -1)
        teacher_non_target = teacher[mask].reshape(teacher.shape[0], -1)
        non_target_loss = -(
            F.softmax(teacher_non_target / temperature, dim=1)
            * F.log_softmax(student_non_target / temperature, dim=1)
        ).sum(dim=1).mean()
        expected = target_loss + gamma * temperature ** 2 * non_target_loss

        actual = nkd_loss(
            student, teacher, targets, temperature, gamma)
        self.assertTrue(
            torch.allclose(actual, expected, atol=1e-7, rtol=1e-6))
```

- [ ] **Step 6: Run the formula test and verify the stub fails**

Run:

```bash
python -m unittest tests.test_nkd_integration.NKDLossTests.test_loss_matches_reference_formula
```

Expected: assertion failure because the stub returns zero.

- [ ] **Step 7: Add label compatibility and gradient-isolation tests**

```python
    def test_one_hot_and_class_index_targets_match(self):
        nkd_loss = _load_nkd_module().nkd_loss
        student = torch.tensor([[1.0, 2.0, 0.0], [0.5, -1.0, 3.0]])
        teacher = torch.tensor([[2.0, 0.0, 1.0], [1.0, 2.5, 0.0]])
        indices = torch.tensor([1, 2])
        one_hot = F.one_hot(indices, num_classes=3).float()

        by_index = nkd_loss(student, teacher, indices, 1.0, 1.5)
        by_one_hot = nkd_loss(student, teacher, one_hot, 1.0, 1.5)
        self.assertTrue(torch.equal(by_index, by_one_hot))

    def test_only_student_receives_finite_gradients(self):
        nkd_loss = _load_nkd_module().nkd_loss
        student = torch.randn(3, 5, requires_grad=True)
        teacher = torch.randn(3, 5, requires_grad=True)
        targets = torch.tensor([0, 2, 4])

        loss = nkd_loss(student, teacher, targets, 1.0, 1.5)
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(student.grad).all())
        self.assertIsNone(teacher.grad)
```

---

### Task 2: Implement the standalone NKD loss

**Files:**
- Create: `model/nkd.py`
- Test: `tests/test_nkd_integration.py`

- [ ] **Step 1: Replace the stub with the minimal tensor-only loss**

```python
import torch
import torch.nn.functional as F


def nkd_loss(student_logits, teacher_logits, targets, temperature, gamma):
    teacher_logits = teacher_logits.detach()
    if targets.ndim > 1:
        labels = targets.argmax(dim=1, keepdim=True)
    else:
        labels = targets.reshape(-1, 1)

    student_log_probability = F.log_softmax(student_logits, dim=1)
    teacher_probability = F.softmax(teacher_logits, dim=1)
    target_loss = -(
        teacher_probability.gather(1, labels)
        * student_log_probability.gather(1, labels)
    ).mean()

    non_target_mask = torch.ones_like(student_logits, dtype=torch.bool)
    non_target_mask.scatter_(1, labels, False)
    batch_size = student_logits.shape[0]
    student_non_target = student_logits[non_target_mask].reshape(batch_size, -1)
    teacher_non_target = teacher_logits[non_target_mask].reshape(batch_size, -1)

    student_non_target_log_probability = F.log_softmax(
        student_non_target / temperature, dim=1)
    teacher_non_target_probability = F.softmax(
        teacher_non_target / temperature, dim=1)
    non_target_loss = -(
        teacher_non_target_probability * student_non_target_log_probability
    ).sum(dim=1).mean()

    return target_loss + gamma * temperature ** 2 * non_target_loss
```

- [ ] **Step 2: Run the loss tests**

Run:

```bash
python -m unittest tests.test_nkd_integration.NKDLossTests
```

Expected: 4 tests pass.

---

### Task 3: Specify and add isolated NKD integration

**Files:**
- Modify: `tests/test_nkd_integration.py`
- Modify: `model/config.py`
- Modify: `model/rcnn.py`

- [ ] **Step 1: Add AST helpers for defaults and the NKD route**

```python
import ast


CONFIG_PATH = ROOT / "model" / "config.py"
RCNN_PATH = ROOT / "model" / "rcnn.py"


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


def _nkd_branch():
    tree = ast.parse(RCNN_PATH.read_text())
    trainer = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "RCNNKD")
    forward = next(
        node for node in trainer.body
        if isinstance(node, ast.FunctionDef) and node.name == "forward")
    for node in ast.walk(forward):
        if not isinstance(node, ast.If):
            continue
        strings = {
            item.value for item in ast.walk(node.test)
            if isinstance(item, ast.Constant) and isinstance(item.value, str)
        }
        if "NKD" in strings:
            return ast.Module(body=node.body, type_ignores=[])
    raise AssertionError("missing RCNNKD route: NKD")
```

- [ ] **Step 2: Add failing default and route tests**

```python
class NKDConfigDefaultTests(unittest.TestCase):
    def test_defaults_match_confirmed_settings(self):
        assignments = _assigned_literals(CONFIG_PATH)
        self.assertEqual(assignments["cfg.KD.NKD.T"], 1.0)
        self.assertEqual(assignments["cfg.KD.NKD.GAMMA"], 1.5)
        self.assertEqual(assignments["cfg.KD.NKD.LOSS_WEIGHT"], 1.0)


class NKDRouteTests(unittest.TestCase):
    def setUp(self):
        self.branch = _nkd_branch()

    def test_route_imports_and_calls_standalone_loss(self):
        module_tree = ast.parse(RCNN_PATH.read_text())
        imports = {
            (node.module, alias.name)
            for node in module_tree.body
            if isinstance(node, ast.ImportFrom)
            for alias in node.names
        }
        self.assertIn(("nkd", "nkd_loss"), imports)
        calls = {
            ast.unparse(node.func) for node in ast.walk(self.branch)
            if isinstance(node, ast.Call)
        }
        self.assertIn("nkd_loss", calls)

    def test_route_uses_the_same_sampled_proposals(self):
        calls = [
            node for node in ast.walk(self.branch)
            if isinstance(node, ast.Call)
            and ast.unparse(node.func) == "self.forward_pure_roi_head"
        ]
        self.assertEqual(len(calls), 2)
        self.assertEqual(
            [ast.unparse(call.args[2]) for call in calls],
            ["sampled_proposals", "sampled_proposals"],
        )

    def test_route_adds_only_weighted_loss_nkd(self):
        source = ast.unparse(self.branch)
        self.assertIn("self.kd_args.NKD.T", source)
        self.assertIn("self.kd_args.NKD.GAMMA", source)
        self.assertIn("self.kd_args.NKD.LOSS_WEIGHT", source)
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
        self.assertEqual(loss_keys, {"loss_nkd"})
```

- [ ] **Step 3: Run integration tests and verify missing defaults/route failures**

Run:

```bash
python -m unittest tests.test_nkd_integration.NKDConfigDefaultTests tests.test_nkd_integration.NKDRouteTests
```

Expected: failures for missing `cfg.KD.NKD` and missing `RCNNKD route: NKD`.

- [ ] **Step 4: Add NKD defaults after the existing SDD block in `model/config.py`**

```python
    # NKD
    cfg.KD.NKD = CN()
    cfg.KD.NKD.T = 1.0
    cfg.KD.NKD.GAMMA = 1.5
    cfg.KD.NKD.LOSS_WEIGHT = 1.0
```

- [ ] **Step 5: Import NKD in `model/rcnn.py`**

```python
from .nkd import nkd_loss
```

- [ ] **Step 6: Add an independent NKD branch beside the other logit methods**

```python
        elif self.kd_args.TYPE == "NKD":
            _, stu_predictions = self.forward_pure_roi_head(
                self.roi_heads, features, sampled_proposals)
            with torch.no_grad():
                teacher_images = self.teacher_preprocess_image(batched_inputs)
                t_features = self.teacher.backbone(teacher_images.tensor)
                _, tea_predictions = self.forward_pure_roi_head(
                    self.teacher.roi_heads, t_features, sampled_proposals)
            stu_logits, _ = stu_predictions
            tea_logits, _ = tea_predictions
            gt_classes = torch.cat(
                tuple(x.gt_classes for x in sampled_proposals), 0)
            losses["loss_nkd"] = nkd_loss(
                stu_logits,
                tea_logits,
                gt_classes,
                self.kd_args.NKD.T,
                self.kd_args.NKD.GAMMA,
            ) * self.kd_args.NKD.LOSS_WEIGHT
```

- [ ] **Step 7: Run the integration tests**

Run:

```bash
python -m unittest tests.test_nkd_integration.NKDConfigDefaultTests tests.test_nkd_integration.NKDRouteTests
```

Expected: 4 tests pass.

---

### Task 4: Add three runnable NKD configurations

**Files:**
- Create: `configs/NKD/NKD-R18-R101.yaml`
- Create: `configs/NKD/NKD-R50-R101.yaml`
- Create: `configs/NKD/NKD-MV2-R50.yaml`
- Modify: `tests/test_nkd_integration.py`

- [ ] **Step 1: Add configuration expectations**

```python
import yaml


NKD_CONFIG_DIR = ROOT / "configs" / "NKD"
ARCHITECTURES = {
    "R18-R101": {
        "weights": "pretrained/r18-r101.pth",
        "backbone": "build_resnet_fpn_backbone_kd",
        "features": ["res2", "res3", "res4", "res5"],
        "student_depth": 18,
        "teacher_depth": 101,
    },
    "R50-R101": {
        "weights": "pretrained/r50-r101.pth",
        "backbone": "build_resnet_fpn_backbone_kd",
        "features": ["res2", "res3", "res4", "res5"],
        "student_depth": 50,
        "teacher_depth": 101,
    },
    "MV2-R50": {
        "weights": "pretrained/mv2-r50.pth",
        "backbone": "build_mobilenetv2_fpn_backbone",
        "features": ["m2", "m3", "m4", "m5"],
        "teacher_depth": 50,
    },
}


class NKDArchitectureConfigTests(unittest.TestCase):
    def test_all_detection_configs_match_confirmed_settings(self):
        output_dirs = []
        for architecture, spec in ARCHITECTURES.items():
            with self.subTest(architecture=architecture):
                path = NKD_CONFIG_DIR / "NKD-{}.yaml".format(architecture)
                self.assertTrue(
                    path.exists(), "missing configuration: {}".format(path))
                config = yaml.safe_load(path.read_text())
                self.assertEqual(config["_BASE_"], "../Base-Distillation.yaml")
                self.assertEqual(
                    config["OUTPUT_DIR"], "output/NKD-{}".format(architecture))
                output_dirs.append(config["OUTPUT_DIR"])
                self.assertEqual(config["MODEL"]["WEIGHTS"], spec["weights"])
                self.assertEqual(
                    config["MODEL"]["BACKBONE"]["NAME"], spec["backbone"])
                self.assertEqual(
                    config["MODEL"]["FPN"]["IN_FEATURES"], spec["features"])
                self.assertNotIn("MASK_ON", config["MODEL"])
                self.assertEqual(
                    config["TEACHER"]["MODEL"]["RESNETS"]["DEPTH"],
                    spec["teacher_depth"],
                )
                self.assertEqual(config["KD"]["TYPE"], "NKD")
                self.assertEqual(config["KD"]["NKD"], {
                    "T": 1.0,
                    "GAMMA": 1.5,
                    "LOSS_WEIGHT": 1.0,
                })
                self.assertEqual(config["SOLVER"]["IMS_PER_BATCH"], 8)
                self.assertEqual(config["SOLVER"]["BASE_LR"], 0.01)
                self.assertEqual(config["SOLVER"]["MAX_ITER"], 180000)
                self.assertEqual(
                    config["SOLVER"]["CHECKPOINT_PERIOD"], 180001)
                self.assertEqual(
                    config["SOLVER"]["STEPS"], [120000, 160000])

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
                    self.assertEqual(
                        config["MODEL"]["BACKBONE"]["FREEZE_AT"], 0)
                    self.assertEqual(
                        config["MODEL"]["MOBILENETV2"]["OUT_FEATURES"],
                        spec["features"],
                    )

        self.assertEqual(len(output_dirs), len(set(output_dirs)))
```

- [ ] **Step 2: Run the configuration test and verify it fails**

Run:

```bash
python -m unittest tests.test_nkd_integration.NKDArchitectureConfigTests
```

Expected: three subtest failures naming the missing YAML files.

- [ ] **Step 3: Create `configs/NKD/NKD-R18-R101.yaml`**

```yaml
_BASE_: "../Base-Distillation.yaml"
OUTPUT_DIR: output/NKD-R18-R101
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
  TYPE: "NKD"
  NKD:
    T: 1.0
    GAMMA: 1.5
    LOSS_WEIGHT: 1.0

SOLVER:
  IMS_PER_BATCH: 8
  BASE_LR: 0.01
  MAX_ITER: 180000
  CHECKPOINT_PERIOD: 180001
  STEPS:
    - 120000
    - 160000
```

- [ ] **Step 4: Create `configs/NKD/NKD-R50-R101.yaml`**

```yaml
_BASE_: "../Base-Distillation.yaml"
OUTPUT_DIR: output/NKD-R50-R101
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
  TYPE: "NKD"
  NKD:
    T: 1.0
    GAMMA: 1.5
    LOSS_WEIGHT: 1.0

SOLVER:
  IMS_PER_BATCH: 8
  BASE_LR: 0.01
  MAX_ITER: 180000
  CHECKPOINT_PERIOD: 180001
  STEPS:
    - 120000
    - 160000
```

- [ ] **Step 5: Create `configs/NKD/NKD-MV2-R50.yaml`**

```yaml
_BASE_: "../Base-Distillation.yaml"
OUTPUT_DIR: output/NKD-MV2-R50
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
  TYPE: "NKD"
  NKD:
    T: 1.0
    GAMMA: 1.5
    LOSS_WEIGHT: 1.0

SOLVER:
  IMS_PER_BATCH: 8
  BASE_LR: 0.01
  MAX_ITER: 180000
  CHECKPOINT_PERIOD: 180001
  STEPS:
    - 120000
    - 160000
```

- [ ] **Step 6: Run the complete NKD test module**

Run:

```bash
python -m unittest tests.test_nkd_integration
```

Expected: all 9 NKD tests pass.

---

### Task 5: Regression verification

**Files:**
- Verify: `model/nkd.py`
- Verify: `model/config.py`
- Verify: `model/rcnn.py`
- Verify: `tests/test_nkd_integration.py`
- Verify: `configs/NKD/*.yaml`

- [ ] **Step 1: Compile every changed Python file**

Run:

```bash
python -m py_compile model/nkd.py model/config.py model/rcnn.py tests/test_nkd_integration.py
```

Expected: exit code 0 with no output.

- [ ] **Step 2: Run the full test suite**

Run:

```bash
python -m unittest discover -s tests
```

Expected: all new NKD tests pass. The only allowed failures are the four
pre-existing expectation mismatches:

- Three `ReviewSDKD-Mask` `INARANK_WEIGHT` assertions.
- One base `EVAL_PERIOD` assertion (`10000` expected versus `3000` configured).

- [ ] **Step 3: Confirm the change boundary**

Inspect the modified sections and verify:

- Existing distillation branches are byte-for-byte unchanged.
- Existing configuration values are unchanged.
- NKD introduces only `model/nkd.py`, one config block, one import, one route,
  three YAML files, tests, and NKD documentation.
- No end-to-end training claim is made unless Detectron2 and the required GPU
  environment are available and an actual training smoke test succeeds.
