# LSKD Object Detection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an isolated Logit Standardization plus KD route for three object-detection teacher/student architectures without changing existing distillation methods.

**Architecture:** Put all new loss math in `model/lskd.py`, expose it through one new `KD.TYPE == "LSKD"` branch in `RCNNKD`, and add an isolated configuration node plus three new experiment YAMLs. Student and teacher use the same sampled ROIs; each ROI's class logits are standardized independently before the standard temperature-scaled KL loss.

**Tech Stack:** Python 3.9, PyTorch, Detectron2, YACS configuration, `unittest`, PyYAML.

---

## File structure

- Create `model/lskd.py`: standalone logit standardization and LSKD loss.
- Modify `model/config.py`: add only the `KD.LSKD` defaults.
- Modify `model/rcnn.py`: import `lskd_loss` and add only the new `LSKD` route.
- Create `configs/LSKD/LSKD-R18-R101.yaml`: ResNet-18 student and ResNet-101 teacher.
- Create `configs/LSKD/LSKD-R50-R101.yaml`: ResNet-50 student and ResNet-101 teacher.
- Create `configs/LSKD/LSKD-MV2-R50.yaml`: MobileNetV2 student and ResNet-50 teacher.
- Create `tests/test_lskd_integration.py`: numerical, route, configuration, and isolation coverage.

The current workspace is not a Git repository, so the usual per-task commit
steps cannot be executed. Preserve task boundaries and inspect the final file
list instead.

### Task 1: Specify and implement standalone LSKD math

**Files:**
- Create: `tests/test_lskd_integration.py`
- Create: `model/lskd.py`

- [ ] **Step 1: Write failing numerical tests**

Create `tests/test_lskd_integration.py` with these imports and tests:

```python
import ast
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "model" / "config.py"
RCNN_PATH = ROOT / "model" / "rcnn.py"
LSKD_CONFIG_DIR = ROOT / "configs" / "LSKD"


class LSKDLossTests(unittest.TestCase):
    def test_loss_matches_official_standardized_kd_formula(self):
        from model.lskd import lskd_loss

        student = torch.tensor(
            [[1.0, 2.0, 4.0], [3.0, -1.0, 0.5]], requires_grad=True)
        teacher = torch.tensor(
            [[2.0, 1.0, 5.0], [1.0, -2.0, 2.5]])
        temperature = 1.7

        student_z = (
            (student - student.mean(dim=-1, keepdim=True))
            / (student.std(dim=-1, keepdim=True) + 1e-7)
        )
        teacher_z = (
            (teacher - teacher.mean(dim=-1, keepdim=True))
            / (teacher.std(dim=-1, keepdim=True) + 1e-7)
        )
        expected = F.kl_div(
            F.log_softmax(student_z / temperature, dim=1),
            F.softmax(teacher_z / temperature, dim=1),
            reduction="none",
        ).sum(1).mean() * temperature ** 2

        actual = lskd_loss(student, teacher, temperature)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-7, rtol=1e-6))

    def test_loss_is_invariant_to_per_roi_shift_and_positive_scale(self):
        from model.lskd import lskd_loss

        student = torch.tensor(
            [[1.0, 3.0, -2.0, 0.5], [4.0, 0.0, 2.0, -1.0]])
        teacher = torch.tensor(
            [[2.0, -1.0, 3.5, 0.0], [1.0, 5.0, -2.0, 3.0]])
        base = lskd_loss(student, teacher, 1.0)
        transformed = lskd_loss(
            student * torch.tensor([[2.0], [0.5]]) + torch.tensor([[7.0], [-3.0]]),
            teacher * torch.tensor([[3.0], [1.5]]) + torch.tensor([[-4.0], [6.0]]),
            1.0,
        )
        self.assertTrue(torch.allclose(base, transformed, atol=1e-6, rtol=1e-6))

    def test_constant_logits_have_finite_loss_and_student_gradient(self):
        from model.lskd import lskd_loss

        student = torch.ones(2, 4, requires_grad=True)
        teacher = torch.tensor(
            [[1.0, 2.0, 3.0, 4.0], [4.0, 2.0, 1.0, 3.0]])
        loss = lskd_loss(student, teacher, 1.0)
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(student.grad).all())
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
/opt/anaconda3/bin/python3.9 -m unittest \
  tests.test_lskd_integration.LSKDLossTests -v
```

Expected: all three tests error with `ModuleNotFoundError: No module named 'model.lskd'`.

- [ ] **Step 3: Add the minimal standalone implementation**

Create `model/lskd.py`:

```python
import torch.nn.functional as F


def standardize_logits(logits):
    mean = logits.mean(dim=-1, keepdim=True)
    std = logits.std(dim=-1, keepdim=True)
    return (logits - mean) / (std + 1e-7)


def lskd_loss(student_logits, teacher_logits, temperature):
    student_logits = standardize_logits(student_logits)
    teacher_logits = standardize_logits(teacher_logits)
    log_student = F.log_softmax(student_logits / temperature, dim=1)
    teacher_prob = F.softmax(teacher_logits / temperature, dim=1)
    loss = F.kl_div(log_student, teacher_prob, reduction="none").sum(1).mean()
    return loss * temperature ** 2
```

- [ ] **Step 4: Run the numerical tests and verify GREEN**

Run the command from Step 2.

Expected: `Ran 3 tests` and `OK`.

### Task 2: Specify and add the isolated detector route

**Files:**
- Modify: `tests/test_lskd_integration.py`
- Modify: `model/config.py`
- Modify: `model/rcnn.py`

- [ ] **Step 1: Add AST helpers and failing route/default tests**

Append before the existing test class definitions:

```python
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


def _lskd_branch():
    tree = ast.parse(RCNN_PATH.read_text())
    trainer = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "RCNNKD"
    )
    forward = next(
        node for node in trainer.body
        if isinstance(node, ast.FunctionDef) and node.name == "forward"
    )
    for node in ast.walk(forward):
        if not isinstance(node, ast.If) or not isinstance(node.test, ast.Compare):
            continue
        strings = {
            item.value for item in ast.walk(node.test)
            if isinstance(item, ast.Constant) and isinstance(item.value, str)
        }
        if "LSKD" in strings:
            return ast.Module(body=node.body, type_ignores=[])
    raise AssertionError("missing RCNNKD route: LSKD")
```

Append these tests:

```python
class LSKDConfigDefaultTests(unittest.TestCase):
    def test_defaults_match_existing_project_kd(self):
        assignments = _assigned_literals(CONFIG_PATH)
        self.assertEqual(assignments["cfg.KD.LSKD.T"], 1.0)
        self.assertEqual(assignments["cfg.KD.LSKD.LOSS_WEIGHT"], 1.0)


class LSKDRouteTests(unittest.TestCase):
    def setUp(self):
        self.branch = _lskd_branch()

    def test_route_imports_and_calls_standalone_loss(self):
        module_tree = ast.parse(RCNN_PATH.read_text())
        imports = {
            (node.module, alias.name)
            for node in module_tree.body
            if isinstance(node, ast.ImportFrom)
            for alias in node.names
        }
        self.assertIn(("lskd", "lskd_loss"), imports)
        calls = {
            ast.unparse(node.func)
            for node in ast.walk(self.branch)
            if isinstance(node, ast.Call)
        }
        self.assertIn("lskd_loss", calls)

    def test_route_aligns_student_and_teacher_on_same_proposals(self):
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

    def test_route_adds_only_weighted_loss_lskd(self):
        source = ast.unparse(self.branch)
        self.assertIn("self.kd_args.LSKD.T", source)
        self.assertIn("self.kd_args.LSKD.LOSS_WEIGHT", source)
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
        self.assertEqual(loss_keys, {"loss_lskd"})
```

- [ ] **Step 2: Run route/default tests and verify RED**

Run:

```bash
/opt/anaconda3/bin/python3.9 -m unittest \
  tests.test_lskd_integration.LSKDConfigDefaultTests \
  tests.test_lskd_integration.LSKDRouteTests -v
```

Expected: missing `cfg.KD.LSKD.*` assignments and `missing RCNNKD route: LSKD`.

- [ ] **Step 3: Add only the new configuration defaults**

Insert after the existing SDKD defaults in `model/config.py`:

```python
    # LSKD
    cfg.KD.LSKD = CN()
    cfg.KD.LSKD.T = 1.0
    cfg.KD.LSKD.LOSS_WEIGHT = 1.0
```

- [ ] **Step 4: Add only the import and LSKD route**

Add to the local imports in `model/rcnn.py`:

```python
from .lskd import lskd_loss
```

Insert a new branch immediately before the existing `DKD` branch:

```python
        elif self.kd_args.TYPE == "LSKD":
            teacher_images = self.teacher_preprocess_image(batched_inputs)
            t_features = self.teacher.backbone(teacher_images.tensor)
            _, stu_predictions = self.forward_pure_roi_head(
                self.roi_heads, features, sampled_proposals)
            _, tea_predictions = self.forward_pure_roi_head(
                self.teacher.roi_heads, t_features, sampled_proposals)
            stu_logits, _ = stu_predictions
            tea_logits, _ = tea_predictions
            losses["loss_lskd"] = lskd_loss(
                stu_logits,
                tea_logits,
                self.kd_args.LSKD.T,
            ) * self.kd_args.LSKD.LOSS_WEIGHT
```

- [ ] **Step 5: Run route/default and numerical tests and verify GREEN**

Run:

```bash
/opt/anaconda3/bin/python3.9 -m unittest tests.test_lskd_integration -v
```

Expected: all currently defined LSKD tests pass.

### Task 3: Specify and add three detection configurations

**Files:**
- Modify: `tests/test_lskd_integration.py`
- Create: `configs/LSKD/LSKD-R18-R101.yaml`
- Create: `configs/LSKD/LSKD-R50-R101.yaml`
- Create: `configs/LSKD/LSKD-MV2-R50.yaml`

- [ ] **Step 1: Add failing architecture configuration tests**

Append:

```python
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


class LSKDArchitectureConfigTests(unittest.TestCase):
    def test_all_detection_configs_match_confirmed_settings(self):
        output_dirs = []
        for architecture, spec in ARCHITECTURES.items():
            with self.subTest(architecture=architecture):
                path = LSKD_CONFIG_DIR / "LSKD-{}.yaml".format(architecture)
                self.assertTrue(path.exists(), "missing configuration: {}".format(path))
                config = yaml.safe_load(path.read_text())
                self.assertEqual(config["_BASE_"], "../Base-Distillation.yaml")
                self.assertEqual(config["OUTPUT_DIR"], "output/LSKD-{}".format(architecture))
                output_dirs.append(config["OUTPUT_DIR"])
                self.assertEqual(config["MODEL"]["WEIGHTS"], spec["weights"])
                self.assertEqual(config["MODEL"]["BACKBONE"]["NAME"], spec["backbone"])
                self.assertEqual(config["MODEL"]["FPN"]["IN_FEATURES"], spec["features"])
                self.assertNotIn("MASK_ON", config["MODEL"])
                self.assertEqual(
                    config["TEACHER"]["MODEL"]["RESNETS"]["DEPTH"],
                    spec["teacher_depth"],
                )
                self.assertEqual(config["KD"]["TYPE"], "LSKD")
                self.assertEqual(config["KD"]["LSKD"], {"T": 1.0, "LOSS_WEIGHT": 1.0})
                self.assertEqual(config["SOLVER"]["IMS_PER_BATCH"], 8)
                self.assertEqual(config["SOLVER"]["BASE_LR"], 0.01)
                self.assertEqual(config["SOLVER"]["MAX_ITER"], 180000)
                self.assertEqual(config["SOLVER"]["CHECKPOINT_PERIOD"], 180001)
                self.assertEqual(config["SOLVER"]["STEPS"], [120000, 160000])

                if "student_depth" in spec:
                    self.assertEqual(config["MODEL"]["RESNETS"]["DEPTH"], spec["student_depth"])
                    self.assertEqual(config["MODEL"]["RESNETS"]["OUT_FEATURES"], spec["features"])
                else:
                    self.assertEqual(config["MODEL"]["BACKBONE"]["FREEZE_AT"], 0)
                    self.assertEqual(config["MODEL"]["MOBILENETV2"]["OUT_FEATURES"], spec["features"])

        self.assertEqual(len(output_dirs), len(set(output_dirs)))
```

- [ ] **Step 2: Run configuration tests and verify RED**

Run:

```bash
/opt/anaconda3/bin/python3.9 -m unittest \
  tests.test_lskd_integration.LSKDArchitectureConfigTests -v
```

Expected: failures reporting all three missing configuration files.

- [ ] **Step 3: Create the R18-R101 configuration**

Create `configs/LSKD/LSKD-R18-R101.yaml`:

```yaml
_BASE_: "../Base-Distillation.yaml"
OUTPUT_DIR: output/LSKD-R18-R101
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
  TYPE: "LSKD"
  LSKD:
    T: 1.0
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

- [ ] **Step 4: Create the R50-R101 configuration**

Create `configs/LSKD/LSKD-R50-R101.yaml`:

```yaml
_BASE_: "../Base-Distillation.yaml"
OUTPUT_DIR: output/LSKD-R50-R101
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
  TYPE: "LSKD"
  LSKD:
    T: 1.0
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

- [ ] **Step 5: Create the MV2-R50 configuration**

Create `configs/LSKD/LSKD-MV2-R50.yaml`:

```yaml
_BASE_: "../Base-Distillation.yaml"
OUTPUT_DIR: output/LSKD-MV2-R50
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
  TYPE: "LSKD"
  LSKD:
    T: 1.0
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

- [ ] **Step 6: Run all LSKD tests and verify GREEN**

Run:

```bash
/opt/anaconda3/bin/python3.9 -m unittest tests.test_lskd_integration -v
```

Expected: all LSKD numerical, route, and configuration tests pass.

### Task 4: Regression and static verification

**Files:**
- Verify: `model/lskd.py`
- Verify: `model/config.py`
- Verify: `model/rcnn.py`
- Verify: `configs/LSKD/*.yaml`
- Verify: `tests/test_lskd_integration.py`

- [ ] **Step 1: Compile all changed Python files**

Run:

```bash
/opt/anaconda3/bin/python3.9 -m py_compile \
  model/lskd.py model/config.py model/rcnn.py tests/test_lskd_integration.py
```

Expected: exit status 0 with no output.

- [ ] **Step 2: Run the complete local test suite**

Run:

```bash
/opt/anaconda3/bin/python3.9 -m unittest discover -s tests -v
```

Expected: all existing and new tests pass. If a pre-existing test fails because
of an unrelated checked-in configuration mismatch, record it separately and do
not alter that method to make this feature pass.

- [ ] **Step 3: Inspect scope mechanically**

Run:

```bash
find model configs/LSKD tests docs/superpowers \
  -path '*/__pycache__' -prune -o \
  -type f -newermt '2026-10-28 00:00:00' -print
```

Expected feature files are the new design/plan, `model/lskd.py`, three LSKD
YAMLs, one LSKD test file, and the surgical additions to `model/config.py` and
`model/rcnn.py`. No existing method YAML or loss module should be changed.

- [ ] **Step 4: Record the runtime verification boundary**

Do not claim COCO accuracy or full CUDA training success from unit tests. The
first real run should use one of the new YAMLs and verify finite `loss_lskd`,
periodic `bbox/AP`, and `model_best.pth` selection by the highest `bbox/AP`.
