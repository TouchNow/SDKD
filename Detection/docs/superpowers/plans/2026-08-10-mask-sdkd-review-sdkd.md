# Mask SDKD and ReviewSDKD Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add six configuration-only Mask R-CNN experiments that combine the standard `loss_mask` with SDKD logits distillation, with optional ReviewKD feature distillation.

**Architecture:** Reuse the existing `KD.TYPE: "SDKD"` and `KD.TYPE: "ReviewSDKD"` routes in `RCNNKD`; `MODEL.MASK_ON: True` independently enables the student mask head. Production changes consist only of six new YAML files under `configs/SDKD/`, with one new static regression test proving their exact contents and the reused loss routes.

**Tech Stack:** Python 3, `unittest`, Python `ast`, PyYAML, Detectron2 YAML configuration, Mask R-CNN, SDKD, ReviewKD

---

## File map and constraints

- Create `tests/test_mask_sdkd_configs.py`: parse and validate all six new configurations without importing Detectron2; characterize the existing SDKD and ReviewSDKD loss routes through the Python AST.
- Create `configs/SDKD/SDKD-R18-R101-Mask.yaml`: R18 student, R101 teacher, standard mask loss plus SDKD logits loss.
- Create `configs/SDKD/SDKD-R50-R101-Mask.yaml`: R50 student, R101 teacher, standard mask loss plus SDKD logits loss.
- Create `configs/SDKD/SDKD-MV2-R50-Mask.yaml`: MobileNetV2 student, R50 teacher, standard mask loss plus SDKD logits loss.
- Create `configs/SDKD/ReviewSDKD-R18-R101-Mask.yaml`: R18/R101 ReviewKD-Mask with added SDKD logits loss.
- Create `configs/SDKD/ReviewSDKD-R50-R101-Mask.yaml`: R50/R101 ReviewKD-Mask with added SDKD logits loss.
- Create `configs/SDKD/ReviewSDKD-MV2-R50-Mask.yaml`: MV2/R50 ReviewKD-Mask with added SDKD logits loss.
- Do not modify any pre-existing source, configuration, test, or documentation file.
- Git commit steps are omitted because `/Users/touchnow/Documents/Pycharm/Detection` has no `.git` repository.

### Task 1: Add failing configuration contract tests

**Files:**
- Create: `tests/test_mask_sdkd_configs.py`
- Inspect only: `model/rcnn.py`

- [ ] **Step 1: Create the complete regression test**

Create `tests/test_mask_sdkd_configs.py` with:

```python
import ast
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs" / "SDKD"
RCNN_PATH = ROOT / "model" / "rcnn.py"

SDKD_ARGS = {
    "T": 1.0,
    "HEAD_WEIGHT": 1.0,
    "TAIL_WEIGHT": 0.25,
}

ARCHITECTURES = {
    "R18-R101": {
        "weights": "pretrained/r18-r101mask.pth",
        "backbone": "build_resnet_fpn_backbone_kd",
        "student_family": "resnet",
        "student_depth": 18,
        "teacher_depth": 101,
        "review_weight": 1.5,
    },
    "R50-R101": {
        "weights": "pretrained/r50-r101mask.pth",
        "backbone": "build_resnet_fpn_backbone_kd",
        "student_family": "resnet",
        "student_depth": 50,
        "teacher_depth": 101,
        "review_weight": 0.8,
    },
    "MV2-R50": {
        "weights": "pretrained/mv2-r50mask.pth",
        "backbone": "build_mobilenetv2_fpn_backbone",
        "student_family": "mobilenetv2",
        "teacher_depth": 50,
        "review_weight": 1.0,
    },
}


def _load_config(filename):
    path = CONFIG_DIR / filename
    if not path.exists():
        raise AssertionError(f"missing configuration: {path}")
    return yaml.safe_load(path.read_text())


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
    raise AssertionError(f"missing RCNNKD route: {type_name}")


class MaskSDKDConfigTests(unittest.TestCase):
    def _assert_config(self, method, architecture):
        spec = ARCHITECTURES[architecture]
        filename = f"{method}-{architecture}-Mask.yaml"
        config = _load_config(filename)

        self.assertEqual(config["_BASE_"], "../Base-Distillation.yaml")
        self.assertEqual(config["OUTPUT_DIR"], f"output/{method}-{architecture}-Mask")
        self.assertIs(config["MODEL"]["MASK_ON"], True)
        self.assertEqual(config["MODEL"]["WEIGHTS"], spec["weights"])
        self.assertEqual(config["MODEL"]["BACKBONE"]["NAME"], spec["backbone"])
        self.assertEqual(
            config["TEACHER"]["MODEL"]["RESNETS"]["DEPTH"],
            spec["teacher_depth"],
        )
        self.assertEqual(config["SOLVER"]["IMS_PER_BATCH"], 8)
        self.assertEqual(config["SOLVER"]["BASE_LR"], 0.01)
        self.assertEqual(config["SOLVER"]["MAX_ITER"], 180000)
        self.assertEqual(config["SOLVER"]["STEPS"], [120000, 160000])

        if spec["student_family"] == "resnet":
            self.assertEqual(
                config["MODEL"]["RESNETS"]["DEPTH"],
                spec["student_depth"],
            )
            self.assertEqual(
                config["MODEL"]["RESNETS"]["OUT_FEATURES"],
                ["res2", "res3", "res4", "res5"],
            )
        else:
            self.assertEqual(config["MODEL"]["BACKBONE"]["FREEZE_AT"], 0)
            self.assertEqual(
                config["MODEL"]["MOBILENETV2"]["OUT_FEATURES"],
                ["m2", "m3", "m4", "m5"],
            )

        self.assertEqual(config["KD"]["TYPE"], method)
        expected_sdkd = dict(SDKD_ARGS)
        expected_sdkd["INARANK_WEIGHT"] = 0.1 if method == "SDKD" else 0.5
        self.assertEqual(config["KD"]["SDKD"], expected_sdkd)

        if method == "ReviewSDKD":
            self.assertEqual(
                config["KD"]["REVIEWKD"]["LOSS_WEIGHT"],
                spec["review_weight"],
            )
        else:
            self.assertNotIn("REVIEWKD", config["KD"])

    def test_sdkd_mask_configs(self):
        for architecture in ARCHITECTURES:
            with self.subTest(architecture=architecture):
                self._assert_config("SDKD", architecture)

    def test_review_sdkd_mask_configs(self):
        for architecture in ARCHITECTURES:
            with self.subTest(architecture=architecture):
                self._assert_config("ReviewSDKD", architecture)

    def test_mask_output_directories_are_unique(self):
        output_dirs = []
        for method in ("SDKD", "ReviewSDKD"):
            for architecture in ARCHITECTURES:
                config = _load_config(f"{method}-{architecture}-Mask.yaml")
                output_dirs.append(config["OUTPUT_DIR"])
        self.assertEqual(len(output_dirs), len(set(output_dirs)))
        self.assertTrue(all(path.endswith("-Mask") for path in output_dirs))

    def test_existing_routes_support_configuration_only_extension(self):
        sdkd_branch = _branch_for_type("SDKD")
        review_sdkd_branch = _branch_for_type("ReviewSDKD")

        sdkd_calls = {
            ast.unparse(node.func)
            for node in ast.walk(sdkd_branch)
            if isinstance(node, ast.Call)
        }
        review_sdkd_calls = {
            ast.unparse(node.func)
            for node in ast.walk(review_sdkd_branch)
            if isinstance(node, ast.Call)
        }
        review_sdkd_strings = {
            node.value
            for node in ast.walk(review_sdkd_branch)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        }

        self.assertIn("sdkd_loss", sdkd_calls)
        self.assertIn("sdkd_loss", review_sdkd_calls)
        self.assertIn("hcl", review_sdkd_calls)
        self.assertIn("loss_reviewkd", review_sdkd_strings)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the SDKD-Mask contract and verify the RED state**

Run:

```bash
python3 -m unittest tests.test_mask_sdkd_configs.MaskSDKDConfigTests.test_sdkd_mask_configs -v
```

Expected: FAIL with `AssertionError: missing configuration` for `configs/SDKD/SDKD-R18-R101-Mask.yaml`. The failure must be caused by the absent feature files, not a syntax or import error.

### Task 2: Add the three SDKD-Mask configurations

**Files:**
- Create: `configs/SDKD/SDKD-R18-R101-Mask.yaml`
- Create: `configs/SDKD/SDKD-R50-R101-Mask.yaml`
- Create: `configs/SDKD/SDKD-MV2-R50-Mask.yaml`
- Test: `tests/test_mask_sdkd_configs.py`

- [ ] **Step 1: Create the R18/R101 SDKD-Mask configuration**

Create `configs/SDKD/SDKD-R18-R101-Mask.yaml`:

```yaml
_BASE_: "../Base-Distillation.yaml"
OUTPUT_DIR: output/SDKD-R18-R101-Mask
INPUT:
  FORMAT: 'RGB'
MODEL:
  PIXEL_STD: [57.375, 57.120, 58.395]
  MASK_ON: True
  BACKBONE:
    NAME: "build_resnet_fpn_backbone_kd"
  WEIGHTS: "pretrained/r18-r101mask.pth"
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
  TYPE: "SDKD"
  SDKD:
    INARANK_WEIGHT: 0.1
    T: 1.0
    HEAD_WEIGHT: 1.0
    TAIL_WEIGHT: 0.25

SOLVER:
  IMS_PER_BATCH: 8
  BASE_LR: 0.01
  MAX_ITER: 180000
  CHECKPOINT_PERIOD: 180001
  STEPS:
    - 120000
    - 160000
```

- [ ] **Step 2: Create the R50/R101 SDKD-Mask configuration**

Create `configs/SDKD/SDKD-R50-R101-Mask.yaml`:

```yaml
_BASE_: "../Base-Distillation.yaml"
OUTPUT_DIR: output/SDKD-R50-R101-Mask
MODEL:
  MASK_ON: True
  BACKBONE:
    NAME: "build_resnet_fpn_backbone_kd"
  WEIGHTS: "pretrained/r50-r101mask.pth"
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
  TYPE: "SDKD"
  SDKD:
    INARANK_WEIGHT: 0.1
    T: 1.0
    HEAD_WEIGHT: 1.0
    TAIL_WEIGHT: 0.25

SOLVER:
  IMS_PER_BATCH: 8
  BASE_LR: 0.01
  MAX_ITER: 180000
  CHECKPOINT_PERIOD: 180001
  STEPS:
    - 120000
    - 160000
```

- [ ] **Step 3: Create the MV2/R50 SDKD-Mask configuration**

Create `configs/SDKD/SDKD-MV2-R50-Mask.yaml`:

```yaml
_BASE_: "../Base-Distillation.yaml"
OUTPUT_DIR: output/SDKD-MV2-R50-Mask
MODEL:
  MASK_ON: True
  BACKBONE:
    NAME: "build_mobilenetv2_fpn_backbone"
    FREEZE_AT: 0
  WEIGHTS: "pretrained/mv2-r50mask.pth"
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
  TYPE: "SDKD"
  SDKD:
    INARANK_WEIGHT: 0.1
    T: 1.0
    HEAD_WEIGHT: 1.0
    TAIL_WEIGHT: 0.25

SOLVER:
  IMS_PER_BATCH: 8
  BASE_LR: 0.01
  MAX_ITER: 180000
  CHECKPOINT_PERIOD: 180001
  STEPS:
    - 120000
    - 160000
```

- [ ] **Step 4: Run the SDKD-Mask contract and verify the GREEN state**

Run:

```bash
python3 -m unittest tests.test_mask_sdkd_configs.MaskSDKDConfigTests.test_sdkd_mask_configs -v
```

Expected: PASS, one test with three successful architecture subtests.

- [ ] **Step 5: Run the ReviewSDKD-Mask contract and verify its RED state**

Run:

```bash
python3 -m unittest tests.test_mask_sdkd_configs.MaskSDKDConfigTests.test_review_sdkd_mask_configs -v
```

Expected: FAIL with `AssertionError: missing configuration` for `configs/SDKD/ReviewSDKD-R18-R101-Mask.yaml`.

### Task 3: Add the three ReviewSDKD-Mask configurations

**Files:**
- Create: `configs/SDKD/ReviewSDKD-R18-R101-Mask.yaml`
- Create: `configs/SDKD/ReviewSDKD-R50-R101-Mask.yaml`
- Create: `configs/SDKD/ReviewSDKD-MV2-R50-Mask.yaml`
- Test: `tests/test_mask_sdkd_configs.py`

- [ ] **Step 1: Create the R18/R101 ReviewSDKD-Mask configuration**

Create `configs/SDKD/ReviewSDKD-R18-R101-Mask.yaml`:

```yaml
_BASE_: "../Base-Distillation.yaml"
OUTPUT_DIR: output/ReviewSDKD-R18-R101-Mask
INPUT:
  FORMAT: 'RGB'
MODEL:
  PIXEL_STD: [57.375, 57.120, 58.395]
  MASK_ON: True
  BACKBONE:
    NAME: "build_resnet_fpn_backbone_kd"
  WEIGHTS: "pretrained/r18-r101mask.pth"
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
  TYPE: "ReviewSDKD"
  REVIEWKD:
    LOSS_WEIGHT: 1.5
  SDKD:
    INARANK_WEIGHT: 0.5
    T: 1.0
    HEAD_WEIGHT: 1.0
    TAIL_WEIGHT: 0.25

SOLVER:
  IMS_PER_BATCH: 8
  BASE_LR: 0.01
  MAX_ITER: 180000
  STEPS:
    - 120000
    - 160000
```

- [ ] **Step 2: Create the R50/R101 ReviewSDKD-Mask configuration**

Create `configs/SDKD/ReviewSDKD-R50-R101-Mask.yaml`:

```yaml
_BASE_: "../Base-Distillation.yaml"
OUTPUT_DIR: output/ReviewSDKD-R50-R101-Mask
MODEL:
  MASK_ON: True
  BACKBONE:
    NAME: "build_resnet_fpn_backbone_kd"
  WEIGHTS: "pretrained/r50-r101mask.pth"
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
  TYPE: "ReviewSDKD"
  REVIEWKD:
    LOSS_WEIGHT: 0.8
  SDKD:
    INARANK_WEIGHT: 0.5
    T: 1.0
    HEAD_WEIGHT: 1.0
    TAIL_WEIGHT: 0.25

SOLVER:
  IMS_PER_BATCH: 8
  BASE_LR: 0.01
  MAX_ITER: 180000
  STEPS:
    - 120000
    - 160000
```

- [ ] **Step 3: Create the MV2/R50 ReviewSDKD-Mask configuration**

Create `configs/SDKD/ReviewSDKD-MV2-R50-Mask.yaml`:

```yaml
_BASE_: "../Base-Distillation.yaml"
OUTPUT_DIR: output/ReviewSDKD-MV2-R50-Mask
MODEL:
  MASK_ON: True
  BACKBONE:
    NAME: "build_mobilenetv2_fpn_backbone"
    FREEZE_AT: 0
  WEIGHTS: "pretrained/mv2-r50mask.pth"
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
  TYPE: "ReviewSDKD"
  REVIEWKD:
    LOSS_WEIGHT: 1.0
  SDKD:
    INARANK_WEIGHT: 0.5
    T: 1.0
    HEAD_WEIGHT: 1.0
    TAIL_WEIGHT: 0.25

SOLVER:
  IMS_PER_BATCH: 8
  BASE_LR: 0.01
  MAX_ITER: 180000
  STEPS:
    - 120000
    - 160000
```

- [ ] **Step 4: Run the ReviewSDKD-Mask contract and verify the GREEN state**

Run:

```bash
python3 -m unittest tests.test_mask_sdkd_configs.MaskSDKDConfigTests.test_review_sdkd_mask_configs -v
```

Expected: PASS, one test with three successful architecture subtests.

- [ ] **Step 5: Verify the six output directories are unique**

Run:

```bash
python3 -m unittest tests.test_mask_sdkd_configs.MaskSDKDConfigTests.test_mask_output_directories_are_unique -v
```

Expected: PASS.

### Task 4: Verify behavior and the no-modification boundary

**Files:**
- Verify: `tests/test_mask_sdkd_configs.py`
- Verify: `tests/test_review_sdkd_integration.py`
- Verify: all six new `configs/SDKD/*-Mask.yaml` files
- Verify unchanged: `model/rcnn.py`, `model/sdkd.py`, `model/reviewkd.py`, `model/config.py`, `configs/Base-Distillation.yaml`

- [ ] **Step 1: Run the complete new regression test**

Run:

```bash
python3 -m unittest tests.test_mask_sdkd_configs -v
```

Expected: 4 tests pass.

- [ ] **Step 2: Run the existing unit-test suite**

Run:

```bash
python3 -m unittest discover -s tests -v
```

Expected: all discovered tests pass with no failures or errors.

- [ ] **Step 3: Compile the new Python test**

Run:

```bash
python3 -m py_compile tests/test_mask_sdkd_configs.py
```

Expected: exit status 0 with no output.

- [ ] **Step 4: List the new Mask configurations**

Run:

```bash
rg --files configs/SDKD | rg -- '-Mask\.yaml$' | sort
```

Expected: exactly the following six paths:

```text
configs/SDKD/ReviewSDKD-MV2-R50-Mask.yaml
configs/SDKD/ReviewSDKD-R18-R101-Mask.yaml
configs/SDKD/ReviewSDKD-R50-R101-Mask.yaml
configs/SDKD/SDKD-MV2-R50-Mask.yaml
configs/SDKD/SDKD-R18-R101-Mask.yaml
configs/SDKD/SDKD-R50-R101-Mask.yaml
```

- [ ] **Step 5: Confirm protected pre-existing files retain their initial hashes**

Run:

```bash
openssl dgst -sha256 \
  model/rcnn.py \
  model/sdkd.py \
  model/reviewkd.py \
  model/config.py \
  tests/test_review_sdkd_integration.py \
  configs/Base-Distillation.yaml
```

Expected hashes:

```text
SHA256(model/rcnn.py)= f4a0bf3d37ccf1591a59c924090e9b37216c8310603fe94d86662a3cf17445f2
SHA256(model/sdkd.py)= a483f4eb060c10b174b673783cd7c6053a585d499149e768d38ae7b781226e45
SHA256(model/reviewkd.py)= 8ea89536379245363fcc491f11d5b838a12c3eed62106999bfa0ce4328f0c93b
SHA256(model/config.py)= bdaaceb17930b4155e5e34cd30a66eb96ef087b4870e5c93b4d34c91175a6017
SHA256(tests/test_review_sdkd_integration.py)= 40c6326a24feab1f70b90688cf9dc8ed4540de33723c4ee28ca536f59af008a9
SHA256(configs/Base-Distillation.yaml)= 407ee32dca8f47f9e766ce5b7cf41fc43782282f486094797d9606bb871f9ac3
```

- [ ] **Step 6: Record the local verification boundary**

Report that static configuration and regression checks passed. Do not claim COCO training accuracy or runtime throughput because the local verification does not execute full Detectron2 training with pretrained checkpoints, COCO data, and GPUs.
