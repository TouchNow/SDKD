# ReviewSDKD Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `ReviewSDKD` training method that returns the existing ReviewKD feature loss together with the existing SDKD INARank and AHTKD logit losses for all non-Mask student-teacher pairs already supported by SDKD.

**Architecture:** Extend the existing `RCNNKD` type routing without changing any existing branch. Reuse the existing ReviewKD adapter/HCL implementation and `sdkd_loss`; configure both through their existing config nodes.

**Tech Stack:** Python 3, PyTorch, Detectron2, YACS YAML configuration, `unittest`, Python AST

---

The workspace is not a Git repository, so commit steps are unavailable. Verification will use focused tests, syntax compilation, YAML parsing, and final file inspection.

### Task 1: Add failing ReviewSDKD integration tests

**Files:**
- Create: `tests/test_review_sdkd_integration.py`
- Inspect: `model/rcnn.py`
- Inspect: `configs/SDKD/ReviewSDKD-R18-R101.yaml`

- [ ] **Step 1: Write the failing test**

```python
import ast
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
RCNN_PATH = ROOT / "model" / "rcnn.py"
CONFIG_PATH = ROOT / "configs" / "SDKD" / "ReviewSDKD-R18-R101.yaml"


def _method_node(class_name, method_name):
    tree = ast.parse(RCNN_PATH.read_text())
    class_node = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    return next(
        node for node in class_node.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == method_name
    )


def _selects_type(test, type_name):
    return (
        isinstance(test, ast.Compare)
        and any(isinstance(op, ast.Eq) for op in test.ops)
        and any(
            isinstance(node, ast.Constant) and node.value == type_name
            for node in ast.walk(test)
        )
    )


class ReviewSDKDIntegrationTests(unittest.TestCase):
    def test_review_sdkd_initializes_review_adapter(self):
        init_node = _method_node("RCNNKD", "__init__")
        supported_types = set()
        for node in ast.walk(init_node):
            if not isinstance(node, ast.Compare):
                continue
            if not any(isinstance(op, ast.In) for op in node.ops):
                continue
            for item in ast.walk(node):
                if isinstance(item, ast.Constant) and isinstance(item.value, str):
                    supported_types.add(item.value)
        self.assertIn("ReviewSDKD", supported_types)

    def test_review_sdkd_branch_combines_sdkd_and_reviewkd(self):
        forward_node = _method_node("RCNNKD", "forward")
        branch = next(
            (
                node for node in ast.walk(forward_node)
                if isinstance(node, ast.If)
                and _selects_type(node.test, "ReviewSDKD")
            ),
            None,
        )
        self.assertIsNotNone(branch)
        branch_tree = ast.Module(body=branch.body, type_ignores=[])
        call_names = {
            ast.unparse(node.func)
            for node in ast.walk(branch_tree)
            if isinstance(node, ast.Call)
        }
        string_values = {
            node.value
            for node in ast.walk(branch_tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        }
        self.assertIn("sdkd_loss", call_names)
        self.assertIn("hcl", call_names)
        self.assertIn("loss_reviewkd", string_values)

    def test_review_sdkd_config_uses_confirmed_weights(self):
        config = yaml.safe_load(CONFIG_PATH.read_text())
        self.assertEqual(config["OUTPUT_DIR"], "output/ReviewSDKD-R18-R101")
        self.assertEqual(config["KD"]["TYPE"], "ReviewSDKD")
        self.assertEqual(config["KD"]["REVIEWKD"]["LOSS_WEIGHT"], 1.2)
        self.assertEqual(
            config["KD"]["SDKD"],
            {
                "INARANK_WEIGHT": 0.5,
                "T": 1.0,
                "HEAD_WEIGHT": 1.0,
                "TAIL_WEIGHT": 0.25,
            },
        )


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
python -m unittest tests.test_review_sdkd_integration -v
```

Expected: failures because `ReviewSDKD` is not in `RCNNKD` and the new YAML does not exist.

### Task 2: Add the minimal ReviewSDKD route and loss composition

**Files:**
- Modify: `model/rcnn.py:126`
- Modify: `model/rcnn.py` immediately after the existing `ReviewDKD` branch
- Test: `tests/test_review_sdkd_integration.py`

- [ ] **Step 1: Extend ReviewKD adapter initialization**

Add only `"ReviewSDKD"` to the existing supported-type tuple:

```python
if self.kd_args.TYPE in (
        "NDReviewKD", "ReviewKD", "ReviewDKD", "LTFReviewKD", "ReviewSDKD"
):
    self.kd_trans = build_kd_trans(self.kd_args)
```

- [ ] **Step 2: Add the independent combination branch**

```python
elif self.kd_args.TYPE == "ReviewSDKD":
    teacher_images = self.teacher_preprocess_image(batched_inputs)
    t_features = self.teacher.backbone(teacher_images.tensor)
    s_emb, stu_predictions = self.forward_pure_roi_head(
        self.roi_heads, features, sampled_proposals
    )
    t_emb, tea_predictions = self.forward_pure_roi_head(
        self.teacher.roi_heads, t_features, sampled_proposals
    )
    detector_losses.update(sdkd_loss(
        stu_predictions, tea_predictions,
        [x.gt_classes for x in sampled_proposals],
        self.kd_args.SDKD.INARANK_WEIGHT,
        self.kd_args.SDKD.T,
        self.kd_args.SDKD.HEAD_WEIGHT,
        self.kd_args.SDKD.TAIL_WEIGHT,
    ))
    t_features = [t_features[f] for f in t_features]
    s_features = [features[f] for f in features]
    s_features = self.kd_trans(s_features)
    losses["loss_reviewkd"] = (
        hcl(s_features, t_features) * self.kd_args.REVIEWKD.LOSS_WEIGHT
    )
```

- [ ] **Step 3: Run the focused test**

Run:

```bash
python -m unittest tests.test_review_sdkd_integration -v
```

Expected: routing and branch tests pass; configuration test still fails because the YAML is absent.

### Task 3: Add the ReviewSDKD R18 to R101 configuration

**Files:**
- Create: `configs/SDKD/ReviewSDKD-R18-R101.yaml`
- Test: `tests/test_review_sdkd_integration.py`

- [ ] **Step 1: Create the combined configuration**

Copy the unchanged R18 to R101 model and solver settings from `SDKD-R18-R101.yaml`, change the output/type, and define both existing config nodes:

```yaml
_BASE_: "../Base-Distillation.yaml"
OUTPUT_DIR: output/ReviewSDKD-R18-R101
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
  TYPE: "ReviewSDKD"
  REVIEWKD:
    LOSS_WEIGHT: 1.2
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

- [ ] **Step 2: Run the focused test**

Run:

```bash
python -m unittest tests.test_review_sdkd_integration -v
```

Expected: all three tests pass.

### Task 4: Verify syntax and scope

**Files:**
- Verify: `model/rcnn.py`
- Verify: `model/sdkd.py`
- Verify: `tests/test_review_sdkd_integration.py`
- Verify: `configs/SDKD/ReviewSDKD-R18-R101.yaml`

- [ ] **Step 1: Compile Python files**

Run:

```bash
python -m py_compile model/rcnn.py model/sdkd.py tests/test_review_sdkd_integration.py
```

Expected: exit code 0 with no output.

- [ ] **Step 2: Parse the YAML independently**

Run:

```bash
python -c "import yaml; p='configs/SDKD/ReviewSDKD-R18-R101.yaml'; c=yaml.safe_load(open(p)); print(c['KD'])"
```

Expected: `TYPE`, `REVIEWKD`, and `SDKD` show the confirmed values.

- [ ] **Step 3: Inspect the exact changed surface**

Run:

```bash
rg -n "ReviewSDKD" model/rcnn.py configs/SDKD/ReviewSDKD-R18-R101.yaml tests/test_review_sdkd_integration.py
```

Expected: only the new route, branch, config, and focused test contain the new method name.

### Task 5: Add failing tests for the remaining architecture configurations

**Files:**
- Modify: `tests/test_review_sdkd_integration.py`
- Test: `configs/SDKD/ReviewSDKD-R50-R101.yaml`
- Test: `configs/SDKD/ReviewSDKD-MV2-R50.yaml`

- [ ] **Step 1: Add one test for each remaining architecture pair**

Add this helper and the two test methods without changing the existing R18 test:

```python
def _load_sdkd_config(filename):
    path = ROOT / "configs" / "SDKD" / filename
    return yaml.safe_load(path.read_text())


def _assert_shared_sdkd_config(test_case, config):
    test_case.assertEqual(config["KD"]["TYPE"], "ReviewSDKD")
    test_case.assertEqual(
        config["KD"]["SDKD"],
        {
            "INARANK_WEIGHT": 0.5,
            "T": 1.0,
            "HEAD_WEIGHT": 1.0,
            "TAIL_WEIGHT": 0.25,
        },
    )
```

```python
def test_review_sdkd_r50_r101_config(self):
    config = _load_sdkd_config("ReviewSDKD-R50-R101.yaml")
    _assert_shared_sdkd_config(self, config)
    self.assertEqual(config["OUTPUT_DIR"], "output/ReviewSDKD-R50-R101")
    self.assertEqual(config["MODEL"]["WEIGHTS"], "pretrained/r50-r101.pth")
    self.assertEqual(config["MODEL"]["RESNETS"]["DEPTH"], 50)
    self.assertEqual(config["TEACHER"]["MODEL"]["RESNETS"]["DEPTH"], 101)
    self.assertEqual(config["KD"]["REVIEWKD"]["LOSS_WEIGHT"], 1.0)

def test_review_sdkd_mv2_r50_config(self):
    config = _load_sdkd_config("ReviewSDKD-MV2-R50.yaml")
    _assert_shared_sdkd_config(self, config)
    self.assertEqual(config["OUTPUT_DIR"], "output/ReviewSDKD-MV2-R50")
    self.assertEqual(config["MODEL"]["WEIGHTS"], "pretrained/mv2-r50.pth")
    self.assertEqual(
        config["MODEL"]["BACKBONE"]["NAME"],
        "build_mobilenetv2_fpn_backbone",
    )
    self.assertEqual(
        config["MODEL"]["MOBILENETV2"]["OUT_FEATURES"],
        ["m2", "m3", "m4", "m5"],
    )
    self.assertEqual(config["TEACHER"]["MODEL"]["RESNETS"]["DEPTH"], 50)
    self.assertEqual(config["KD"]["REVIEWKD"]["LOSS_WEIGHT"], 2.0)
```

- [ ] **Step 2: Run the focused test to verify red**

Run:

```bash
python -m unittest tests.test_review_sdkd_integration -v
```

Expected: the three existing tests pass, while the R50→R101 and MV2→R50 tests fail with `FileNotFoundError` for their missing YAML files.

### Task 6: Add the remaining explicit ReviewSDKD configurations

**Files:**
- Create: `configs/SDKD/ReviewSDKD-R50-R101.yaml`
- Create: `configs/SDKD/ReviewSDKD-MV2-R50.yaml`
- Test: `tests/test_review_sdkd_integration.py`

- [ ] **Step 1: Create `ReviewSDKD-R50-R101.yaml`**

```yaml
_BASE_: "../Base-Distillation.yaml"
OUTPUT_DIR: output/ReviewSDKD-R50-R101
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

- [ ] **Step 2: Create `ReviewSDKD-MV2-R50.yaml`**

```yaml
_BASE_: "../Base-Distillation.yaml"
OUTPUT_DIR: output/ReviewSDKD-MV2-R50
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
  TYPE: "ReviewSDKD"
  REVIEWKD:
    LOSS_WEIGHT: 2.0
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

- [ ] **Step 3: Run all focused tests**

Run:

```bash
python -m unittest discover -s tests -v
```

Expected: all five ReviewSDKD tests pass.

- [ ] **Step 4: Compile and parse all ReviewSDKD artifacts**

Run:

```bash
python -m py_compile model/rcnn.py model/sdkd.py tests/test_review_sdkd_integration.py
```

Expected: exit code 0 with no output.

Run:

```bash
python -c "import glob,yaml; files=sorted(glob.glob('configs/SDKD/ReviewSDKD-*.yaml')); print([(f, yaml.safe_load(open(f))['KD']) for f in files])"
```

Expected: three configuration files with `TYPE: ReviewSDKD`, the confirmed ReviewKD weights, and identical SDKD values.
