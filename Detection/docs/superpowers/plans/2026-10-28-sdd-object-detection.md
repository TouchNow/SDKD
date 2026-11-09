# SDD Object Detection Implementation Plan

**Goal:** Add isolated SDD + KD training for Faster R-CNN using confirmed
proposal scales `[1, 2]` and the project's existing KD temperature and weight.

## 1. Specify behavior with failing tests

- Add `tests/test_sdd_integration.py`.
- Cover region generation, official SDD weighting, gradients, config defaults,
  route isolation, and the three architecture configurations.
- Verify the new test fails because SDD has not been implemented.

## 2. Add the standalone SDD implementation

- Add `model/sdd.py` with the minimum tensor-only helpers needed by SDD.
- Implement scale-grid box construction and region-major logit reshaping.
- Implement temperature-scaled, complementary-weighted SDD KL loss.
- Verify the pure loss and geometry tests pass.

## 3. Integrate only the SDD route

- Add `KD.SDD` defaults to `model/config.py`.
- Add an SDD-specific ROI forwarding helper and `SDD` branch to
  `model/rcnn.py`.
- Reuse `sampled_proposals` for student and teacher.
- Verify route and default tests pass.

## 4. Add runnable configurations

- Add R18-R101, R50-R101, and MV2-R50 files under `configs/SDD/`, following
  the corresponding existing KD/LSKD architecture settings.
- Verify all configuration tests pass.

## 5. Regression verification

- Compile changed Python files.
- Run the complete test suite.
- Confirm no new failures beyond the four already identified baseline
  expectation mismatches.
