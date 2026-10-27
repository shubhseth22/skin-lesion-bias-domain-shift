# Project Status: Bias and Domain Shift in Skin-Lesion Classification

Last Updated: 2026-10-27

---

## 1. Executive Summary

This document serves as the dynamic status board for the CSE Major Research Project **"Bias and Domain Shift in Skin-Lesion Classification"**. It tracks all milestones across dataset preparation, model development, mitigation algorithms, statistical evaluation, backend services, and interactive visualization.

---

## 2. Status Board

### ✅ Completed Work
- [x] Initialized Git repository and configured remote tracking (`shubhseth22/skin-lesion-bias-domain-shift`).
- [x] Defined and established directory architecture separating `src/`, `src/research/`, `experiments/` (`baseline/`, `fairness/`, `domain_adaptation/`, `calibration/`), `backend/`, `frontend/`, `docs/`, `configs/`, and `tests/`.
- [x] Configured reinforced repository `.gitignore` providing universal protection against raw/processed medical datasets, model checkpoints (`*.pt`, `*.pth`, `*.bin`, `*.onnx`, `*.safetensors`, `*.ckpt`), experiment outputs (`experiments/**/runs/`, `experiments/**/checkpoints/`), virtual environments, and secrets (`*.pem`, `*.key`, `credentials.json`, `token.json`).
- [x] Created `AGENTS.md` specifying operational integrity guidelines for Gemini and Claude.
- [x] Drafted core documentation structure (`README.md`, `docs/architecture.md`, `docs/research_questions.md`, `docs/dataset_documentation.md`, `docs/experiment_log.md`, `docs/decisions.md`, `docs/team_contributions.md`).
- [x] Created modular YAML configurations:
  - Dataset configs: `configs/datasets/isic2019.yaml`, `configs/datasets/fitzpatrick17k.yaml`, `configs/datasets/pad_ufes_20.yaml`
  - Experiment configs: `configs/experiments/baseline_erm.yaml`, `configs/experiments/fairness_group_dro.yaml`, `configs/experiments/domain_adaptation_dann.yaml`
- [x] Implemented reproducible CLI driver `scripts/prepare_splits.py` with automated integrity auditing.
- [x] **Critical Fixes (1–3) [Reported Implemented]**:
  - **Fix 1 (Stratified Group Splitting)**: Implemented in `src/datasets/splits.py` (`create_stratified_group_split`) with multi-class deficit optimization, zero patient/lesion leakage guarantees, and explicit small-class/missing-group handling.
  - **Fix 2 (Label Taxonomy Harmonizer)**: Implemented in `src/datasets/taxonomy.py` (`HarmonizedCategory`, `LabelHarmonizer`) with clinical rationales, raw label preservation, and non-neoplastic / pre-malignant policy management.
  - **Fix 3 (Target Domain Adaptation Split)**: Implemented in `src/datasets/splits.py` (`create_domain_adaptation_split`) enforcing target test isolation, group-aware partitioning, and unsupervised target label masking.
- [x] **High-Priority Fixes (4–6) [Reported Implemented]**:
  - **Fix 4 (Test-to-Source Module Connection)**: Eliminated all local mock self-testing; tests directly exercise actual implementations in `src/`. Implemented `SubgroupEvaluator` (`src/research/subgroup_analysis.py`), `CalibrationEvaluator` (`src/evaluation/calibration.py`), `InferencePipeline` (`src/evaluation/inference.py`), and `DermatologyModelContract` (`src/models/contracts.py`).
  - **Fix 5 (ITA Epsilon Sign Guard & Stability)**: Fixed algebraic sign inversion in `src/datasets/base.py` using `np.copysign`, added physical $L^* \in [0, 100]$ bounds, handled $0/0$ indeterminate cases, and prevented `NaN` from silently defaulting to `"Dark"`.
  - **Fix 6 (Unlesioned Peripheral Skin Extraction Heuristic)**: Implemented `PeripheralSkinExtractor` in `src/datasets/skin_extractor.py` with border/vignette cropping, central lesion exclusion, biological chromaticity filtering, and uniformity quality scoring.
- [x] **Latest Test Suite Verification**:
  - Total tests: **63 passed, 0 failed** using `pytest` across all 7 test modules (`test_calibration.py`, `test_datasets.py`, `test_inference.py`, `test_metrics.py`, `test_models.py`, `test_splits.py`, `test_taxonomy.py`, `test_utils.py`).

### ⚠️ Current State & Scientific Governance Notes
- **Independent Verification Pending**: The six implemented fixes (Fixes 1–6) have not yet undergone independent source-code and scientific verification.
- **Phase Status**: Phase 3 has **NOT** started.
- **Pending Prompt**: Prompt 8 is pending.
- **Data & Model Status**: No dataset acquisition or model training has started as part of this phase.

### ⏳ Pending Work
- [ ] **Prompt 8 Action Items (Medium & Low Priority Audit Fixes)**:
  - [ ] Implement permutation disparity test improvements (group-membership shuffling in `src/research/statistical_tests.py`).
  - [ ] Implement missing patient-ID handling optimizations in dataset loaders.
  - [ ] Implement subgroup sparsity threshold guard in `src/research/subgroup_analysis.py`.
  - [ ] Document ADR-005 in `docs/decisions.md` (unlesioned peripheral skin extraction limitations and colorimetry assumptions).
- [ ] **Phase 1: Environment & Dataset Infrastructure**
  - [ ] Set up Python environment dependencies (`requirements.txt`, `environment.yml`).
  - [ ] Implement dataset download and verification script with license compliance checks.
  - [ ] Complete license verification for candidate datasets prior to drafting ingestion scripts.
- [ ] **Phase 2: Baseline Model Implementation & Training Pipeline**
  - [ ] Implement PyTorch model backbones (ResNet-50, EfficientNet, ViT).
  - [ ] Baseline empirical risk minimization (ERM) training pipeline.
- [ ] **Phase 3: Subgroup Bias & Domain Shift Research Suite**
  - [ ] Cross-dataset generalization evaluation across dermoscopic vs. clinical domains.
  - [ ] Automated LaTeX and Markdown research reporting.
- [ ] **Phase 4: Calibration & Explainability**
  - [ ] Expected Calibration Error (ECE) and Temperature Scaling across tone subgroups.
  - [ ] Grad-CAM and visual confounder localization analysis.
- [ ] **Phase 5: Mitigation & Adaptation**
  - [ ] Group-DRO and balanced reweighting.
  - [ ] Unsupervised Domain Adaptation (UDA) / DANN feature alignment.
- [ ] **Phase 6: Full-Stack Demonstration & Dashboard**
  - [ ] FastAPI backend for model serving and metrics API.
  - [ ] React frontend with Plotly visualizations.
  - [ ] MongoDB persistence for evaluation runs.

---

## 3. Immediate Next Action

Resume with Prompt 8: implement the permutation disparity test improvements, missing patient-ID handling, subgroup sparsity threshold guard, and ADR-005 documentation.
