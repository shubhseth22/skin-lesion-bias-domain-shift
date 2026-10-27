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
- [x] Implemented `src/utils/seed.py` for deterministic multi-framework seeding (`seed_everything`) and hardware device selection (`get_device`).
- [x] Implemented `src/datasets/base.py` abstract dataset contract with colorimetric ITA angle estimation and 6-bucket classification.
- [x] Implemented `src/datasets/splits.py` supporting multi-attribute stratification, patient-level disjoint splits (0 leakage), and cross-domain partition generation.
- [x] Created modular YAML configurations:
  - Dataset configs: `configs/datasets/isic2019.yaml`, `configs/datasets/fitzpatrick17k.yaml`, `configs/datasets/pad_ufes_20.yaml`
  - Experiment configs: `configs/experiments/baseline_erm.yaml`, `configs/experiments/fairness_group_dro.yaml`, `configs/experiments/domain_adaptation_dann.yaml`
- [x] Implemented reproducible CLI driver `scripts/prepare_splits.py` with automated integrity auditing.
- [x] Implemented complete test suite across 7 modules in `tests/`:
  - `test_datasets.py` (image dimensions, ITA colorimetry, and skin-tone mapping)
  - `test_metrics.py` (subgroup AUROC, equalized odds, demographic parity, bootstrap CIs)
  - `test_calibration.py` (ECE, MCE, reliability binning, temperature scaling smoothing)
  - `test_splits.py` (patient-level grouping, zero-leakage assertions, rare strata fallback, ratio validation, CSV roundtrip)
  - `test_models.py` (multi-backbone contracts for ResNet-50, EfficientNet-B4, ViT-Base; feature bottleneck extraction; mode toggles)
  - `test_inference.py` (batch inference, temperature scaling calibration, clinical risk thresholding, Grad-CAM shape bounds)
  - `test_utils.py` (deterministic seed repeatability and hardware device fallback)
- [x] Configured reinforced repository `.gitignore` providing universal protection against raw/processed medical datasets, model checkpoints (`*.pt`, `*.pth`, `*.bin`, `*.onnx`, `*.safetensors`, `*.ckpt`), experiment outputs (`experiments/**/runs/`, `experiments/**/checkpoints/`), virtual environments, and secrets (`*.pem`, `*.key`, `credentials.json`, `token.json`).
- [x] Created `AGENTS.md` specifying operational integrity guidelines for Gemini and Claude.
- [x] Drafted core documentation structure:
  - `README.md`
  - `docs/architecture.md`
  - `docs/research_questions.md`
  - `docs/dataset_documentation.md`
  - `docs/experiment_log.md`
  - `docs/decisions.md`
  - `docs/team_contributions.md`

### 🔄 In-Progress Work
- [ ] Finalizing repository scaffolding and initial documentation baseline.
- [ ] Reviewing open-access medical dataset licenses and download access requirements (ISIC, Fitzpatrick17k, PAD-UFES-20).

### ⏳ Pending Work
- [ ] **Phase 1: Environment & Dataset Infrastructure**
  - [ ] Set up Python environment dependencies (`requirements.txt`, `environment.yml`).
  - [ ] Implement dataset download and verification script with license compliance checks.
  - [ ] Implement skin-tone estimation pipeline using Individual Typology Angle (ITA) and Fitzpatrick scale mappings.
  - [ ] Unit tests for dataset parsers and metadata integrity.
- [ ] **Phase 2: Baseline Model Implementation & Training Pipeline**
  - [ ] Implement standardized PyTorch model backbones (e.g., EfficientNet, ResNet-50, ViT).
  - [ ] Implement cross-validation and stratified train/val/test split generation across skin tone and acquisition site.
  - [ ] Baseline empirical risk minimization (ERM) training pipeline.
- [ ] **Phase 3: Subgroup Bias & Domain Shift Research Suite**
  - [ ] Implement `src/research/subgroup_analysis.py` for granular fairness metric calculation.
  - [ ] Implement `src/research/domain_shift_analysis.py` for cross-dataset generalization evaluation.
  - [ ] Implement `src/research/statistical_tests.py` for permutation tests, bootstrap confidence intervals, and significance testing.
  - [ ] Implement `src/research/reporting.py` for automated tabular and markdown reporting.
- [ ] **Phase 4: Calibration & Explainability**
  - [ ] Implement Expected Calibration Error (ECE), Temperature Scaling, and Reliability Diagrams.
  - [ ] Implement Grad-CAM and confounder visual saliency analysis.
- [ ] **Phase 5: Mitigation & Adaptation**
  - [ ] Group-DRO and fair re-weighting strategies.
  - [ ] Unsupervised Domain Adaptation (UDA) / adversarial feature alignment.
- [ ] **Phase 6: Full-Stack Demonstration & Dashboard**
  - [ ] FastAPI backend for model serving, Grad-CAM generation, and metrics API.
  - [ ] React frontend with Plotly visualizations for subgroup disparity and domain generalization curves.
  - [ ] MongoDB persistence for model run histories and evaluation artifacts.

---

## 3. Immediate Next Steps

1. Review and approve the initial documentation suite (`README.md`, `docs/*`).
2. Generate baseline dependency files (`requirements.txt`, `environment.yml`).
3. Complete license verification for candidate datasets prior to drafting ingestion scripts.
