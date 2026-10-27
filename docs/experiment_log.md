# Experiment Log & Benchmark Tracker

> **Scientific Integrity Notice**:
> In accordance with [AGENTS.md](file:///Users/shubhseth/Projects/skin-lesion-bias/AGENTS.md), this document records **only genuine, locally executed experiment runs**.
> No synthetic, simulated, or hallucinated results may be entered here. Until physical runs are completed, this ledger remains unpopulated.

---

## 1. Standard Run Logging Schema

When logging an experiment, populate the following fields:

- **Run ID**: Unique identifier formatted as `EXP-{YYYYMMDD}-{NN}` (e.g., `EXP-20261101-01`).
- **Date**: Execution timestamp.
- **Git Commit**: SHA hash of the exact code state.
- **Model Backbone**: Architecture (e.g., `resnet50`, `efficientnet_b4`, `vit_base_patch16_224`).
- **Source Dataset**: Training dataset (e.g., `ISIC-2019`).
- **Evaluation Dataset**: Test dataset (e.g., `Fitzpatrick17k`, `PAD-UFES-20`).
- **Mitigation Strategy**: Baseline ERM, Group-DRO, Class-balanced reweighting, DANN, etc.
- **Hyperparameters**: Learning rate, batch size, epochs, loss function, optimizer, seed.
- **Overall Metrics**: Macro AUROC, Macro F1, Overall ECE.
- **Subgroup Disparities**:
  - Light Skin AUROC (Fitzpatrick I–II)
  - Intermediate Skin AUROC (Fitzpatrick III–IV)
  - Dark Skin AUROC (Fitzpatrick V–VI)
  - Worst-Group AUROC
  - Equalized Odds Disparity ($\Delta_{\text{EOD}}$)
- **Artifact Link**: Path to local checkpoint and test evaluation JSON (untracked in git).

---

## 2. Completed Experiment Runs

| Run ID | Date | Commit | Model | Train Data | Test Data | Mitigation | Overall AUROC | Worst-Group AUROC | $\Delta_{\text{EOD}}$ | Overall ECE | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| *No experiments run yet* | — | — | — | — | — | — | — | — | — | — | Foundation Phase |

---

## 3. Detailed Run Notes

*(Detailed individual run summaries, hyperparameter files, and calibration curves will be appended here as runs are executed).*
