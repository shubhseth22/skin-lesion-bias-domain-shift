# Persistent Agent Instructions: Gemini & Claude

This file governs the operational protocols, behavioral constraints, and engineering standards for all AI assistants (including Google Gemini and Anthropic Claude) contributing to the **Bias and Domain Shift in Skin-Lesion Classification** CSE Major Research Project.

---

## 1. Core Operating Tenets & Scientific Integrity

1. **No Hallucinated Results or Fabricated Metrics**:
   - **NEVER** claim that an experiment has been conducted unless it was executed in the workspace and reproducible run logs exist.
   - **NEVER** invent dataset statistics (e.g., patient counts, class distributions, Fitzpatrick tone frequencies) or model performance metrics (e.g., accuracy, AUC, ECE, disparity scores).
   - If an evaluation has not yet been executed, explicitly mark metrics as `Pending Execution` or leave metric fields blank in logs.

2. **Data Governance & License Verification**:
   - Before downloading, pre-processing, or integrating any third-party medical or dermatological dataset (e.g., ISIC archives, Fitzpatrick17k, PAD-UFES-20), **always verify and document the dataset's license and permitted use cases** in `docs/dataset_documentation.md`.
   - Adhere strictly to non-commercial, academic, or attribution requirements.

3. **Repository Protection & Hygiene**:
   - **NEVER** commit datasets (raw or processed), binary medical images, model checkpoints (`.pt`, `.pth`, `.ckpt`, `.onnx`), API secrets, or `.env` credential files.
   - All data must reside in untracked `data/` directories, and all checkpoints must reside in untracked `checkpoints/` or external model artifact storage.

4. **Synchronous Status Tracking**:
   - Continuously update [PROJECT_STATUS.md](file:///Users/shubhseth/Projects/skin-lesion-bias/PROJECT_STATUS.md) upon completing or modifying any phase of the project.
   - Maintain clear separation between:
     - **Completed Work**: Features and docs fully verified and merged.
     - **Current Work**: Active, in-flight implementations.
     - **Pending Tasks**: Planned backlog items.
     - **Next Steps**: The immediate, ordered actionable items.

5. **Decision Records**:
   - Whenever an architectural, algorithmic, metric, or data decision is made, document it in [docs/decisions.md](file:///Users/shubhseth/Projects/skin-lesion-bias/docs/decisions.md) using the Architecture Decision Record (ADR) format.

6. **Code Standards**:
   - Write modular, PEP-8 compliant, fully typed, docstring-documented, and unit-tested Python code.
   - Keep business logic, data loaders, model architectures, and statistical evaluations strictly decoupled.

---

## 2. Research Focus & Methodological Scope

AI agents must focus exclusively on the five core research objectives:

1. **Skin-Tone Subgroup Bias Evaluation**:
   - Quantify performance disparities across skin types using both categorical Fitzpatrick scales (I–VI / Light vs. Dark groupings) and continuous/objective metrics like Individual Typology Angle (ITA).
   - Metrics to compute: Subgroup-specific AUROC, Sensitivity (TPR), Specificity (TNR), Equalized Odds disparity, and Demographic Parity difference.

2. **Cross-Domain Generalization & Shift Analysis**:
   - Evaluate model degradation under covariate shift (e.g., dermoscopic images vs. clinical camera / smartphone photos).
   - Track Out-of-Distribution (OOD) performance degradation ratios.

3. **Domain Adaptation & Bias Mitigation**:
   - Implement principled algorithmic mitigation:
     - Group-reweighted empirical risk minimization (Group-DRO, class/subgroup balance weighting).
     - Adversarial debiasing / domain adversarial neural networks (DANN).
     - Contrastive representation learning for invariant representations.

4. **Calibration & Uncertainty Estimation**:
   - Ensure probabilities reflect genuine diagnostic confidence, especially on marginalized groups.
   - Measure Expected Calibration Error (ECE), Maximum Calibration Error (MCE), and Temperature Scaling / Brier scores across subgroups and domain splits.

5. **Grad-CAM & Visual Explainability**:
   - Generate Grad-CAM heatmaps to assess whether models focus on genuine lesion pathology or confounders (e.g., skin markings, hair, ruler marks, vignette boundaries).

---

## 3. Workflow Protocol for Agents

- **Step 1: Planning & Approval**: Always state proposed file changes and architectural plans before executing major changes.
- **Step 2: Execution**: Implement modular, tested code; verify against syntax errors and type consistency.
- **Step 3: Verification**: Run tests (`pytest`) and lint checks before considering a task complete.
- **Step 4: Status Update**: Update `PROJECT_STATUS.md` and `docs/decisions.md` (if applicable) in the same turn.
- **Step 5: Transparency**: If an error, bottleneck, or data limitation is encountered, document it explicitly without masking failure modes.
