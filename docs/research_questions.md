# Formal Research Questions and Hypotheses

This document formally specifies the core research questions (RQs), scientific hypotheses, evaluation metrics, and validation protocols for the **Bias and Domain Shift in Skin-Lesion Classification** project.

---

## Research Question 1 (RQ1): Skin-Tone Subgroup Disparity

> **Question**: *To what extent do state-of-the-art deep learning architectures trained on standard dermatological benchmarks exhibit diagnostic performance disparities across Fitzpatrick skin tone types (I–VI) and ITA-derived color brackets?*

- **Motivation**: Most benchmark skin-lesion repositories (e.g., ISIC archives) are predominantly sourced from fair-skinned populations in Europe, North America, and Australia. Models trained on these distributions may underperform on darker skin tones (Fitzpatrick types IV–VI).
- **Hypothesis ($H_1$)**: Deep learning models trained under standard empirical risk minimization (ERM) exhibit statistically significant degradation in AUROC and Sensitivity (True Positive Rate at 95% specificity) on darker skin tones compared to lighter skin tones.
- **Evaluation Metrics**:
  - Subgroup-specific AUROC, Sensitivity (Recall), Specificity, and Positive Predictive Value (PPV).
  - Equalized Odds Disparity: $\max_{y \in \{0, 1\}} |\text{TPR}_{G_1} - \text{TPR}_{G_2}| + |\text{FPR}_{G_1} - \text{FPR}_{G_2}|$.
  - Demographic Parity Difference: $|\hat{P}(\hat{Y}=1 | G=g_1) - \hat{P}(\hat{Y}=1 | G=g_2)|$.
- **Statistical Validation**: Non-parametric bootstrap (1,000 resamples) for 95% confidence intervals; permutation testing for inter-group metric differences.

---

## Research Question 2 (RQ2): Cross-Domain Generalization & Shift

> **Question**: *What is the degradation magnitude when models trained on high-quality dermoscopic images are evaluated on clinical camera and smartphone photography, and does this degradation disproportionately impact underrepresented subgroups?*

- **Motivation**: Real-world triage often begins with consumer smartphones or standard clinical photography, whereas models are commonly trained on dermoscopy with specialized polarized lighting and immersion fluid.
- **Hypothesis ($H_2$)**: Covariate shift between dermoscopic and clinical acquisition modalities induces a disproportionate drop in both calibration and accuracy for darker skin tones due to compounding lighting and contrast variations.
- **Evaluation Metrics**:
  - Out-of-Distribution (OOD) Transfer Gap: $\Delta_{\text{transfer}} = \text{Metric}(\mathcal{D}_{\text{in-domain}}) - \text{Metric}(\mathcal{D}_{\text{out-of-domain}})$.
  - Subgroup Performance Retention Rate: $\frac{\text{AUROC}_{\text{target}}^{(g)}}{\text{AUROC}_{\text{source}}^{(g)}} \times 100\%$.
- **Validation Protocol**: Train on dermoscopic benchmark (e.g., ISIC), evaluate zero-shot on clinical benchmark (e.g., Fitzpatrick17k, PAD-UFES-20).

---

## Research Question 3 (RQ3): Algorithmic Mitigation and Domain Adaptation

> **Question**: *Can group-regularized optimization (Group-DRO, balanced reweighting) and domain adversarial adaptation (DANN) reduce subgroup disparity and domain transfer gap without causing catastrophic degradation in overall diagnostic utility?*

- **Motivation**: Naive fairness constraints can sometimes degrade performance across all groups (the "fairness-accuracy tradeoff"). Principled mitigation must preserve overall diagnostic efficacy.
- **Hypothesis ($H_3$)**: Combining group-weighted optimization with domain-adversarial feature alignment reduces worst-case group error and cross-domain disparity with less than a 2% reduction in overall AUROC.
- **Evaluation Metrics**:
  - Min-Group AUROC (worst-case group performance).
  - Pareto Frontier Analysis: Plotting overall AUROC vs. Equalized Odds Disparity across different regularization strengths ($\lambda$).
  - Domain Discrepancy Distance ($\mathcal{H}$-divergence or Maximum Mean Discrepancy / MMD in latent space).
- **Validation Protocol**: Ablation study comparing Baseline ERM vs. Class-balanced vs. Group-reweighted vs. Group-DRO vs. DANN.

---

## Research Question 4 (RQ4): Calibration and Uncertainty Quantification

> **Question**: *How does prediction calibration vary across skin-tone subgroups and across domain transitions, and can post-hoc calibration techniques (e.g., Temperature Scaling) correct group-specific overconfidence?*

- **Motivation**: In clinical triage, a well-calibrated confidence score is critical. An overconfident false negative on a patient with darker skin tone could lead to delayed diagnosis of melanoma.
- **Hypothesis ($H_4$)**: Deep neural networks exhibit severe miscalibration on out-of-distribution domain samples and underrepresented skin-tone subgroups; subgroup-specific temperature scaling significantly reduces group-level ECE compared to global temperature scaling.
- **Evaluation Metrics**:
  - Expected Calibration Error (ECE) with $M=10$ and $M=15$ bins.
  - Maximum Calibration Error (MCE).
  - Brier Score: $\frac{1}{N} \sum_{i=1}^N (p_i - y_i)^2$.
  - Negative Log-Likelihood (NLL).
- **Validation Protocol**: Reliability diagrams generated per skin-tone subgroup before and after global vs. group-specific temperature scaling.

---

## Research Question 5 (RQ5): Explainability and Visual Confounder Fidelity

> **Question**: *Do Grad-CAM visual explanations demonstrate attention to genuine dermatological features (pigment network, irregular borders, erythema) or do they latch onto demographic correlates and acquisition artifacts (skin hair, surgical markings, calibration stickers)?*

- **Motivation**: Models can achieve seemingly high benchmark accuracy by exploiting spurious artifacts (e.g., dermoscopy rulers, dark vignetting) rather than pathology.
- **Hypothesis ($H_5$)**: Grad-CAM saliency maps on out-of-domain samples and underrepresented skin tones show higher attention dispersion and higher overlap with non-lesion artifacts than on in-domain, fair-skin samples.
- **Evaluation Metrics**:
  - Energy-based Pointing Game / Saliency IoU: Intersection over Union between binarized Grad-CAM heatmaps and ground-truth lesion segmentation masks.
  - Background Saliency Ratio (fraction of gradient energy localized outside the lesion mask).
- **Validation Protocol**: Comparative saliency evaluation on samples with and without synthetic/real artifacts across all Fitzpatrick classes.
