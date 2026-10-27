# Baseline Experiments (Empirical Risk Minimization)

This directory contains experimental runners, configuration overrides, and benchmark scripts for baseline dermatological classification models trained under standard Empirical Risk Minimization (ERM) without active fairness or domain mitigation.

## Objectives
1. Train reference models (ResNet-50, EfficientNet-B4, Vision Transformer) on primary benchmark datasets (e.g., ISIC 2019/2020).
2. Establish in-distribution baseline metrics (AUROC, Sensitivity, Specificity, F1).
3. Provide the non-mitigated reference point for:
   - Skin-tone subgroup disparity evaluation ([RQ1](file:///Users/shubhseth/Projects/skin-lesion-bias/docs/research_questions.md#research-question-1-rq1-skin-tone-subgroup-disparity))
   - Cross-domain transfer degradation ([RQ2](file:///Users/shubhseth/Projects/skin-lesion-bias/docs/research_questions.md#research-question-2-rq2-cross-domain-generalization--shift))
   - Baseline Expected Calibration Error ([RQ4](file:///Users/shubhseth/Projects/skin-lesion-bias/docs/research_questions.md#research-question-4-rq4-calibration-and-uncertainty-quantification))

## Structure
- `config.yaml`: Baseline experiment hyperparameters and dataset splits.
- `train_baseline.py`: Runner executing ERM training and baseline checkpointing (to untracked storage).
- `evaluate_baseline.py`: Evaluator computing baseline subgroup and domain transfer metrics.
