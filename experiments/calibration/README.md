# Calibration & Uncertainty Experiments

This directory houses experiments evaluating diagnostic prediction confidence, probability calibration, and uncertainty quantification across skin-tone subgroups and domain shifts.

## Objectives
1. Quantify miscalibration and overconfidence across demographic subgroups and domain transitions ([RQ4](file:///Users/shubhseth/Projects/skin-lesion-bias/docs/research_questions.md#research-question-4-rq4-calibration-and-uncertainty-quantification)).
2. Implement and evaluate post-hoc calibration strategies:
   - **Global Temperature Scaling**: Single learned temperature parameter $T > 0$ applied globally.
   - **Group-Specific Temperature Scaling**: Independent temperature parameters $T_g$ learned per Fitzpatrick group or ITA bucket to correct group-specific overconfidence.
   - **Vector & Matrix Scaling**: Multi-parameter affine transformations on logits.
3. Compute Expected Calibration Error (ECE) across 10 and 15 bins and generate reliability diagrams.

## Structure
- `config.yaml`: Calibration bin sizes, optimization parameters for temperature fitting.
- `run_calibration.py`: Post-hoc calibration fitting on validation splits.
- `evaluate_calibration.py`: Reliability diagram generation and subgroup ECE evaluation.
