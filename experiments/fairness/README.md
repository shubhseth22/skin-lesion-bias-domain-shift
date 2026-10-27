# Subgroup Fairness & Mitigation Experiments

This directory houses experimental configurations and evaluation scripts for algorithmic fairness interventions designed to equalize diagnostic performance across Fitzpatrick skin types (I–VI) and ITA-derived color brackets.

## Objectives
1. Implement and benchmark fairness mitigation algorithms:
   - **Balanced Class & Subgroup Reweighting**: Loss penalties inversely proportional to joint group-class sample frequencies.
   - **Group Distributionally Robust Optimization (Group-DRO)**: Minimizing the worst-case subgroup risk rather than average empirical risk.
   - **Fair Contrastive Regularization**: Aligning intermediate feature representations across tone groups for identical pathological classes.
2. Evaluate trade-offs along the Pareto frontier (Overall AUROC vs. Equalized Odds Disparity $\Delta_{\text{EOD}}$) as formulated in [RQ3](file:///Users/shubhseth/Projects/skin-lesion-bias/docs/research_questions.md#research-question-3-rq3-algorithmic-mitigation-and-domain-adaptation).

## Structure
- `config.yaml`: Fairness regularization hyperparameters, group weights, and Group-DRO step sizes.
- `train_fairness.py`: Mitigation training runner.
- `evaluate_fairness.py`: Fairness auditing runner computing demographic parity, equalized odds, and worst-group AUROC.
