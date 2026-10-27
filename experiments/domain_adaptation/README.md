# Domain Adaptation Experiments

This directory contains experimental suites for cross-modality domain adaptation and out-of-distribution (OOD) generalization from dermoscopic sources to clinical and smartphone imaging targets.

## Objectives
1. Quantify baseline cross-domain transfer gap ($\Delta_{\text{transfer}}$) between dermoscopy (e.g., ISIC) and clinical photography (e.g., Fitzpatrick17k, PAD-UFES-20).
2. Implement and evaluate unsupervised domain adaptation (UDA) methods:
   - **Domain Adversarial Neural Networks (DANN)**: Learning domain-invariant feature representations using a Gradient Reversal Layer (GRL).
   - **Deep CORAL (Correlation Alignment)**: Minimizing second-order statistics (covariance) between source and target feature distributions.
   - **Adaptive Batch Normalization (AdaBN)**: Updating feature normalization statistics on unlabeled target domains.
3. Validate retention of subgroup fairness under domain shift ([RQ2 & RQ3](file:///Users/shubhseth/Projects/skin-lesion-bias/docs/research_questions.md)).

## Structure
- `config.yaml`: Source/target dataset specifications, adversarial loss weights ($\lambda_{\text{dann}}$).
- `train_domain_adaptation.py`: UDA training runner.
- `evaluate_domain_adaptation.py`: Cross-domain transfer evaluator generating transfer matrices.
