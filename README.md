# Bias and Domain Shift in Skin-Lesion Classification

[![Project Status: Active](https://img.shields.io/badge/Project%20Status-Foundation%20Phase-blue.svg)](file:///Users/shubhseth/Projects/skin-lesion-bias/PROJECT_STATUS.md)
[![License: Academic / Research Use](https://img.shields.io/badge/License-Academic%20Use-green.svg)](#licensing-and-ethics)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)

A Computer Science & Engineering (CSE) Major Research Project investigating algorithmic bias across demographic subgroups (skin-tone types) and performance degradation under domain shift (clinical vs. dermoscopic imaging modalities) in deep-learning-based dermatological computer vision models.

---

## 1. Research Objectives

Deep learning systems for skin lesion diagnostics frequently exhibit critical failure modes when deployed across diverse patient populations and acquisition environments. This project investigates and addresses these challenges through five core pillars:

1. **Skin-Tone Subgroup Bias Evaluation**: Assessing performance disparities across Fitzpatrick skin types (I–VI) and objective colorimetric metrics (Individual Typology Angle / ITA).
2. **Cross-Domain Generalization**: Evaluating model degradation under covariate and acquisition domain shifts (e.g., high-resolution dermoscopic images vs. unconstrained clinical/smartphone photography).
3. **Domain Adaptation & Bias Mitigation**: Implementing and benchmarking algorithmic mitigation techniques (Group-DRO, balanced re-weighting, and adversarial feature alignment) to equalize diagnostic parity without compromising overall utility.
4. **Calibration & Uncertainty Estimation**: Quantifying diagnostic reliability and overconfidence using Expected Calibration Error (ECE), Brier scores, and Temperature Scaling across subgroups.
5. **Explainability & Confounder Analysis**: Generating Grad-CAM visual explanations to verify whether models attend to genuine pathological characteristics or spurious demographic/environmental confounders (e.g., skin markings, ruler artifacts, lighting).

---

## 2. Technology Stack

- **Core & Machine Learning**: Python 3.10+, PyTorch, Torchvision, Scikit-learn, SciPy
- **Computer Vision & Image Processing**: OpenCV, Albumentations, PIL
- **Statistical Analytics & Research Suite**: NumPy, Pandas, Statsmodels
- **Backend API & Serving**: FastAPI, Uvicorn, Pydantic
- **Frontend Dashboard**: React.js, Vite, Plotly.js / react-plotly.js, Tailwind CSS
- **Database & Artifact Storage**: MongoDB (metadata, run logs, evaluation metrics)

---

## 3. Repository Architecture

```text
skin-lesion-bias/
├── AGENTS.md                           # Persistent AI agent guidelines (Gemini & Claude)
├── PROJECT_STATUS.md                   # Real-time project status and milestone board
├── README.md                           # Project documentation overview
├── configs/                            # Modular configuration hierarchy
│   ├── base_config.yaml                # Global default hyperparameters
│   ├── datasets/                       # Per-benchmark specifications (ISIC, Fitzpatrick17k, PAD-UFES)
│   └── experiments/                    # Specific ablations (Baseline ERM, Group-DRO, DANN)
├── docs/                               # Detailed research specifications
│   ├── architecture.md                 # System and ML pipeline architecture
│   ├── research_questions.md           # Formal research questions (RQ1–RQ5)
│   ├── dataset_documentation.md        # Dataset sources, licenses, and tone metadata
│   ├── experiment_log.md               # Controlled experiment run log
│   ├── decisions.md                    # Architecture Decision Records (ADRs)
│   └── team_contributions.md           # Project team contributions & responsibilities
├── experiments/                        # Controlled experiment suite
│   ├── baseline/                       # Empirical risk minimization reference runs
│   ├── fairness/                       # Subgroup mitigation (Group-DRO, balanced weights)
│   ├── domain_adaptation/              # Dermoscopy -> clinical transfer & adaptation
│   └── calibration/                    # ECE & subgroup temperature scaling benchmarks
├── src/                                # Core ML research source code
│   ├── datasets/                       # Abstract dataset contract, ITA computation, tone bucketing, splits
│   ├── models/                         # Model backbones & classification heads
│   ├── mitigation/                     # Fairness & domain adaptation algorithms
│   ├── evaluation/                     # Metric calculations (fairness, calibration)
│   ├── explainability/                 # Grad-CAM and saliency visualizers
│   ├── research/                       # Dedicated statistical & domain shift research tools
│   │   ├── subgroup_analysis.py        # Fine-grained subgroup disparity calculations
│   │   ├── domain_shift_analysis.py    # Cross-domain transfer degradation analysis
│   │   ├── statistical_tests.py        # Bootstrap CIs, permutation tests, significance
│   │   └── reporting.py                # Automated export of research tables & figures
│   └── utils/                          # Shared logging, seed reproducibility, device utilities
├── backend/                            # FastAPI model serving and analytics server
├── frontend/                           # React analytics dashboard with Plotly visualization
├── scripts/                            # Pipeline CLI drivers (e.g., prepare_splits.py)
└── tests/                              # Unit and integration test suites (7 test modules)
```

---

## 4. Setup & Installation

### Prerequisites
- Python 3.10 or higher
- Node.js 18+ (for frontend dashboard)
- MongoDB instance (local or remote URI)

### Local Environment Setup
```bash
# Clone the repository
git clone https://github.com/shubhseth22/skin-lesion-bias-domain-shift.git
cd skin-lesion-bias-domain-shift

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies (once requirements.txt is configured)
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 5. Development Principles & Governance

All contributors (human team members and AI assistants) must strictly adhere to the policies in [AGENTS.md](file:///Users/shubhseth/Projects/skin-lesion-bias/AGENTS.md):
- **Zero Result Fabrication**: Never state or log benchmark numbers or dataset statistics that have not been physically computed in this environment.
- **Dataset Licensing**: Verify open-access research licenses before downloading medical imaging datasets.
- **Data Privacy**: Never commit medical imagery, patient metadata, model checkpoint binaries, or environment secrets to version control.
- **Decision Tracking**: Document architectural and research strategy decisions in [docs/decisions.md](file:///Users/shubhseth/Projects/skin-lesion-bias/docs/decisions.md).

---

## 6. Licensing and Ethics

This project is conducted strictly for academic and research purposes as part of a CSE Major capstone project. Deep learning models developed in this project are experimental and are **not** certified medical diagnostic devices.
