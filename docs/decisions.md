# Architecture & Methodology Decision Records (ADRs)

This document tracks all foundational architecture, algorithmic, metric, and data engineering decisions using the standard Architecture Decision Record (ADR) format: **Context, Decision, Status, and Consequences**.

---

## Index of Decisions

- [ADR-001: Modular Architecture Separating Core ML, Research Suite, Backend, and Frontend](#adr-001-modular-architecture-separating-core-ml-research-suite-backend-and-frontend)
- [ADR-002: Dual Skin-Tone Representation (Fitzpatrick Scale + ITA Colorimetry)](#adr-002-dual-skin-tone-representation-fitzpatrick-scale--ita-colorimetry)
- [ADR-003: Strict Scientific Integrity, Verification, and Data Governance Protocol](#adr-003-strict-scientific-integrity-verification-and-data-governance-protocol)
- [ADR-004: Dedicated Statistical Research Package (`src/research/`)](#adr-004-dedicated-statistical-research-package-srcresearch)

---

### ADR-001: Modular Architecture Separating Core ML, Research Suite, Backend, and Frontend

- **Date**: 2026-10-27
- **Status**: Accepted
- **Context**: The project spans deep learning model training, domain adaptation experiments, academic statistical reporting, an API server, and an interactive demonstration UI. Combining research scripts with API or dashboard logic creates tight coupling and inhibits reproducibility.
- **Decision**: Structurally decouple the repository into:
  - `src/`: Reusable model definitions, data pipelines, mitigation losses, and calibration tools.
  - `src/research/`: High-level statistical evaluation, domain shift transfer testing, and reporting scripts.
  - `backend/`: FastAPI REST endpoints for live inference and analytics querying.
  - `frontend/`: Independent React/Vite dashboard for visualization with Plotly.
- **Consequences**:
  - *Positive*: Independent testing, cleaner dependency management, clean reproduction of scientific benchmarks.
  - *Tradeoff*: Requires coordinated interface contracts (Pydantic models / API schemas) between `src/` and `backend/`.

---

### ADR-002: Dual Skin-Tone Representation (Fitzpatrick Scale + ITA Colorimetry)

- **Date**: 2026-10-27
- **Status**: Accepted
- **Context**: Fitzpatrick skin type is widely reported in dermatology but suffers from subjective labeling noise and inter-annotator variance. Conversely, Individual Typology Angle (ITA) provides an objective, continuous colorimetric measurement from CIE $L^*a^*b^*$ pixel values but requires unlesioned background skin segmentation.
- **Decision**: Support both skin-tone representations simultaneously:
  1. Categorical Fitzpatrick scale (Types I–VI) for clinical benchmark alignment (e.g., Fitzpatrick17k).
  2. Automated ITA extraction on background skin patches mapped to 6 standard dermatological categories (Very Light, Light, Intermediate, Tan, Brown, Dark).
- **Consequences**:
  - *Positive*: Enables comparison between subjective clinical labels and objective physical measurements; enables sensitivity analysis on annotation bias.
  - *Tradeoff*: Pipeline must include robust background skin masking to prevent lesion pixels from distorting ITA calculations.

---

### ADR-003: Strict Scientific Integrity, Verification, and Data Governance Protocol

- **Date**: 2026-10-27
- **Status**: Accepted
- **Context**: In AI research, particularly using LLM-assisted workflows, there is risk of fabricating benchmark metrics, misrepresenting experiment execution, or accidentally leaking clinical datasets/checkpoints into version control.
- **Decision**: Mandate a strict integrity protocol in `AGENTS.md`:
  - Never fabricate numbers, dataset statistics, or completed runs.
  - Keep `docs/experiment_log.md` blank until runs physically execute.
  - Enforce `.gitignore` excluding datasets, checkpoints, weights, and environment secrets.
  - Verify and record licenses prior to dataset download.
- **Consequences**:
  - *Positive*: Total academic credibility, compliance with medical data licenses, and reproducible findings.
  - *Tradeoff*: Requires disciplined manual or automated logging for all local experiment runs.

---

### ADR-004: Dedicated Statistical Research Package (`src/research/`)

- **Date**: 2026-10-27
- **Status**: Accepted
- **Context**: Investigating bias and domain shift requires rigorous statistical verification (bootstrap confidence intervals, permutation tests, subgroup disparity indices, and LaTeX/markdown reporting). Placing these in ad-hoc one-off notebooks leads to non-reproducible papers.
- **Decision**: Create a dedicated `src/research/` module containing:
  - `subgroup_analysis.py`: Granular disparity metrics across tones.
  - `domain_shift_analysis.py`: Quantitative cross-domain drop evaluations.
  - `statistical_tests.py`: Statistical significance and bootstrap intervals.
  - `reporting.py`: Automated export of publication-ready tables and plots.
- **Consequences**:
  - *Positive*: Clean command-line interface for research sweeps; automated reporting for thesis and papers.
  - *Tradeoff*: Requires unit tests to verify statistical metric correctness.
