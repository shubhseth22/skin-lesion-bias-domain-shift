# CSE Major Project Team Contributions & Milestone Matrix

**Project Title**: Bias and Domain Shift in Skin-Lesion Classification  
**Academic Program**: Computer Science & Engineering (CSE) Major Capstone Project  
**Academic Year**: 2026–2027  

---

## 1. Team Roster and Research Responsibilities

| Team Member | Role / Primary Domain | Key Responsibilities | Module Ownership |
| :--- | :--- | :--- | :--- |
| **Shubh Seth** (Lead) | Lead Researcher & ML Architect | Architecture design, subgroup bias evaluation, domain adaptation modeling, research reporting | `src/models/`, `src/mitigation/`, `src/research/` |
| *Team Member 2* | Data & Calibration Engineer | Dataset ingestion, ITA estimation pipeline, calibration (ECE) & uncertainty metrics | `src/datasets/`, `src/evaluation/calibration.py` |
| *Team Member 3* | Explainability & Backend Engineer | Grad-CAM interpretability pipeline, FastAPI model serving, REST routes | `src/explainability/`, `backend/` |
| *Team Member 4* | Frontend & Visualization Engineer | React dashboard, Plotly interactive analytics, MongoDB integration | `frontend/`, `backend/app/db/` |

*(Note: Roles and member names can be adjusted to reflect your project team lineup).*

---

## 2. Milestone Deliverables & Timeline

| Phase | Milestone Description | Target Window | Status | Primary Output |
| :--- | :--- | :--- | :--- | :--- |
| **Phase 0** | **Repository Foundation & Governance** | Week 1–2 | ✅ Completed | Folder layout, documentation suite, `AGENTS.md`, ADRs |
| **Phase 1** | **Dataset Pipeline & Skin-Tone Annotation** | Week 3–4 | ⏳ Pending | Verified dataset ingestion, ITA calculator, train/val/test splits |
| **Phase 2** | **Baseline Modeling & Subgroup Evaluation** | Week 5–7 | ⏳ Pending | ResNet/EfficientNet/ViT baselines, subgroup disparity metrics |
| **Phase 3** | **Domain Shift Analysis & Adaptation** | Week 8–10 | ⏳ Pending | Dermoscopy $\to$ Clinical evaluation, Group-DRO, DANN mitigation |
| **Phase 4** | **Calibration & Grad-CAM Explainability** | Week 11–12 | ⏳ Pending | ECE temperature scaling, artifact localization analysis |
| **Phase 5** | **FastAPI & React Plotly Dashboard** | Week 13–14 | ⏳ Pending | Interactive live triage demonstration & evaluation portal |
| **Phase 6** | **Thesis Compilation & Final Presentation** | Week 15–16 | ⏳ Pending | Final major project report, code freeze, presentation slides |

---

## 3. Weekly Contribution Log

| Week / Date Range | Member | Activities Completed | PR / Commit Reference |
| :--- | :--- | :--- | :--- |
| **2026-10-27** | Shubh Seth | Repository initialization, architecture definition, documentation suite setup | Initial commit |
