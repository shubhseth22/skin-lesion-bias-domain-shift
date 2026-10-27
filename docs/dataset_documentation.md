# Dataset Documentation, Governance & Ethics

This document catalogs benchmark datasets targeted for this research, along with their licensing requirements, ethical usage guidelines, demographic attributes, and skin-tone annotation protocols.

---

## 1. Data Governance & Repository Rules

- **Zero Data in Git**: No raw image files (`.jpg`, `.png`), patient metadata CSVs, or segmentation masks may ever be committed to git. All data must reside strictly in the untracked `data/` directory (enforced by `.gitignore`).
- **License Compliance**: Every dataset used in this project must be verified for academic research permission before downloading. Commercial distribution or unauthorized re-hosting is strictly prohibited.
- **Privacy and De-identification**: All data must remain de-identified. No attempt may be made to re-identify patients or extract private health information.

---

## 2. Benchmark Datasets Profile

| Dataset | Modality | Primary Task | Demographic Metadata Available | License / Access Terms |
| :--- | :--- | :--- | :--- | :--- |
| **Fitzpatrick 17k** | Clinical Photography | 114 skin disease conditions | Fitzpatrick scale labels (I–VI), condition labels, 3 diagnostic tiers | Open academic research (CC BY-NC 4.0 / Source: Groh et al., CVPR 2021). Non-commercial only. |
| **ISIC 2019 / 2020** | Dermoscopy | 8/9 diagnostic classes (Melanoma, Nevus, BCC, etc.) | Age, sex, anatomical site (Implicit fair-skin dominance; ITA skin-tone computed post-hoc) | Creative Commons CC-BY-NC 4.0 / CC-0 depending on challenge subset. Citation required. |
| **PAD-UFES-20** | Clinical Smartphone Photography | 6 lesion types (BCC, SCC, Melanoma, ACK, NEV, SEK) | Age, sex, Fitzpatrick skin type, diagnostic biopsy confirmation | Open access research (CC BY 4.0 / Pacheco et al., Data in Brief 2020). Attribution required. |
| **DDI (Diverse Dermatology Images)** | Clinical Photography | Benign vs. Malignant classification | Fitzpatrick skin types (I–II, III–IV, V–VI), biopsy-confirmed ground truth | Academic research agreement (Daneshjou et al., Science Advances 2022). |

---

## 3. Skin-Tone Categorization Protocols

To systematically study subgroup disparities, the project utilizes two complementary skin-tone classification frameworks:

### 3.1 Categorical Fitzpatrick Scale (I–VI)
- **Type I**: Pale white skin, blue/green eyes, blond/red hair; always burns, never tans.
- **Type II**: Fair skin; burns easily, tans minimally.
- **Type III**: Medium white skin; burns moderately, tans gradually to light brown.
- **Type IV**: Olive skin; burns minimally, tans easily to moderate brown.
- **Type V**: Brown skin; rarely burns, tans profusely to dark brown.
- **Type VI**: Deeply pigmented dark brown to black skin; never burns.

*Binary/Ternary Grouping for Statistical Power*:
- Binary: Light Skin Types (Fitzpatrick I–II / III) vs. Dark Skin Types (Fitzpatrick IV–VI / V–VI).
- Ternary: Light (I–II), Intermediate (III–IV), Dark (V–VI).

### 3.2 Objective Colorimetric Metric: Individual Typology Angle (ITA)
To mitigate subjective human annotation noise, objective colorimetry is extracted from non-lesional skin patches:
1. Identify peripheral background skin region around the lesion bounding box or segmentation mask.
2. Convert RGB skin pixels to CIE $L^*a^*b^*$ color space.
3. Compute the mean luminance ($L^*$) and yellow-blue chromaticity ($b^*$).
4. Compute ITA:
   $$\text{ITA} = \arctan\left(\frac{L^* - 50}{b^*}\right) \times \frac{180}{\pi}$$
5. Map ITA values to 6 standard dermatological classes:
   - **Very Light**: $\text{ITA} > 55^\circ$
   - **Light**: $41^\circ < \text{ITA} \le 55^\circ$
   - **Intermediate**: $28^\circ < \text{ITA} \le 41^\circ$
   - **Tan**: $10^\circ < \text{ITA} \le 28^\circ$
   - **Brown**: $-30^\circ < \text{ITA} \le 10^\circ$
   - **Dark**: $\text{ITA} \le -30^\circ$

---

## 4. Dataset Directory Layout (Local Environment)

```text
data/                                   # Ignored by .git
├── raw/
│   ├── fitzpatrick17k/
│   │   ├── images/
│   │   └── fitzpatrick17k.csv
│   ├── isic2019/
│   │   ├── images/
│   │   └── metadata.csv
│   └── pad_ufes_20/
│       ├── images/
│       └── metadata.csv
└── processed/
    ├── manifests/                      # Stratified split CSVs with computed ITA
    │   ├── train_split.csv
    │   ├── val_split.csv
    │   └── test_split.csv
    └── annotations/
```

---

## 5. Verification Checklist Before Downloading Any Data
1. [ ] Check original dataset URL and official data use agreement.
2. [ ] Confirm dataset is under a research-compatible open license (e.g. CC BY, CC BY-NC).
3. [ ] Confirm local storage path is within `data/` and excluded by `.gitignore`.
4. [ ] Verify MD5 or SHA256 checksums if provided by the data curators.
5. [ ] Document the download date and version in this file.
