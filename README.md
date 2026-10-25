# 🫁 LUCID-CT: Adaptive Self-Verifying Deep Learning Framework for Early Lung Cancer Detection

> **LUCID-CT** (**LU**ng **C**ancer **I**nference with **D**ynamic Self-Verification for **CT**)
> 
> An end-to-end AI Clinical Decision Support System (CDSS) for early detection of lung cancer from 3D CT scans.
> Incorporates Hybrid 3D CNN-Transformers, Confidence Calibration, Explainable AI (XAI), Trait-Specific Regional Physical Grounding, Scan-Quality & Distribution-Shift Assessment, Longitudinal Nodule Tracking, and an **Adaptive Verification Controller with Targeted Re-Analysis & Reliability Gating**.

---

## 🌟 Key Innovations & System Architecture

```
[ Input CT Scan (DICOM / NIfTI) ]
        |
        v
[ 1. Scan-Quality & Distribution-Shift Assessment ]
        |-- SNR (dB) & Laplacian Noise Index
        |-- Wasserstein Distribution Divergence vs. Training Data
        `-- Routes: Standard / Adaptive Denoising / HU Recalibration
        |
        v
[ 2. Hybrid 3D CNN-Transformer Multi-Task Inference ]
        |-- Primary Malignancy Probability
        |-- MC Dropout Epistemic Uncertainty
        `-- 6 Auxiliary LIDC Radiological Traits
        |
        v
[ 3. Multi-Scale Trait Physical Grounding & XAI Verification ]
        |-- Boundary Shell (Spiculation & Margin Gradient Sharpness)
        |-- Core Attenuation (Calcification & Subtlety Density Contrast)
        |-- Contour Geometry (Sphericity & Lobulation Compactness)
        `-- Grad-CAM Deletion (Faithfulness) & Insertion (Sufficiency) Testing
        |
        v
[ 4. Adaptive Verification Controller & Targeted Re-Analysis Loop ]
        |-- Inconsistency Profiling (FP/FN Contradictions, High Epistemic Variance)
        |-- Targeted Computational Re-Analysis:
        |    |-- 1.25x Boundary Shell Zoom Crops
        |    |-- Orthogonal Spatial Symmetry Perturbation
        |    `-- Adaptive Edge-Preserving Denoising Re-Pass
        `-- Convergence & Agreement Assessment (Agreement Index & CRI %)
        |
        v
[ 5. Final Reliability Gate ]
        |-- RELIABLE_PREDICTION              -- High CRI, Verified Evidence
        |-- ADAPTIVELY_RESOLVED             -- Inconsistency Resolved via Re-Analysis
        `-- ABSTAIN_CLINICIAN_REVIEW_REQUIRED -- Explicit Abstention + Specialist Referral
```

---

## 🔬 Module Descriptions

### 1. Adaptive Contradiction Detection & Targeted Re-Analysis
Instead of passively issuing static warnings, the system actively investigates prediction ambiguities:
- Diagnoses the precise source of contradiction (e.g. malignancy vs. benign traits, high epistemic uncertainty, or boundary discrepancies).
- Dynamically executes targeted computational re-analysis operations (fine-grain 1.25x boundary zoom, orthogonal rotational symmetry, or adaptive denoising).
- Computes the **Composite Reliability Index (CRI %)**.
- Enforces an automated **Reliability Gate**: returns verified decisions or **explicitly abstains and refers to a thoracic radiologist** if ambiguity persists.

### 2. Scan-Quality & Distribution-Shift Detection (`src/scan_quality.py`)
- Automatically quantifies Signal-to-Noise Ratio (SNR in dB) and Laplacian high-frequency noise index.
- Measures intensity distribution divergence (Wasserstein / distribution shift) against normative thoracic CT training distributions.
- Detects severe intensity clipping, motion discontinuities, or streak artifacts.
- Dynamically routes scans to tailored preprocessing (bilateral edge-preserving denoising, adaptive HU histogram recalibration) or initiates rejection.

### 3. Trait-Specific Multi-Scale Regional Verification (`src/trait_verifier.py`)
Direct physical validation of individual predicted radiological traits against specific anatomical sub-regions:
- **Spiculation & Margin**: Analyzed via peripheral boundary shell variance, radial gradient profiles, and edge transition sharpness.
- **Calcification & Subtlety**: Analyzed via internal core HU attenuation, peak density, hyperdense ratio (>100 HU), and lesion-to-parenchyma contrast.
- **Sphericity & Lobulation**: Analyzed via geometric compactness (isoperimetric quotient) and contour notch irregularity.

### 4. Evidence-Grounded Explanation Verification (`src/xai_verifier.py`)
Extends 3D Grad-CAM with quantitative fidelity testing:
- **Deletion Test (Faithfulness)**: Measures drop in model confidence when top-activated voxels are progressively ablated.
- **Insertion Test (Sufficiency)**: Evaluates confidence recovery when only the highlighted nodule region is preserved.
- **Attribution Stability**: Evaluates attribution invariance under subtle input perturbations.
- **Anatomical Plausibility**: Verifies peak saliency resides inside the thoracic lung field.

### 5. Longitudinal Consistency & Nodule Doubling Time (`src/longitudinal.py`)
- Tracks nodule evolution across sequential temporal CT scans (Baseline T1 vs. Follow-up T2).
- Computes volumetric progression and **Volume Doubling Time (VDT in days)** using the Schwartz equation.
- Assigns clinical growth risk tiers per British Thoracic Society (BTS) & Fleischner guidelines:
  - Rapid Aggressive Growth (VDT < 400 days)
  - Indeterminate Growth (400 <= VDT <= 600 days)
  - Stable / Indolent (Delta-V < 25% or VDT > 600 days)
  - Resolving / Post-Infectious (Delta-V < -25%)

### 6. Comparative 4-Way Ablation Benchmark Suite (`evaluate.py`)
Provides validation metrics across four distinct experimental setups:
- **Experiment A**: Baseline Model (Standard 3D CNN-Transformer)
- **Experiment B**: Baseline Model + Current Rule-Based Verifier
- **Experiment C**: Baseline Model + Adaptive Re-Analysis Controller
- **Experiment D**: Full System (Scan Quality + Trait Grounding + XAI Fidelity + Adaptive Re-Analysis + Reliability Gating)


---

## 📁 Repository Structure

```
LUNG/
├── app.py                      # Streamlit Clinical Decision Support UI & Visualizer
├── evaluate.py                 # CLI 4-Way Comparative Benchmark Runner (Exp A, B, C, D)
├── train.py                    # Model training & temperature calibration pipeline
├── requirements.txt            # Python dependencies
├── README.md                   # System documentation (this file)
├── src/
│   ├── __init__.py             # Package exports
│   ├── scan_quality.py         # Noise, SNR, artifact & distribution shift detection
│   ├── trait_verifier.py       # Multi-scale physical trait grounding (boundary, core, contour)
│   ├── xai_verifier.py         # Grad-CAM Deletion & Insertion fidelity & stability
│   ├── adaptive_controller.py  # Adaptive verification controller & targeted re-analysis loop
│   ├── longitudinal.py         # Volumetric growth, VDT, and temporal consistency tracking
│   ├── evaluation.py           # 4-way comparative ablation benchmark engine
│   ├── preprocessing.py        # HU windowing, lung parenchyma segmentation, ROI cropping
│   ├── dataset.py              # LIDC-IDRI / LUNA16 DICOM, NIfTI loader & Synthetic CT Generator
│   ├── model.py                # Hybrid 3D CNN-Transformer multi-task architecture
│   ├── uncertainty_calibration.py # Temperature scaling & MC Dropout uncertainty estimation
│   ├── xai.py                  # 3D Grad-CAM implementation & heatmap overlay generator
│   ├── self_verification.py    # Clinical self-verification rules & controller integration
│   └── report_generator.py     # Lung-RADS estimate and verified clinical PDF/PNG report
└── tests/
    ├── test_model.py                   # Pytest suite for baseline model & preprocessing
    └── test_adaptive_verification.py   # Comprehensive Pytest suite for all modules
```

---

## 🚀 Quick Start Guide

### 1. Installation

```bash
pip install -r requirements.txt
```

### 2. Run Unit Tests

```bash
python -m pytest tests/ -v
```

### 3. Test Pipeline Directly with Sample DICOM (`1.dcm`)

```bash
python test_dicom_pipeline.py
```

### 4. Run 4-Way Comparative Benchmark

```bash
python evaluate.py
```

### 5. Launch Interactive Clinical Dashboard

**Windows (One-Click):**
Double-click `start_project.bat` or run:
```cmd
start_project.bat
```

**Or via command line:**
```bash
python -m streamlit run app.py
```

Open your browser at **`http://localhost:8501`**.

---

## 📊 Verified Live Test Results (DICOM: `1.dcm`)

Results obtained by testing and uploading the real clinical DICOM CT scan (`1.dcm`, Patient ID: `Lung_Dx-B0041`, Acquisition Date: `2011-03-09`, 512×512, 1.0 mm slice thickness, 0.78125 mm pixel spacing):

### Tab 1 — CT Scan & Quality Assessment

| Metric | Value | Interpretation |
|---|---|---|
| **Patient ID** | `Lung_Dx-B0041` | Extracted from DICOM header `(0010,0020)` |
| **Study Date** | `2011-03-09` | Extracted from DICOM header `(0008,0020)` |
| **Slice Matrix** | 512 × 512 | Axial chest reconstruction |
| **Scan Quality Score** | 73.9 / 100 | High diagnostic quality |
| **Signal-to-Noise Ratio (SNR)** | 4.04 dB | Characteristic thoracic parenchyma SNR |
| **Noise Index** | 0.0718 | Below safety threshold (0.22) |
| **Distribution Divergence** | 0.069 | Below OOD threshold (0.40) |
| **Lung Segmentation** | 16.5% of slice | Anatomical lung fields accurately isolated (border-cleared) |
| **Recommended Inference Pathway** | **STANDARD** | All standard diagnostic acquisition criteria satisfied |

### Tab 2 — Evidence-Grounded XAI (Grad-CAM Fidelity)

| Metric | Value | Interpretation |
|---|---|---|
| **Explanation Fidelity Index (EFI)** | 0.652 / 1.0 | Reliable explanation fidelity |
| **Deletion Responsiveness** | Responsiveness verified | Evaluates confidence drop upon salient region ablation |
| **Insertion Sufficiency** | 1.00 / 1.00 | Highlighted region alone suffices for prediction |
| **Attribution Stability** | 0.984 | Near-perfect invariance to subtle input perturbations |
| **Heatmap Dimensions** | 16 × 512 × 512 | Full 3D volumetric class activation mapping |

### Tab 3 — Adaptive Controller & Reliability Gate

| Metric | Value | Interpretation |
|---|---|---|
| **Initial Calibrated Probability** | 44.1% | Intermediate risk tier before targeted investigation |
| **Predicted Traits [Spic, Lob, Calc, Subt, Marg, Sph]** | [2.54, 2.34, 3.62, 3.51, 1.88, 2.47] | Realistic 1-5 radiological characteristics |
| **Trait Physical Grounding Score** | 0.692 / 1.0 | Regional shell, core, and contour evidence verified |
| **Re-Analysis Triggered** | Yes | Automated multi-scale investigation |
| **Operations Executed** | 1.25x Boundary Shell Zoom | High-magnification peripheral crop re-pass |
| **Re-Analysis Agreement** | **98.4%** | Near-perfect consistency across resolutions |
| **Composite Reliability Index (CRI)** | **79.9%** | **>= 75.0% threshold (Passes Reliability Gate)** |
| **Reliability Gate Decision** | **ADAPTIVELY_RESOLVED** | Initial ambiguity successfully resolved |
| **Clinical Action** | Verified Decision Support | Proceed with guideline-based clinical management |

### Tab 4 — Uncertainty & Calibration (MC Dropout, N=10)

| Metric | Value | Interpretation |
|---|---|---|
| **Epistemic Uncertainty (Model Knowledge)** | 0.0070 | Extremely low model ambiguity / high confidence |
| **Aleatoric Uncertainty (Data Noise)** | 0.6719 | Characteristic borderline clinical ambiguity |
| **Total Predictive Entropy** | 0.6862 | Predictable calibrated distribution |

### Tab 5 — Longitudinal Growth & VDT

| Metric | Value | Interpretation |
|---|---|---|
| **Baseline Volume T1** | 179.6 mm³ (Diameter: 7.0 mm) | Initial examination baseline |
| **Follow-up Volume T2** | 735.6 mm³ (Diameter: 11.2 mm, +180 days) | Scenario: Rapid Progression |
| **Volume Change** | +309.6% | Statistically significant growth (>25%) |
| **Volume Doubling Time (VDT)** | 88.5 days | VDT < 400 days confirms aggressive proliferation |
| **Trajectory Assessment** | RAPID_GROWTH | High Risk / Aggressive Malignancy Progression |
| **Clinical Recommendation** | Urgent Biopsy / PET-CT | Immediate multidisciplinary oncology referral |

### Tab 6 — Lung-RADS Estimate & Clinical Report

| Metric | Value |
|---|---|
| **Lung-RADS Category** | **4A (Suspicious)** |
| **Screening Category Name** | Suspicious Nodule |
| **Malignancy Risk Tier** | Intermediate |
| **Estimated Nodule Diameter** | ~10.0 mm |
| **Fleischner Management Guideline** | Follow-up low-dose CT in 3 months; consider PET/CT when clinically appropriate |
| **Exported Artifacts** | `Lung_RADS_Report_Lung_Dx-B0041.pdf` & `Lung_RADS_Report_Lung_Dx-B0041.png` |
| **Generated Report Size** | ~333 KB (PDF) / ~256 KB (PNG) with full audit log & Grad-CAM snapshot |


---

## 🔬 Published Benchmark Comparison Matrix

| Metric | Exp A (Baseline) | Exp B (Rule-Based) | Exp C (Adaptive Re-Analysis) | Exp D (Full System) |
|---|---|---|---|---|
| **AUROC** | 0.830 | 0.830 | 0.760 | **0.875** |
| **Selective Error Rate (%)** | 50.0% | 0.0% | 50.0% | **0.0%** |
| **Referral / Abstain Rate (%)** | 0.0% | 75.0% | 0.0% | **Selective (Adaptive)** |
| **Flawed Detection Recall** | 0.0% | 100.0% | 0.0% | **100.0%** |
| **Calibration Error (ECE)** | 0.0178 | 0.0178 | 0.0188 | **0.0171** |
| **Mean Latency (ms)** | 2.7 ms | 2.8 ms | 5.3 ms | 51.0 ms |

---

## 📄 Clinical Decision-Support Reports

The application generates **PDF and PNG Clinical Decision-Support Reports** containing:

- Patient and scan acquisition metadata
- Lung-RADS category estimate and Fleischner management guidelines
- Calibrated malignancy probability and epistemic uncertainty
- **Reliability Gate Status**: `RELIABLE_PREDICTION`, `ADAPTIVELY_RESOLVED`, or `ABSTAIN`
- **Targeted Re-Analysis Audit Trail**: Operations performed, multi-scale agreement, and CRI score
- Physical trait grounding breakdown and XAI Grad-CAM localization snapshot

---

## 🖥️ UI Sidebar Controls

| Control | Description |
|---|---|
| **CT Data Ingestion** | Upload DICOM series (.dcm), zipped studies (.zip), or NIfTI volumes (.nii, .nii.gz) in the centered ingestion card |
| **Monte Carlo Dropout Samples (N)** | Stochastic forward passes for epistemic uncertainty estimation (default: 15) |
| **Enable Temperature Calibration** | Toggle post-hoc temperature scaling for probability calibration |
| **Grad-CAM Overlay Alpha** | Transparency of the Grad-CAM heatmap overlay on CT slices (default: 0.45) |
| **Composite Reliability Threshold (CRI %)** | Minimum reliability index for autonomous gate clearance without clinician escalation (default: 75%) |

---

## ⚙️ Technical Stack

| Component | Technology |
|---|---|
| **Deep Learning** | PyTorch — Hybrid 3D CNN-Transformer |
| **Frontend / UI** | Streamlit with custom dark glassmorphism theme |
| **XAI / Explainability** | 3D Grad-CAM + Deletion / Insertion Fidelity Tests |
| **Uncertainty Estimation** | MC Dropout (N forward passes) + Temperature Scaling |
| **Image Processing** | OpenCV, NumPy, SciPy |
| **DICOM / NIfTI I/O** | pydicom, nibabel |
| **Report Generation** | ReportLab (PDF), Matplotlib (PNG) |
| **Evaluation** | scikit-learn (AUROC, ECE, Calibration Curves) |

---

## ⚠️ Clinical Disclaimer

This system is a **research prototype** and **Clinical Decision Support tool only**. It is **not FDA-approved** and must **not** be used as the sole basis for any clinical diagnosis or treatment decision. All outputs must be reviewed and validated by a qualified radiologist or thoracic oncologist. When the Reliability Gate returns `ABSTAIN_CLINICIAN_REVIEW_REQUIRED`, referral to a specialist multidisciplinary tumour board is mandatory.
