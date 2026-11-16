# User Guide: Medical Imaging Triage and Lesion Analysis Workstation

## 1. System Overview
The **Medical Imaging Triage and Lesion Analysis Workstation** is a clinical decision-support software suite designed to assist dermatologists, general practitioners, and tele-dermatology triage centers in analyzing dermoscopic skin lesion images.

The workstation performs four simultaneous functions on each dermoscopy image:
1. **Melanoma Triage Classification:** Computes the probability of malignant melanoma and assigns a priority status (`HIGH PRIORITY` vs `LOW PRIORITY`) based on a validation-calibrated high-sensitivity threshold ($t=0.518$).
2. **Lesion Segmentation:** Segments the lesion boundary and calculates the percentage footprint of the lesion relative to the field of view.
3. **Lesion Localization (Detection):** Detects discrete suspicious lesion foci using multi-scale anchor bounding boxes.
4. **Structured Decision Support Summary:** Generates a factually verified text report with clinical recommendations and execution latency.

> [!IMPORTANT]
> **Clinical Human-in-the-Loop Constraint:** This workstation is an AI decision-support aid. It DOES NOT provide autonomous medical diagnosis. All outputs must be evaluated and confirmed by a qualified clinician.

---

## 2. Installation & Environment Setup

### 2.1. Prerequisites
- **Python:** 3.10 – 3.13 (Verified on Python 3.13.6 Windows 64-bit).
- **GPU (Optional):** NVIDIA CUDA-compatible GPU (CUDA 11.8+). Automatic fallback to CPU is fully supported.
- **Key Dependencies:** `torch`, `torchvision`, `onnx`, `onnxruntime`, `gradio`, `pillow`, `scipy`, `pandas`, `numpy`, `matplotlib`.

### 2.2. Installation Steps
Clone the repository and install in editable mode:
```bash
git clone https://github.com/example/medimg_python.git
cd medimg_python/medimg

# Install the medimg package in editable development mode
pip install -e .

# Verify test suite passes
pytest
```
*Expected test outcome: 47 passed in ~35 seconds.*

---

## 3. Command-Line Interface (CLI) Guide

The workstation exposes unified entry points via `python -m medimg <command>`:

### 3.1. Single Image Triage & Analysis (`infer`)
Runs the full multi-task pipeline on an individual dermoscopy image and saves visual overlays, segmentation masks, and clinical reports:

```bash
python -m medimg infer --image data/isic_seg_det/images/ISIC_0000000.png --out runs/infer_demo
```

**Options:**
- `--image`: (Required) Path to input dermoscopic image (`.png`, `.jpg`, `.jpeg`).
- `--out`: Directory where output files will be written (default: `runs/infer_out`).
- `--threshold`: Triage decision operating threshold (default: `0.518`, calibrated for $\ge 95\%$ target sensitivity).
- `--size`: Internal image resolution for neural processing (default: `128`).

**Generated Outputs in Output Directory:**
- `triage_overlay.png`: Composite image containing the semi-transparent red lesion segmentation mask, bounding boxes with detection confidence, and the color-coded top priority banner.
- `segmentation_mask.png`: Binary mask ($128 \times 128$) of the segmented lesion.
- `triage_summary.txt`: Plain-text clinical decision support summary.
- `triage_report.json`: Full machine-readable JSON object containing quantitative probabilities, bounding box coordinates, and latency breakdown.

---

### 3.2. Comprehensive Clinical Evaluation Package (`evaluate`)
Evaluates the production model suite on the held-out test splits, computing AUROC, 95% bootstrap confidence intervals, sensitivity, specificity, calibration curves (ECE), Dice/IoU, and CPU inference latency:

```bash
python -m medimg evaluate
```

**Generated Artifacts:**
- Report: `runs/evaluation/comprehensive_evaluation.json`
- Plots: `out/evaluation/calibration_curve.png`, `out/evaluation/confusion_matrix.png`

---

### 3.3. Clinical Robustness Stress Testing (`robustness`)
Simulates 6 clinical imaging degradation stress tests (Gaussian noise, blur, illumination dimming, JPEG compression, downsampling, and hair artifacts) across 5 severity levels:

```bash
python -m medimg robustness
```

**Generated Artifacts:**
- Data: `runs/robustness/robustness_report.json`
- Multi-Panel Degradation Chart: `out/robustness/robustness_curves.png`

---

### 3.4. Grad-CAM Error Analysis & Subgroup Audit (`gradcam`)
Computes Grad-CAM heatmaps on test images, extracts the 20 worst false negatives and false positives, categorizes failure modes, and benchmarks demographic and anatomical parity:

```bash
python -m medimg gradcam
```

**Generated Artifacts:**
- Overlays: `out/error_analysis/fn_*.png` and `out/error_analysis/fp_*.png`
- Reports: `runs/error_analysis/worst_errors.json`, `runs/error_analysis/subgroup_analysis.json`
- Parity Chart: `out/error_analysis/subgroup_auroc.png`

---

### 3.5. Interactive Gradio Web Workstation (`gui`)
Launches the interactive local browser interface:

```bash
python -m medimg gui --port 7860
```
Open `http://127.0.0.1:7860` in any web browser.

---

## 4. Interactive Web Workstation User Guide

The Gradio web interface provides a streamlined visual workstation:

```
+-----------------------------------------------------------------------------------+
|               Medical Imaging Triage and Lesion Analysis Workstation              |
+----------------------------------------+------------------------------------------+
|  [INPUT PANEL]                         |  [TRIAGE DECISION BANNER]                |
|  Upload Dermoscopic Lesion Image       |  +-------------------------------------+ |
|  [ Drop file or click to browse ]      |  | [HIGH PRIORITY] Melanoma Risk: 91.4%| |
|                                        |  +-------------------------------------+ |
|  Triage Sensitivity Threshold Slider   |                                          |
|  [====O======================] 0.518   |  [MULTI-TASK VISUAL OVERLAY]             |
|                                        |  [ Lesion contour + Bounding Boxes ]     |
|  [ Analyze Lesion & Triage ] (Button)  |                                          |
|                                        |  [LATENCY BREAKDOWN]                     |
|  [Example Images]                      |  Total: 24.3 ms | Inf: 18.5 ms           |
|  - ISIC_0000000 (Benign nevus)         |                                          |
|  - ISIC_0000001 (Atypical lesion)      |  [CLINICAL DECISION SUPPORT SUMMARY]     |
|  - ISIC_0000003 (Malignant melanoma)   |  Factual recommendations & next steps    |
+----------------------------------------+------------------------------------------+
```

### Steps for Clinical Analysis:
1. **Upload Image:** Drag and drop a high-resolution dermoscopic photograph into the left pane.
2. **Threshold Setting:** The default threshold is set to `0.518` (val-calibrated $\ge 95\%$ target sensitivity). For higher-prevalence referral clinics, the slider can be adjusted to `0.480` for heightened sensitivity.
3. **Analyze:** Click **"Analyze Lesion & Triage"**.
4. **Review Outputs:**
   - **Triage Badge:** Immediate visual banner: Red (`HIGH PRIORITY`) or Green (`LOW PRIORITY`).
   - **Multi-Task Overlay:** Review the red mask overlay for lesion borders and yellow boxes for suspicious foci.
   - **Clinical Summary:** Read the structured report detailing lesion footprint % and recommended clinical pathway.

---

## 5. Interpreting Clinical Results & Safety Advice

### Priority Badges:
- `HIGH PRIORITY (Melanoma Suspected)`: Probability exceeds $51.8\%$. Recommendation: Fast-track urgent dermatologist consultation and dermatoscopy review.
- `LOW PRIORITY (Benign / Non-urgent)`: Probability below $51.8\%$. Recommendation: Routine clinical follow-up; monitor for changes using ABCDE criteria.

### Known Edge Cases:
- **Hypopigmented / Amelanotic Lesions:** Lesions lacking dark melanin may produce false negatives. If clinical suspicion exists despite a `LOW PRIORITY` score, always proceed with biopsy.
- **Facial Lesions:** Facial skin frequently suffers from solar elastosis and actinic lentigines, which can elevate false-positive rates (specificity drops to 35.7% on facial skin).
- **Lighting / Focus:** Images that are severely underexposed (illumination factor $< 0.6$) or blurred must be retaken before analysis.
