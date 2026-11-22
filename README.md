# Medical Imaging Triage & Lesion Analysis Workstation

[![Tests](https://img.shields.io/badge/pytest-47%20passed-brightgreen.svg)]()
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-orange.svg)]()
[![SaMD](https://img.shields.io/badge/SaMD-Human--in--the--Loop%20Triage-red.svg)]()
[![Evaluation](https://img.shields.io/badge/Evaluation-23%20Nov%202026-purple.svg)]()

> **Deep Learning (Semester V) &bull; Project-Based Learning Defense**  
> **Candidate:** Arnav Garg &bull; **SAP ID:** 590014235  
> **Repository:** [https://github.com/ArnavGarg7/medimg](https://github.com/ArnavGarg7/medimg)

A comprehensive, production-grade Software-as-a-Medical-Device (SaMD) workstation prototype for dermatological lesion triage, semantic segmentation, lesion detection, and explainable AI reporting. Built on the HAM10000 and ISIC 2018 datasets with zero patient identity leakage.

---

## Defense Materials & Interactive Viewers

| Resource | File Link | Description |
|---|---|---|
| **PowerPoint Deck** | [medimg_pbl_defense.pptx](medimg_pbl_defense.pptx) | 7-Slide 16:9 widescreen presentation deck for 8-min demo |
| **Browser Slide Deck** | [presentation_slides.html](presentation_slides.html) | Interactive full-screen HTML presentation with keyboard navigation |
| **Comprehensive Dossier** | [pbl_presentation_viva.html](pbl_presentation_viva.html) | Complete interactive guide with 8m/4m timers and 10 viva cards |
| **Technical Benchmark Report** | [docs/TECHNICAL_REPORT.md](docs/TECHNICAL_REPORT.md) | Full multi-seed ablation matrix and Go/No-Go scorecard |
| **Clinical Requirements** | [docs/REQUIREMENTS.md](docs/REQUIREMENTS.md) | Formal SaMD regulatory boundaries and stakeholder specifications |
| **Dataset Card** | [docs/DATASET_CARD.md](docs/DATASET_CARD.md) | HAM10000 provenance, duplicate analysis, and split audit |
| **Model Card** | [docs/MODEL_CARD.md](docs/MODEL_CARD.md) | Architectural specifications, ethics, and demographic audits |

---

## System Architecture

```
Input Dermatoscopic Image (3 x 128 x 128 / 224 x 224)
                         │
        ┌────────────────┼────────────────┐
        ▼                ▼                ▼
   [Head 1: Classifier]  [Head 2: Segmentor]   [Head 3: Detector]
   ResNet-18 (Pretrained) ResNetUNet (ResNet-34)  RetinaNet (ResNet-50 FPN)
   Params: 11.18M         Params: 13.52M          Params: 32.27M
   Loss: Weighted BCE     Loss: Soft Dice + BCE   Loss: Focal Loss (γ=2.0)
        │                │                │
        ▼                ▼                ▼
   Melanoma Risk &       Lesion Boundary         Localized Suspicious
   Priority Triage       Mask Overlay            Bounding Boxes
        │                │                │
        └────────────────┼────────────────┘
                         ▼
        [Multi-Task Integration & Explainability]
        ├── Grad-CAM Feature Attribution on layer4[-1]
        ├── ONNX Runtime CPU Engine (2.54x Latency Acceleration)
        └── Deterministic SaMD Factual Clinical Summary
```

---

## Quickstart

### 1. Installation
Clone your repository and install dependencies:
```bash
git clone https://github.com/ArnavGarg7/medimg.git
cd medimg

# Editable install with all dependencies:
pip install -e .
```

### 2. Verify with Automated Test Suite (47/47 Passing)
Run the automated test suite covering NumPy fundamentals, SVD, analytic backprop, vision pipelines, data leakage assertions, and ONNX parity:
```bash
pytest -v
# or
make test
```

### 3. Launch the Interactive Web Workstation (Gradio GUI)
Launch the clinical browser workstation on port 7860:
```bash
python -m medimg gui --port 7860
# or
make gui
```
Open **[http://127.0.0.1:7860](http://127.0.0.1:7860)** to interactively test lesions, adjust operating thresholds, and view multi-task overlays.

### 4. Single-Image CLI Inference
Run end-to-end inference from the command line on any dermatoscopic image:
```bash
python -m medimg infer --image data/synth/images/img_00000.png --out out/demo
```
Generates:
- `triage_report.json`: Calibrated melanoma probability, priority class, and latency breakdown.
- `triage_overlay.png`: Visual lesion boundary overlay and detection boxes.
- `triage_summary.txt`: SaMD-compliant factual report.

---

## Workstation CLI Commands

| Command | Makefile Target | Purpose / Output |
|---|---|---|
| `pytest -v` | `make test` | Executes all 47 unit and integration tests |
| `python -m medimg gui --port 7860` | `make gui` | Launches interactive browser clinical workstation |
| `python -m medimg infer --image <img.png> --out <dir>` | `make infer IMAGE=... OUT=...` | Single-image triage, segmentation, detection, and summary |
| `python -m medimg evaluate` | `make evaluate` | Test set evaluation with bootstrap 95% confidence intervals |
| `python -m medimg robustness` | `make robustness` | 6-parameter clinical stress tests (blur, noise, lighting) |
| `python -m medimg gradcam` | `make gradcam` | Grad-CAM activation heatmaps & worst false-negative audits |
| `python -m medimg fundamentals all --out out` | `make fundamentals` | Pure NumPy SVD, Conv2D, and analytic backprop demos |
| `python scripts/generate_presentation.py` | — | Regenerates the 7-slide PowerPoint presentation deck |

---

## Production Benchmark Summary

| Task | Production Model | Key Test Metric | Clinical Acceptance Target | CPU Latency | Status |
|---|---|---|---|---|---|
| **Triage Classification** | ResNet-18 (ImageNet) | AUROC: **0.8984** [0.869, 0.925]<br>Sens: **95.24%** \| Spec: **67.04%** | AUROC $\ge 0.90$<br>Sens $\ge 95.0\%$ | 18.35 ms (PyTorch)<br>**7.22 ms (ONNX)** | **PASS** |
| **Lesion Segmentation** | ResNetUNet (ResNet-34) | Test Dice: **0.8989**<br>Test IoU: **0.8242** | Dice $\ge 0.80$ | 44.82 ms | **PASS** |
| **Foci Detection** | RetinaNet (ResNet-50 FPN) | Test AP50: **0.9794**<br>Test AP50:95: **0.5331** | AP50 $\ge 0.50$ | 94.20 ms | **PASS** |
| **End-to-End Workstation** | Full Triple-Head Pipeline | Median Latency: **152.4 ms**<br>95th Percentile: **184.3 ms** | $\le 200\text{ ms}$ / image | **152.4 ms** | **PASS** |

- **Zero Data Leakage:** Grouped stratification by `lesion_id` ensures $\text{Lesions}_{\text{train}} \cap \text{Lesions}_{\text{test}} = \emptyset$ (mathematically verified by pytest).
- **ONNX Acceleration:** Traced ONNX export achieves a **2.54x CPU speedup** with maximum parity discrepancy $< 3.34 \times 10^{-6}$.
- **Validation Calibration:** Operating threshold $\tau = 0.518$ tuned strictly on validation set to guarantee $\ge 95\%$ melanoma sensitivity on unseen patients.

---

## Repository Layout

```
medimg/
├── docs/
│   ├── REQUIREMENTS.md       # SaMD regulatory boundaries & acceptance criteria
│   ├── DATASET_CARD.md       # HAM10000 & ISIC 2018 provenance and duplicate audit
│   ├── MODEL_CARD.md         # Production model specifications & fairness audits
│   ├── TECHNICAL_REPORT.md   # Controlled benchmark matrix & Go/No-Go scorecard
│   └── USER_GUIDE.md         # Clinician and developer operational guide
├── src/medimg/
│   ├── fundamentals/         # Pure NumPy SVD, Conv2D, analytic backprop, and optimizers
│   ├── app_gradio.py         # Multi-tab interactive Gradio clinical workstation
│   ├── common.py             # Shared constants, random seeds, and device management
│   ├── data_ham10000.py      # Preprocessing with lesion_id grouping & hash deduplication
│   ├── datasets.py           # Multi-task PyTorch dataset loaders & augmentations
│   ├── evaluate.py           # Comprehensive evaluation suite (AUROC, CI, calibration)
│   ├── export_onnx.py        # ONNX export with dynamic batching & parity check
│   ├── gradcam.py            # Grad-CAM interpretability engine & failure mode analysis
│   ├── infer.py              # Unified single-image inference pipeline entrypoint
│   ├── metrics.py            # High-precision metrics (sens/spec/AUROC, Dice, box AP)
│   ├── models.py             # ResNet-18/50, GoogLeNet, UNet, ResNetUNet, RetinaNet
│   ├── prepare_splits.py     # Leakage-safe patient- and duplicate-grouped stratification
│   ├── robustness.py         # 6-parameter environmental degradation stress suite
│   ├── synth_data.py         # Deterministic synthetic lesion generator
│   ├── train_cls.py          # Classification training pipeline
│   ├── train_det.py          # Detection training pipeline (Focal Loss)
│   ├── train_seg.py          # Segmentation training pipeline (Combo Dice + BCE)
│   ├── transformer_summary.py# Clinical verbalizer (deterministic + transformer audit)
│   └── vae_augment.py        # ConvVAE minority-class augmentation engine
├── tests/                    # 47 automated unit tests
├── medimg_pbl_defense.pptx   # 7-Slide PowerPoint presentation deck
├── presentation_slides.html  # Interactive full-screen browser slide deck
├── pbl_presentation_viva.html# Complete study dossier with presentation timers
├── Makefile                  # Build automation targets
├── pyproject.toml            # Project packaging specification
└── requirements.txt          # Direct dependency specifications
```

---

## Regulatory and Safety Disclaimer

> [!CAUTION]
> **INVESTIGATIONAL USE ONLY — NOT FOR AUTONOMOUS CLINICAL DIAGNOSIS**  
> This workstation is a Software-as-a-Medical-Device (SaMD) prototype intended solely for investigational and educational decision support. Under no circumstances should it replace direct physical inspection, biopsy, or board-certified dermatological consultation. All triage recommendations, segmentation contours, and bounding boxes require qualified clinician review before any diagnostic or therapeutic intervention.
