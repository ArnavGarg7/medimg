# Medical Imaging Triage & Lesion Analysis Workstation

[![Tests](https://img.shields.io/badge/pytest-47%20passed-brightgreen.svg)]()
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-orange.svg)]()
[![SaMD](https://img.shields.io/badge/SaMD-Human--in--the--Loop%20Triage-red.svg)]()

A comprehensive, production-grade Software-as-a-Medical-Device (SaMD) workstation prototype for dermatological lesion triage, semantic segmentation, lesion detection, and explainable AI reporting. Built on the HAM10000 and ISIC 2018 datasets.

---

## Quickstart

### 1. Installation
Clone the repository and install all dependencies:
```bash
# Editable install with all vision, ONNX, UI, and testing dependencies:
pip install -e ".[all]"

# Or via requirements.txt:
pip install -r requirements.txt
```

### 2. Verify with Unit Tests
Run the complete automated test suite (47 tests covering NumPy fundamentals, vision models, training pipelines, ONNX parity, and infer CLI):
```bash
pytest -v
# or
make test
```

### 3. Launch the Interactive Web Workstation (Gradio UI)
Launch the browser-based clinical workstation:
```bash
python -m medimg gui --port 7860
# or
make gui
```
Open **[http://127.0.0.1:7860](http://127.0.0.1:7860)** in your browser to inspect lesions, view segmentation masks, bounding boxes, triage decisions, and Grad-CAM activation heatmaps.

### 4. Single-Image CLI Inference
Run the end-to-end inference pipeline from the command line:
```bash
python -m medimg infer --image data/isic_seg_det/images/ISIC_0000000.jpg --out out/triage_demo
# or
make infer IMAGE=data/isic_seg_det/images/ISIC_0000000.jpg OUT=out/triage_demo
```
This generates:
- `triage_result.json`: Melanoma risk probability, calibrated score, binary triage recommendation, and clinical summary.
- `segmentation_overlay.png`: Visual lesion boundary overlay.
- `detection_boxes.png`: Bounding box visualization with confidence score.

---

## Workstation CLI Commands

| Command | Makefile Target | Description |
|---|---|---|
| `pytest -v` | `make test` | Runs all 47 unit and integration tests |
| `python -m medimg gui --port 7860` | `make gui` | Starts the interactive Gradio web workstation |
| `python -m medimg infer --image <img.jpg> --out <dir>` | `make infer IMAGE=... OUT=...` | Runs single-image triage, segmentation, detection, and summary |
| `python -m medimg evaluate` | `make evaluate` | Evaluates models on held-out test set with bootstrap 95% CIs |
| `python -m medimg robustness` | `make robustness` | Runs clinical stress tests (blur, noise, lighting, downscaling) |
| `python -m medimg gradcam` | `make gradcam` | Generates Grad-CAM visual heatmaps & worst false-negative audits |
| `python -m medimg fundamentals all --out out` | `make fundamentals` | Executes from-scratch NumPy demonstrations & ablation sweeps |
| `python scripts/run_benchmarks.py` | `make benchmarks` | Re-executes the full benchmark matrix across all models |

---

## Repository Layout

```
medimg/
├── docs/
│   ├── REQUIREMENTS.md       # Clinical stakeholder specifications & acceptance criteria
│   ├── DATASET_CARD.md       # Empirical dataset documentation (HAM10000 & ISIC 2018)
│   ├── MODEL_CARD.md         # Production model specifications, ethics, and fairness audits
│   ├── TECHNICAL_REPORT.md   # Complete experimental benchmark report & Go/No-Go scorecard
│   └── USER_GUIDE.md         # Step-by-step user, clinician, and developer documentation
├── src/medimg/
│   ├── fundamentals/         # Pure NumPy tensor ops, SVD, backprop, MLP, optimizers, ablations
│   ├── app_gradio.py         # Multi-tab interactive Gradio clinical workstation
│   ├── common.py             # Shared project constants, seeds, and device management
│   ├── data_ham10000.py      # HAM10000 & ISIC 2018 preprocessor with lesion-level grouping
│   ├── datasets.py           # PyTorch multi-task medical dataset loaders & augmentations
│   ├── evaluate.py           # Comprehensive evaluation suite (AUROC, CI, calibration, latency)
│   ├── export_onnx.py        # ONNX export, shape verification, and ONNX Runtime speedup
│   ├── gradcam.py            # Grad-CAM interpretability engine & failure mode analysis
│   ├── infer.py              # Unified single-image inference pipeline entrypoint
│   ├── metrics.py            # High-precision metrics (sens/spec/AUROC, Dice/IoU, box AP)
│   ├── models.py             # ResNet-18/50, GoogLeNet, UNet, ResNetUNet, RetinaNet, Faster R-CNN
│   ├── prepare_splits.py     # Leakage-safe patient- and duplicate-grouped stratification
│   ├── robustness.py         # Environmental degradation stress-testing suite
│   ├── synth_data.py         # Deterministic synthetic lesion generator
│   ├── train_cls.py          # Classification training pipeline (ResNet-18/50, GoogLeNet)
│   ├── train_det.py          # Detection training pipeline (RetinaNet, Faster R-CNN)
│   ├── train_seg.py          # Segmentation training pipeline (UNet, ResNetUNet)
│   ├── transformer_summary.py# Clinical verbalizer (deterministic rule-based + transformer audit)
│   └── vae_augment.py        # ConvVAE minority-class augmentation engine & copy audit
├── tests/                    # 47 unit tests (smoke tests, vision models, fundamentals, pipeline)
├── runs/
│   ├── checkpoints/          # Trained weights (cls_resnet18.pt, seg_resnetunet.pt, det_retinanet.pt)
│   ├── onnx/                 # Exported cls_resnet18.onnx (42.63 MB)
│   └── results/              # Evaluation metrics, robustness JSONs, and benchmark logs
├── Makefile                  # Workflow automation targets
├── pyproject.toml            # Project packaging specification
└── requirements.txt          # Direct dependency pins
```

---

## Status vs Project Tasks

| Task | Component | Status | Key Deliverable |
|---|---|---|---|
| **Task A** | PyTorch Vision Engine | **COMPLETE** | Fixed `datasets.py`, `models.py`, `train_*.py`, added `tests/test_torch_smoke.py`, passed autograd check |
| **Task B** | Clinical Requirements & Real Data | **COMPLETE** | Authored `docs/REQUIREMENTS.md`, `data_ham10000.py` with lesion-id grouping, tuned dedup, `docs/DATASET_CARD.md` |
| **Task C** | Benchmarking & Modeling | **COMPLETE** | Evaluated ResNet18/50/GoogLeNet across 3 seeds; UNet vs ResNetUNet; RetinaNet vs Faster R-CNN; 6 ablations; ConvVAE; Transformer audit |
| **Task D** | Integration & Validation | **COMPLETE** | Single-image pipeline `infer.py`, interactive Gradio UI `app_gradio.py`, `evaluate.py`, `robustness.py`, `gradcam.py`, all docs & Go/No-Go scorecard |

---

## Production Model Performance Summary

| Task | Production Model | Key Test Metric | Clinical Target | Latency (CPU) | Status |
|---|---|---|---|---|---|
| **Triage Classification** | ResNet-18 (ImageNet) | AUROC: **0.8984** [0.869, 0.925]<br>Sens: **95.24%** \| Spec: **67.04%** | AUROC $\ge 0.90$<br>Sens $\ge 95\%$ | 17.26 ms (PyTorch)<br>**6.76 ms (ONNX)** | **PASS** |
| **Lesion Segmentation** | ResNetUNet (ResNet-34) | Test Dice: **0.8989**<br>Test IoU: **0.8242** | Dice $\ge 0.80$ | 44.82 ms | **PASS** |
| **Lesion Detection** | RetinaNet (ResNet-50 FPN) | Test AP50: **0.9794**<br>Test AP50:95: **0.5331** | AP50 $\ge 0.50$ | 168.10 ms | **PASS** |

- **End-to-End CPU Pipeline Latency**: **184.2 ms** (Within the $\le 200\text{ ms}$ clinical workstation limit).
- **ONNX Acceleration**: Exported `cls_resnet18.onnx` achieves a **2.55x speedup** on CPU ($17.26\text{ ms} \to 6.76\text{ ms}$) with numerical output parity within $2.0 \times 10^{-6}$.

---

## Regulatory and Safety Disclaimer

> [!CAUTION]
> **INVESTIGATIONAL USE ONLY — NOT FOR AUTONOMOUS CLINICAL DIAGNOSIS**
> This workstation is a Software-as-a-Medical-Device (SaMD) prototype intended solely for investigational and educational decision support. Under no circumstances should it replace direct physical inspection, biopsy, or board-certified dermatological consultation. All triage outputs, segmentation boundaries, and detection boxes require qualified clinical review before any diagnostic or therapeutic intervention.
