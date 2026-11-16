# System Requirements Specification: Medical Imaging Triage and Lesion Analysis Workstation

## 1. Stakeholders and Clinical Workflow
- **Primary Stakeholder**: Dermatologists, tele-dermatology triage clinicians, and clinical dermatology residents.
- **Clinical Workflow Context**:
  1. **Intake Triage**: In high-volume outpatient clinics and tele-dermatology referral queues, hundreds of dermoscopic images arrive daily. Patients with suspected malignant lesions (melanoma) face diagnostic delays if reviewed on a first-in, first-out basis.
  2. **Worklist Prioritization**: The workstation analyzes incoming images to score the probability of malignancy and reorders the review queue so high-risk cases receive same-day clinical evaluation.
  3. **Lesion Delineation & Localization**: The workstation highlights bounding boxes (detection) and extracts boundary contours (segmentation) to assist the specialist in inspecting lesion margins and symmetry.
  4. **Human-in-the-Loop Constraint**: The workstation produces triage priority, lesion masks, and structured parameter summaries. It does **not** provide autonomous clinical diagnoses or prescribe treatments; final clinical decisions remain exclusively with the physician.

---

## 2. Measurable Targets & Performance Metrics

| Task | Metric | Target Threshold | Validation / Selection Protocol |
|---|---|---|---|
| **Classification (Triage)** | Sensitivity (Recall) | $\ge 95.0\%$ | Selected on **Validation split** only; tested once on Test split |
| **Classification (Triage)** | Specificity | $\ge 60.0\%$ | Measured on Test at the 95% sensitivity validation operating point |
| **Classification (Discrimination)** | AUROC with 95% CI | $\ge 0.85$ (Lower CI $\ge 0.75$) | Mann-Whitney U AUROC with 1,000 bootstrap resamples |
| **Lesion Segmentation** | Dice Similarity Coefficient | $\ge 0.80$ | Evaluated across positive lesion test masks |
| **Lesion Segmentation** | Intersection over Union (IoU) | $\ge 0.70$ | Jaccard index on held-out test split |
| **Lesion Detection** | AP@0.5 (VOC-style IoU 0.5) | $\ge 0.50$ | All-point interpolated average precision |
| **Runtime Latency** | CPU Latency (per image) | $\le 200\text{ ms}$ (median) | Evaluated over 20 runs following 3 warmups |
| **Runtime Latency** | GPU Latency (per image) | $\le 50\text{ ms}$ (median) | Evaluated over 20 runs following 3 warmups |
| **Memory Footprint** | Peak Host / RAM | $\le 2.0\text{ GB}$ | Measured during batch=1 inference pipeline |
| **Export Parity** | ONNX vs PyTorch Logits | Max absolute diff $\le 10^{-4}$ | Evaluated across CPU execution providers |

---

## 3. Data Integrity & Leakage Acceptance Criteria
1. **Patient & Lesion-Level Group Isolation**:
   - Dermatoscopic datasets frequently contain multiple crops or sequential images of the same physical lesion or patient.
   - **Acceptance Criterion**: Zero patient IDs or lesion IDs may cross split boundaries (`train`, `val`, `test`).
   - Hard audit test: `df.groupby("patient_id").split.nunique().max() == 1` and `df.groupby("group").split.nunique().max() == 1`.
2. **Duplicate & Near-Duplicate Detection**:
   - Identical and near-identical images (evaluated via SHA-256 hash, 16x16 perceptual average hash with Hamming threshold, and pixel Mean Absolute Difference) must be merged into connected component groups via Union-Find prior to split assignment.
   - **Acceptance Criterion**: All near-duplicate pairs must be assigned to the exact same partition.
3. **Class Stratification**:
   - Each partition (`train`, `val`, `test`) must maintain class balance within $\pm 15\%$ of the empirical dataset prevalence.

---

## 4. Acceptance Tests Mapped to Code & Tests

| Requirement | Verification Code / Test | Pass Condition |
|---|---|---|
| **Numeric & Tensor Fundamentals** | `tests/test_fundamentals.py` | 31/31 tests pass (Jacobi SVD diff vs LAPACK $< 10^{-12}$, gradcheck $< 10^{-7}$) |
| **Autograd Cross-Check** | `tests/test_fundamentals.py::test_gradients_vs_torch_autograd` | NumPy manual gradients match PyTorch autograd within $10^{-7}$ |
| **Metric Implementations** | `tests/test_metrics_data.py` | AUROC matches scikit-learn on tied scores; Dice/IoU/AP50 known values pass |
| **Zero-Leakage Split Pipeline** | `tests/test_metrics_data.py::test_split_pipeline_no_leakage` | `split_report.json` records `"leakage_audit": "passed"` |
| **Vision Model Smoke Suite** | `tests/test_torch_smoke.py` | U-Net shape preservation, ONNX export parity, 1-epoch pipeline execution |
| **Vision Model Architecture** | `tests/test_torch_vision.py` | ResNet, U-Net, RetinaNet mixed-batch & empty-batch execution pass |
| **Deterministic Reproducibility**| `medimg.common.run_meta` | Output JSON records git hash, python/torch version, hyperparameters, timestamp |

---

## 5. Deployment, Regulatory, and Safety Constraints
1. **Human-in-the-Loop Protocol**:
   - The workstation functions strictly as Computer-Aided Triage (CADt) / Computer-Aided Detection (CADe).
   - In accordance with FDA Software as a Medical Device (SaMD) guidance for CADt devices (21 CFR 892.2080), the output serves to prioritize workflow and direct attention, never to replace histological confirmation or biopsy.
2. **Deterministic Triage Thresholding**:
   - Triage priority is determined by a fixed operating threshold chosen strictly on validation data to achieve $\ge 95\%$ sensitivity for melanoma. The test split is never touched during threshold optimization.
3. **Factual Consistency of Automated Summaries**:
   - Any automated text generation component (e.g., summary generator) must draw strictly from computed numerical values (predicted score, measured area in pixels, detection count).
   - Hallucination check: Every numerical claim in generated text must match input pipeline metrics exactly.
4. **Out-of-Distribution & Artifact Handling**:
   - The workstation must flag low-quality images (excessive blur, severe underexposure, or excessive occlusions such as thick surgical markers or hair) for manual clinician review rather than returning silent false-negative predictions.
