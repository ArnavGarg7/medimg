# Model Card (template)
- **Intended use:** decision-support triage (prioritise worklist) and lesion localisation; NOT autonomous diagnosis.
- **Users / setting:** radiologist or dermatologist workstation, human always in the loop.
- **Models:** ResNet classifier, U-Net segmentation, RetinaNet detection (+ transformer summary, VAE augmentation in phase 9).
- **Training data / splits:** see Dataset Card; seed; augmentation; class-imbalance handling.
- **Operating point:** threshold chosen on VALIDATION to reach target sensitivity (default 95%); TEST used once.
- **Metrics (test, 95% CI):** AUROC, sensitivity, specificity, Dice, IoU, AP50, latency (median/p95), memory.
- **Robustness:** noise, blur, brightness/contrast, JPEG quality, resolution, device/site shift.
- **Known failure modes:** (fill from error analysis)
- **Limits:** single-source data, no prospective validation, calibration not guaranteed, not regulatory-cleared.
