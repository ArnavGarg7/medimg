# Dataset Card (template - fill in when real data is chosen)
| Field | Value |
|---|---|
| Name / version | e.g. HAM10000 + ISIC 2018 Task 1 |
| Source URL, access date | |
| License (verify on the source page!) | e.g. CC BY-NC 4.0 - non-commercial: affects the "industry deployment" claim |
| Modality / body site | dermoscopy / chest X-ray / CT |
| N images, N patients/lesions | |
| Label definition and who labelled it | histopathology / consensus / expert |
| Class balance | |
| De-identification | metadata stripped? burned-in text? |
| Dedup method + counts removed | SHA-256 + 16x16 aHash + pixel check (python/medimg/prepare_splits.py) |
| Split unit | patient/lesion-level groups (never image-level) |
| Split seed, ratios, report | data/.../split_report.json |
| Known biases | skin tone, device, site, ruler/ink marks, image-border artefacts |
| Excluded uses | not for clinical diagnosis |
