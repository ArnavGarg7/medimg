# Dataset Card: HAM10000 & ISIC 2018 Task 1

## 1. Overview & Dataset Identity
| Attribute | Specification |
|---|---|
| **Dataset Name** | HAM10000 ("Human Against Machine with 10,000 training images") with ISIC 2018 Task 1 Segmentation Masks |
| **Primary Sources** | 1. Harvard Dataverse: [doi:10.7910/DVN/DBW86T](https://doi.org/10.7910/DVN/DBW86T)<br>2. ISIC 2018 Challenge: [ISIC Archive Data](https://challenge.isic-archive.com/data/#2018) |
| **License** | **Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)** |
| **Modality** | Dermatoscopy (epiluminescence microscopy of pigmented skin lesions) |
| **Target Task** | Multi-task triage: binary melanoma triage classification, lesion segmentation, and lesion bounding box detection |

---

## 2. Licensing & Legal Terms for the User to Verify
> [!IMPORTANT]
> **Terms you must verify manually**:
> 1. **Non-Commercial Restriction (NC)**: The dataset is licensed under **CC BY-NC 4.0**. You may use it for academic coursework, research, benchmarks, and internal evaluations. Any commercial deployment, monetization, or incorporation into closed-source commercial software is prohibited without commercial licensing from the dataset authors.
> 2. **Attribution**: All reports, academic papers, and public repositories must cite:
>    - Tschandl, P., Rosendahl, C. & Kittler, H. *The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions*. Sci Data 5, 180161 (2018). doi:10.1038/sdata.2018.161.
>    - Codella, N. et al. *Skin Lesion Analysis Toward Melanoma Detection 2018: A Challenge Hosted by the International Skin Imaging Collaboration (ISIC)*. arXiv:1902.03368 (2018).
> 3. **Privacy & De-identification**: Images are fully de-identified; personal identifiers have been removed. You must not attempt to re-identify any patient.

---

## 3. What You Need to Download Manually
To execute Tasks B, C, and D on the real dataset, please download the following files and place them into `data/ham10000_raw/` (or provide the directory where you have stored them):

1. **Metadata**:
   - `HAM10000_metadata.csv` (Harvard Dataverse)
2. **Dermoscopic Images** (total 10,015 images):
   - `HAM10000_images_part_1.zip`
   - `HAM10000_images_part_2.zip` (extract into a folder named `images` or keep side-by-side)
3. **Segmentation Ground Truth Masks** (2,594 binary masks from ISIC 2018 Task 1):
   - `ISIC2018_Task1_Training_GroundTruth.zip` (extract into a folder named `masks` or `ISIC2018_Task1_Training_GroundTruth`)

---

## 4. Label Definitions & Clinical Rationale (Empirical Statistics from HAM10000_metadata.csv)
- **Total Images ($N$)**: **10,015**
- **Unique Lesions ($N_{\text{lesions}}$)**: **7,470**
- **Multi-image Lesions**: **1,956** (with up to 6 distinct images per lesion; proves necessity of lesion-level grouped splitting)
- **Binary Triage Label**:
  - `label = 1`: **Melanoma (`mel`)** — **1,113 images (11.11%)**. Invasive melanoma and melanoma in situ. Early clinical triage and prioritization directly reduces mortality and biopsy delay.
  - `label = 0`: **Non-Melanoma (`other`)** — **8,902 images (88.89%)**.
- **7-Class Granular Categories (preserved in `dx_7class`)**:
  - `nv` (Melanocytic nevi): **6,705 (66.95%)**
  - `mel` (Melanoma): **1,113 (11.11%)**
  - `bkl` (Benign keratosis-like lesions): **1,099 (10.97%)**
  - `bcc` (Basal cell carcinoma): **514 (5.13%)**
  - `akiec` (Actinic keratoses / Bowen's disease): **327 (3.27%)**
  - `vasc` (Vascular lesions): **142 (1.42%)**
  - `df` (Dermatofibroma): **115 (1.15%)**
- **Ground Truth Validation Method (`dx_type`)**:
  - `histo` (Histopathology confirmation): **5,340 (53.32%)**
  - `follow_up` (No change across follow-up): **3,704 (36.98%)**
  - `consensus` (Dermatoscopic expert consensus): **902 (9.01%)**
  - `confocal` (In-vivo confocal microscopy): **69 (0.69%)**

---

## 5. Splitting Strategy & Leakage Prevention
- **Group Unit**: **Lesion-level grouping (`patient_id = lesion_id`)**.
  - A single lesion often has multiple photographs taken at different angles, magnifications, or follow-up visits.
  - Splitting purely at the image level causes catastrophic data leakage (inflating test AUROC and Dice).
  - All images sharing the same `lesion_id` are grouped into the same partition (`train`, `val`, or `test`) via Union-Find.
- **Near-Duplicate Grouping**:
  - SHA-256 exact matching + 16x16 average perceptual hash (aHash, Hamming distance) + pixel Mean Absolute Difference (MAD) confirmation ensures near-duplicate images across differing lesion identifiers are colocated into the same split.
- **Hard Leakage Audit**:
  - Enforced by `prepare_splits.py`: asserts that no group or lesion spans more than one split partition.

---

## 6. Known Clinical Biases & Limitations
1. **Skin Tone / Demographic Representation**: Overwhelmingly collected from fair-skinned European populations (Vienna, Austria and Queensland, Australia); severely under-represents darker Fitzpatrick skin phototypes (IV–VI).
2. **Anatomical Distribution**: Higher concentration of sun-exposed areas (back, chest, limbs, face) compared to acral or mucosal surfaces.
3. **Acquisition Artifacts**:
   - Liquid immersion dermoscopy (bubbles, reflections).
   - Polarized contact vs non-contact illumination.
   - Surgical marker ink, hair, vignetting, and calibration ruler markings.
4. **No Prospective Regulatory Clearance**: Intended solely for decision-support research and workflow benchmarking.
