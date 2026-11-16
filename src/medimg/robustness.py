"""Robustness benchmark suite:
Evaluates ResNet18 classifier on the real test split under 6 realistic clinical imaging artifacts:
1. Gaussian noise (sensor / low-light noise)
2. Gaussian blur (out-of-focus optics)
3. Contrast / brightness degradation (uneven dermatoscope lighting)
4. JPEG quality compression (telemedicine transmission artifacts)
5. Resolution reduction (low-end sensor downsampling)
6. Hair / ruler occlusion (clinical surface artifacts)
"""
import io, json, os, time
import numpy as np, pandas as pd
import torch
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter, ImageDraw

from .common import device, seed_all, save_json
from .datasets import MedDataset, collate
from .metrics import auroc, sens_spec
from .models import classifier

DATA_ROOT = "data/ham10000"
RUNS_DIR = "runs/robustness"
OUT_DIR = "out/robustness"
DEFAULT_THRESHOLD = 0.518

os.makedirs(RUNS_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

def apply_gaussian_noise(img_pil, severity):
    # severity: sigma in [0.0, 0.20]
    if severity == 0:
        return img_pil
    arr = np.asarray(img_pil, dtype=np.float32) / 255.0
    rng = np.random.default_rng(42)
    noise = rng.normal(0, severity, arr.shape)
    noisy = np.clip(arr + noise, 0.0, 1.0)
    return Image.fromarray((noisy * 255).astype(np.uint8))

def apply_blur(img_pil, severity):
    # severity: radius in [0.0, 4.0]
    if severity == 0:
        return img_pil
    return img_pil.filter(ImageFilter.GaussianBlur(radius=severity))

def apply_contrast_dim(img_pil, severity):
    # severity: factor in [1.0, 0.2]
    arr = np.asarray(img_pil, dtype=np.float32) * severity
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

def apply_jpeg_compression(img_pil, quality):
    # quality: [100, 75, 50, 25, 10]
    if quality >= 100:
        return img_pil
    buf = io.BytesIO()
    img_pil.save(buf, format="JPEG", quality=int(quality))
    buf.seek(0)
    return Image.open(buf).convert("RGB")

def apply_resolution_reduction(img_pil, scale):
    # scale: [1.0, 0.75, 0.5, 0.33, 0.25]
    if scale >= 1.0:
        return img_pil
    w, h = img_pil.size
    small = img_pil.resize((max(4, int(w * scale)), max(4, int(h * scale))), Image.BILINEAR)
    return small.resize((w, h), Image.BILINEAR)

def apply_hair_occlusion(img_pil, n_hairs):
    # n_hairs: [0, 2, 4, 8, 16]
    if n_hairs == 0:
        return img_pil
    res = img_pil.copy()
    draw = ImageDraw.Draw(res)
    rng = np.random.default_rng(42)
    w, h = res.size
    for _ in range(int(n_hairs)):
        x0, y0 = rng.uniform(0, w), rng.uniform(0, h)
        x1, y1 = x0 + rng.uniform(-40, 40), y0 + rng.uniform(-40, 40)
        draw.line([(x0, y0), (x1, y1)], fill=(20, 15, 10), width=rng.integers(1, 3))
    return res

def evaluate_perturbed(model, dataset, perturb_fn, severity_val, dev, batch_size=32):
    preds, gts = [], []
    batch_tensors = []
    for i in range(len(dataset)):
        r = dataset.df.iloc[i]
        raw_img = Image.open(os.path.join(dataset.root, r.image_path)).convert("RGB")
        pert_img = perturb_fn(raw_img, severity_val)
        resized = pert_img.resize((128, 128), Image.BILINEAR)
        x = np.asarray(resized, dtype=np.float32).transpose(2, 0, 1) / 255.0
        batch_tensors.append(torch.from_numpy(x))
        gts.append(float(r.label))

        if len(batch_tensors) == batch_size or i == len(dataset) - 1:
            x_batch = torch.stack(batch_tensors).to(dev)
            with torch.no_grad():
                logits = model(x_batch)
                probs = torch.sigmoid(logits).squeeze(-1).cpu().numpy().tolist()
                if isinstance(probs, float):
                    probs = [probs]
                preds.extend(probs)
            batch_tensors = []

    preds = np.array(preds)
    gts = np.array(gts)
    auc = auroc(gts, preds)
    sens, spec = sens_spec(gts, preds >= DEFAULT_THRESHOLD)
    return {"auroc": auc, "sensitivity": sens, "specificity": spec}

def run_robustness_suite():
    dev = device()
    seed_all(42)
    te = MedDataset(DATA_ROOT, "test", size=128, augment=False)
    model = classifier("resnet18", pretrained=True).to(dev)
    if os.path.exists("runs/checkpoints/cls_resnet18.pt"):
        model.load_state_dict(torch.load("runs/checkpoints/cls_resnet18.pt", map_location=dev, weights_only=True))
    model.eval()

    experiments = {
        "gaussian_noise": {
            "fn": apply_gaussian_noise,
            "levels": [0.0, 0.05, 0.10, 0.15, 0.20],
            "unit": "Sigma"
        },
        "gaussian_blur": {
            "fn": apply_blur,
            "levels": [0.0, 1.0, 2.0, 3.0, 4.0],
            "unit": "Radius (px)"
        },
        "illumination_dimming": {
            "fn": apply_contrast_dim,
            "levels": [1.0, 0.8, 0.6, 0.4, 0.2],
            "unit": "Intensity Factor"
        },
        "jpeg_compression": {
            "fn": apply_jpeg_compression,
            "levels": [100, 75, 50, 25, 10],
            "unit": "JPEG Quality"
        },
        "resolution_reduction": {
            "fn": apply_resolution_reduction,
            "levels": [1.0, 0.75, 0.50, 0.33, 0.25],
            "unit": "Downscale Factor"
        },
        "hair_occlusion": {
            "fn": apply_hair_occlusion,
            "levels": [0, 2, 4, 8, 16],
            "unit": "Hair Artifact Count"
        }
    }

    report = {}
    print("\n========================================================")
    print("TASK D.3: Clinical Robustness Stress Testing (Test Set)")
    print("========================================================")

    for name, exp in experiments.items():
        print(f"\nTesting Robustness to: {name}...")
        res_list = []
        for lvl in exp["levels"]:
            m = evaluate_perturbed(model, te, exp["fn"], lvl, dev)
            m["level"] = lvl
            res_list.append(m)
            print(f"  Level {lvl:4}: AUROC = {m['auroc']:.4f}, Sens = {m['sensitivity']:.4f}, Spec = {m['specificity']:.4f}")
        report[name] = {"levels": exp["levels"], "unit": exp["unit"], "results": res_list}

    save_json(report, f"{RUNS_DIR}/robustness_report.json")

    # Generate multi-panel figure
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    for idx, (name, exp) in enumerate(experiments.items()):
        ax = axes[idx]
        res = report[name]["results"]
        x_vals = [r["level"] for r in res]
        aurocs = [r["auroc"] for r in res]
        senss = [r["sensitivity"] for r in res]
        specs = [r["specificity"] for r in res]

        ax.plot(x_vals, aurocs, "o-", label="AUROC", color="#1976D2", linewidth=2)
        ax.plot(x_vals, senss, "s--", label="Sensitivity (t=0.518)", color="#D32F2F", linewidth=2)
        ax.plot(x_vals, specs, "^:", label="Specificity (t=0.518)", color="#388E3C", linewidth=2)
        ax.set_title(name.replace("_", " ").title(), fontsize=12, fontweight="bold")
        ax.set_xlabel(exp["unit"])
        ax.set_ylabel("Metric Value")
        ax.set_ylim([0.0, 1.05])
        ax.grid(True, linestyle="--", alpha=0.6)
        if idx == 0:
            ax.legend(loc="lower left", fontsize=9)

    plt.suptitle("Clinical Robustness Stress Tests: Metric Degradation vs Artifact Severity", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig_path = f"{OUT_DIR}/robustness_curves.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"\nSaved robustness curves to: {fig_path}")

    return report

if __name__ == "__main__":
    run_robustness_suite()
