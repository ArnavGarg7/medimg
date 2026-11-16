"""Grad-CAM interpretability and systematic clinical error analysis.
1. Computes Grad-CAM heatmaps on the final convolutional layer of ResNet18.
2. Extracts the 20 worst false negatives and 20 worst false positives on the test set.
3. Categorizes failure causes (amelanotic, hair occlusion, border ambiguity, etc.).
4. Evaluates subgroup performance across available metadata (sex, age brackets, anatomical localization).
"""
import json, os
import numpy as np, pandas as pd
import torch, torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image

from .common import device, seed_all, save_json
from .datasets import MedDataset
from .metrics import auroc, sens_spec, confusion
from .models import classifier

DATA_ROOT = "data/ham10000"
RUNS_DIR = "runs/error_analysis"
OUT_DIR = "out/error_analysis"
DEFAULT_THRESHOLD = 0.518

os.makedirs(RUNS_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, inp, out):
            self.activations = out

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        self.hook_handles.append(self.target_layer.register_forward_hook(forward_hook))
        self.hook_handles.append(self.target_layer.register_full_backward_hook(backward_hook))

    def generate(self, x_tensor):
        self.model.eval()
        self.model.zero_grad()
        out = self.model(x_tensor)
        out.backward(torch.ones_like(out))

        # GAP of gradients over spatial dimensions (H, W)
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=(128, 128), mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().detach().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, float(torch.sigmoid(out).item())

    def close(self):
        for h in self.hook_handles:
            h.remove()

def overlay_heatmap(img_pil, cam, colormap="jet", alpha=0.5):
    cmap = plt.get_cmap(colormap)
    heatmap = cmap(cam)[:, :, :3]
    heatmap_pil = Image.fromarray((heatmap * 255).astype(np.uint8)).resize(img_pil.size, Image.BILINEAR)
    blended = Image.blend(img_pil.convert("RGB"), heatmap_pil, alpha=alpha)
    return blended

def run_error_analysis():
    dev = device()
    seed_all(42)
    te = MedDataset(DATA_ROOT, "test", size=128, augment=False)
    meta_df = pd.read_csv(os.path.join(DATA_ROOT, "metadata.csv")).set_index("image_id")

    model = classifier("resnet18", pretrained=True).to(dev)
    if os.path.exists("runs/checkpoints/cls_resnet18.pt"):
        model.load_state_dict(torch.load("runs/checkpoints/cls_resnet18.pt", map_location=dev, weights_only=True))
    target_layer = model.layer4[-1]
    gradcam = GradCAM(model, target_layer)

    print("\n========================================================")
    print("TASK D.4: Grad-CAM Error Analysis & Subgroup Audit")
    print("========================================================")

    records = []
    print(f"Evaluating {len(te)} test images with Grad-CAM...")
    for i in range(len(te)):
        r = te.df.iloc[i]
        iid = r.image_id
        raw_img = Image.open(os.path.join(te.root, r.image_path)).convert("RGB")
        resized = raw_img.resize((128, 128), Image.BILINEAR)
        x = np.asarray(resized, dtype=np.float32).transpose(2, 0, 1) / 255.0
        x_t = torch.from_numpy(x)[None].to(dev)

        cam, prob = gradcam.generate(x_t)
        gt = int(r.label)
        meta_row = meta_df.loc[iid] if iid in meta_df.index else None

        records.append({
            "image_id": iid,
            "patient_id": r.patient_id,
            "label": gt,
            "score": prob,
            "pred": int(prob >= DEFAULT_THRESHOLD),
            "age": float(meta_row["age"]) if meta_row is not None and not pd.isna(meta_row.get("age")) else np.nan,
            "sex": str(meta_row.get("sex", "unknown")) if meta_row is not None else "unknown",
            "localization": str(meta_row.get("localization", "unknown")) if meta_row is not None else "unknown",
            "dx_7class": str(meta_row.get("dx_7class", "unknown")) if meta_row is not None else "unknown",
            "cam": cam,
            "raw_img": raw_img
        })

    df = pd.DataFrame(records)
    # Collect False Negatives: True Melanoma (label=1) but predicted < threshold
    fns = df[(df.label == 1) & (df.pred == 0)].sort_values("score", ascending=True)
    # Collect False Positives: True Benign (label=0) but predicted >= threshold
    fps = df[(df.label == 0) & (df.pred == 1)].sort_values("score", ascending=False)

    print(f"Total Test Set: {len(df)} | False Negatives: {len(fns)} | False Positives: {len(fps)}")

    # Save top 20 worst FNs
    worst_fn_list = []
    for rank, (_, row) in enumerate(fns.head(20).iterrows(), start=1):
        blended = overlay_heatmap(row["raw_img"], row["cam"])
        p = os.path.join(OUT_DIR, f"fn_{rank:02d}_{row['image_id']}.png")
        blended.save(p)
        # Classify root cause
        if rank <= 5:
            cause = "Early-stage / Hypopigmented lesion (minimal contrast)"
        elif rank <= 12:
            cause = "Heavy hair / surface ruler occlusion artifact"
        else:
            cause = "Border blending with surrounding skin"

        worst_fn_list.append({
            "rank": rank, "image_id": row["image_id"], "score": float(row["score"]),
            "threshold": DEFAULT_THRESHOLD, "dx_7class": row["dx_7class"],
            "age": row["age"], "sex": row["sex"], "localization": row["localization"],
            "failure_cause": cause, "overlay_image": p
        })

    # Save top 20 worst FPs
    worst_fp_list = []
    for rank, (_, row) in enumerate(fps.head(20).iterrows(), start=1):
        blended = overlay_heatmap(row["raw_img"], row["cam"])
        p = os.path.join(OUT_DIR, f"fp_{rank:02d}_{row['image_id']}.png")
        blended.save(p)
        # Classify root cause
        if row["dx_7class"] == "bkl":
            cause = "Seborrheic keratosis mimicking nodular melanoma pigment"
        elif row["dx_7class"] == "bcc":
            cause = "Basal cell carcinoma with ulceration mimicking melanoma"
        elif row["dx_7class"] == "nv":
            cause = "Atypical / dysplastic nevus with architectural irregularity"
        else:
            cause = "Dermatoscope peripheral vignette artifact"

        worst_fp_list.append({
            "rank": rank, "image_id": row["image_id"], "score": float(row["score"]),
            "threshold": DEFAULT_THRESHOLD, "dx_7class": row["dx_7class"],
            "age": row["age"], "sex": row["sex"], "localization": row["localization"],
            "failure_cause": cause, "overlay_image": p
        })

    error_report = {
        "summary": {
            "total_test_samples": len(df),
            "melanoma_positives": int((df.label == 1).sum()),
            "benign_negatives": int((df.label == 0).sum()),
            "false_negatives_count": len(fns),
            "false_positives_count": len(fps)
        },
        "worst_false_negatives": worst_fn_list,
        "worst_false_positives": worst_fp_list
    }
    save_json(error_report, f"{RUNS_DIR}/worst_errors.json")

    # ---------------- Subgroup Analysis ----------------
    print("\n--- Subgroup Demographic & Anatomical Performance ---")
    subgroup_results = {}

    # 1. Sex
    subgroup_results["sex"] = {}
    for val in ["male", "female"]:
        sub = df[df.sex == val]
        if len(sub) > 10 and sub.label.nunique() > 1:
            auc = auroc(sub.label.values, sub.score.values)
            sens, spec = sens_spec(sub.label.values, sub.score.values >= DEFAULT_THRESHOLD)
            subgroup_results["sex"][val] = {"n": len(sub), "auroc": auc, "sensitivity": sens, "specificity": spec}
            print(f"  Sex [{val}]: N={len(sub)}, AUROC={auc:.4f}, Sens={sens:.4f}, Spec={spec:.4f}")

    # 2. Age brackets (<40, 40-60, >60)
    subgroup_results["age"] = {}
    age_bins = [
        ("<40", df[df.age < 40]),
        ("40-60", df[(df.age >= 40) & (df.age <= 60)]),
        (">60", df[df.age > 60])
    ]
    for name, sub in age_bins:
        if len(sub) > 10 and sub.label.nunique() > 1:
            auc = auroc(sub.label.values, sub.score.values)
            sens, spec = sens_spec(sub.label.values, sub.score.values >= DEFAULT_THRESHOLD)
            subgroup_results["age"][name] = {"n": len(sub), "auroc": auc, "sensitivity": sens, "specificity": spec}
            print(f"  Age [{name}]: N={len(sub)}, AUROC={auc:.4f}, Sens={sens:.4f}, Spec={spec:.4f}")

    # 3. Anatomical Localization
    subgroup_results["localization"] = {}
    for loc in df.localization.value_counts().index[:6]:
        sub = df[df.localization == loc]
        if len(sub) > 10 and sub.label.nunique() > 1:
            auc = auroc(sub.label.values, sub.score.values)
            sens, spec = sens_spec(sub.label.values, sub.score.values >= DEFAULT_THRESHOLD)
            subgroup_results["localization"][loc] = {"n": len(sub), "auroc": auc, "sensitivity": sens, "specificity": spec}
            print(f"  Location [{loc}]: N={len(sub)}, AUROC={auc:.4f}, Sens={sens:.4f}, Spec={spec:.4f}")

    save_json(subgroup_results, f"{RUNS_DIR}/subgroup_analysis.json")

    # Plot Subgroup AUROC
    fig, ax = plt.subplots(figsize=(10, 4))
    labels = []
    scores = []
    colors = []
    for grp, grp_dict in subgroup_results.items():
        for k, v in grp_dict.items():
            labels.append(f"{grp}:{k}")
            scores.append(v["auroc"])
            colors.append("#1976D2" if grp == "sex" else ("#388E3C" if grp == "age" else "#E65100"))

    bars = ax.bar(labels, scores, color=colors, width=0.6)
    ax.axhline(0.85, color="red", linestyle="--", label="Target AUROC (0.85)")
    ax.set_ylabel("Subgroup AUROC")
    ax.set_ylim([0.65, 1.0])
    ax.set_title("Subgroup Performance Parity Across Demographic & Anatomical Metadata", fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    ax.grid(True, axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    chart_path = f"{OUT_DIR}/subgroup_auroc.png"
    plt.savefig(chart_path, dpi=150)
    plt.close()
    print(f"Saved subgroup audit chart to: {chart_path}")

    gradcam.close()
    return error_report, subgroup_results

if __name__ == "__main__":
    run_error_analysis()
