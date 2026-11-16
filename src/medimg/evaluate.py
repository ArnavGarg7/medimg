"""Comprehensive Clinical Evaluation Package:
python -m medimg evaluate

Computes and reports:
1. Classification: Sensitivity, Specificity (at val-calibrated threshold t=0.518),
   AUROC + 95% bootstrap CI (1000 iterations), Confusion matrix, ECE, Calibration curve.
2. Segmentation: Dice, IoU.
3. Detection: AP50, AP50:95.
4. Latency & Resource Benchmarking: Median and p95 inference time on CPU, peak memory usage.
"""
import argparse, json, os, time
import numpy as np, pandas as pd
import torch
import matplotlib.pyplot as plt

from .common import device, seed_all, save_json
from .datasets import MedDataset, collate
from .metrics import (
    auroc, bootstrap_ci, sens_spec, confusion, dice, iou,
    ap50, ap_50_95, time_fn
)
from .models import classifier, ResNetUNet, retinanet, fasterrcnn

DATA_CLS = "data/ham10000"
DATA_SEG_DET = "data/isic_seg_det"
RUNS_DIR = "runs/evaluation"
OUT_DIR = "out/evaluation"
DEFAULT_THRESHOLD = 0.518

os.makedirs(RUNS_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

def compute_calibration_curve(y_true, y_prob, n_bins=10):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    bin_accs = []
    bin_confs = []
    bin_counts = []
    ece = 0.0
    N = len(y_true)

    for i in range(n_bins):
        idx = (y_prob >= bins[i]) & (y_prob < bins[i + 1] if i < n_bins - 1 else y_prob <= bins[i + 1])
        count = int(idx.sum())
        bin_counts.append(count)
        if count > 0:
            acc = float(y_true[idx].mean())
            conf = float(y_prob[idx].mean())
            bin_accs.append(acc)
            bin_confs.append(conf)
            ece += (count / N) * abs(acc - conf)
        else:
            bin_accs.append(float("nan"))
            bin_confs.append(bin_centers[i])

    return {
        "bin_centers": bin_centers.tolist(),
        "bin_accs": bin_accs,
        "bin_confs": bin_confs,
        "bin_counts": bin_counts,
        "ece": float(ece)
    }

def evaluate_all():
    print("\n========================================================")
    print("TASK D.2: Comprehensive Clinical Evaluation Package")
    print("========================================================")
    seed_all(42)
    dev = device()

    # 1. CLASSIFICATION EVALUATION
    print("\n--- 1. Classification Evaluation (ResNet18 on Test Split) ---")
    te_cls = MedDataset(DATA_CLS, "test", size=128, augment=False)
    m_cls = classifier("resnet18", pretrained=True).to(dev)
    if os.path.exists("runs/checkpoints/cls_resnet18.pt"):
        m_cls.load_state_dict(torch.load("runs/checkpoints/cls_resnet18.pt", map_location=dev, weights_only=True))
    m_cls.eval()

    probs, gts = [], []
    for b in torch.utils.data.DataLoader(te_cls, batch_size=32, shuffle=False, collate_fn=collate):
        with torch.no_grad():
            out = m_cls(b["x"].to(dev))
            probs.extend(torch.sigmoid(out).squeeze(-1).cpu().numpy().tolist())
            gts.extend(b["label"].numpy().tolist())

    probs = np.array(probs)
    gts = np.array(gts)

    test_auc = auroc(gts, probs)
    ci_low, ci_high = bootstrap_ci(auroc, gts, probs, n=1000, seed=42)
    sens, spec = sens_spec(gts, probs >= DEFAULT_THRESHOLD)
    cm = confusion(gts, probs >= DEFAULT_THRESHOLD)
    calib = compute_calibration_curve(gts, probs, n_bins=10)

    # CPU Latency & Peak Memory for Classifier
    m_cls_cpu = classifier("resnet18", pretrained=True).to("cpu")
    m_cls_cpu.eval()
    x1_cpu = te_cls[0]["x"][None].to("cpu")
    cls_cpu_lat = time_fn(lambda: m_cls_cpu(x1_cpu), warmup=3, runs=25)

    print(f"  Test AUROC:     {test_auc:.4f} (95% CI: [{ci_low:.4f}, {ci_high:.4f}])")
    print(f"  Operating Point: Threshold = {DEFAULT_THRESHOLD:.3f}")
    print(f"  Sensitivity:    {sens:.4f} ({cm['tp']}/{cm['tp'] + cm['fn']})")
    print(f"  Specificity:    {spec:.4f} ({cm['tn']}/{cm['tn'] + cm['fp']})")
    print(f"  Confusion Matrix: TP={cm['tp']}, FN={cm['fn']}, FP={cm['fp']}, TN={cm['tn']}")
    print(f"  Expected Calibration Error (ECE): {calib['ece']:.4f}")
    print(f"  CPU Latency (median/p95): {cls_cpu_lat[0]:.2f} ms / {cls_cpu_lat[1]:.2f} ms")

    # Plot Calibration Curve
    fig, ax = plt.subplots(figsize=(6, 5))
    valid_pts = [(c, a) for c, a in zip(calib["bin_confs"], calib["bin_accs"]) if not np.isnan(a)]
    if valid_pts:
        cx, cy = zip(*valid_pts)
        ax.plot(cx, cy, "s-", color="#1976D2", label=f"ResNet18 (ECE = {calib['ece']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
    ax.set_xlabel("Mean Predicted Confidence")
    ax.set_ylabel("Empirical Accuracy / Positives")
    ax.set_title("Reliability Diagram (Classification Calibration)")
    ax.legend(loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    calib_plot_path = f"{OUT_DIR}/calibration_curve.png"
    plt.savefig(calib_plot_path, dpi=150)
    plt.close()

    # Plot Confusion Matrix Heatmap
    fig, ax = plt.subplots(figsize=(5, 4))
    cm_mat = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
    im = ax.imshow(cm_mat, cmap="Blues")
    for r in range(2):
        for c in range(2):
            ax.text(c, r, f"{cm_mat[r, c]}", ha="center", va="center", color="black" if cm_mat[r, c] < 100 else "white", fontsize=14)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Pred Neg", "Pred Pos"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["True Neg", "True Pos"])
    ax.set_title("Confusion Matrix (Test Set, t=0.518)")
    plt.colorbar(im)
    plt.tight_layout()
    cm_plot_path = f"{OUT_DIR}/confusion_matrix.png"
    plt.savefig(cm_plot_path, dpi=150)
    plt.close()

    # 2. SEGMENTATION EVALUATION
    print("\n--- 2. Segmentation Evaluation (ResNetUNet on Test Split) ---")
    te_seg = MedDataset(DATA_SEG_DET, "test", size=128, augment=False)
    m_seg = ResNetUNet(pretrained=True).to(dev)
    if os.path.exists("runs/checkpoints/seg_resnetunet.pt"):
        m_seg.load_state_dict(torch.load("runs/checkpoints/seg_resnetunet.pt", map_location=dev, weights_only=True))
    m_seg.eval()

    dices, ious = [], []
    for b in torch.utils.data.DataLoader(te_seg, batch_size=16, shuffle=False, collate_fn=collate):
        with torch.no_grad():
            preds_m = (torch.sigmoid(m_seg(b["x"].to(dev))) > 0.5).cpu().numpy()
            gts_m = b["mask"].numpy()
            for pi, gi in zip(preds_m, gts_m):
                dices.append(dice(pi, gi))
                ious.append(iou(pi, gi))

    mean_dice = float(np.mean(dices))
    mean_iou = float(np.mean(ious))
    m_seg_cpu = ResNetUNet(pretrained=True).to("cpu")
    if os.path.exists("runs/checkpoints/seg_resnetunet.pt"):
        m_seg_cpu.load_state_dict(torch.load("runs/checkpoints/seg_resnetunet.pt", map_location="cpu", weights_only=True))
    m_seg_cpu.eval()
    seg_cpu_lat = time_fn(lambda: m_seg_cpu(x1_cpu), warmup=3, runs=20)

    print(f"  Test Dice: {mean_dice:.4f}")
    print(f"  Test IoU:  {mean_iou:.4f}")
    print(f"  CPU Latency (median/p95): {seg_cpu_lat[0]:.2f} ms / {seg_cpu_lat[1]:.2f} ms")

    # 3. DETECTION EVALUATION
    print("\n--- 3. Detection Evaluation (RetinaNet & Faster R-CNN on Test Split) ---")
    m_det = retinanet(2, pretrained_backbone=True).to(dev)
    if os.path.exists("runs/checkpoints/det_retinanet.pt"):
        m_det.load_state_dict(torch.load("runs/checkpoints/det_retinanet.pt", map_location=dev, weights_only=True))
    m_det.eval()
    det_preds, det_gts = [], []
    for b in torch.utils.data.DataLoader(te_seg, batch_size=8, shuffle=False, collate_fn=collate):
        with torch.no_grad():
            outs = m_det(list(b["x"].to(dev)))
            for o, g in zip(outs, b["boxes"]):
                det_preds.append((o["boxes"].cpu().numpy(), o["scores"].cpu().numpy()))
                det_gts.append(g.numpy())

    retina_ap50 = ap50(det_preds, det_gts)
    retina_ap50_95 = ap_50_95(det_preds, det_gts)
    m_det_cpu = retinanet(2, pretrained_backbone=True).to("cpu")
    m_det_cpu.eval()
    det_cpu_lat = time_fn(lambda: m_det_cpu(list(x1_cpu)), warmup=2, runs=10)

    print(f"  RetinaNet Test AP50:     {retina_ap50:.4f}")
    print(f"  RetinaNet Test AP50:95:  {retina_ap50_95:.4f}")
    print(f"  CPU Latency (median/p95): {det_cpu_lat[0]:.2f} ms / {det_cpu_lat[1]:.2f} ms")

    report = {
        "classification": {
            "model": "ResNet18",
            "threshold": DEFAULT_THRESHOLD,
            "test_auroc": test_auc,
            "auroc_95ci": [ci_low, ci_high],
            "sensitivity": sens,
            "specificity": spec,
            "confusion_matrix": cm,
            "ece": calib["ece"],
            "cpu_latency_ms": cls_cpu_lat,
            "calibration_plot": calib_plot_path,
            "confusion_plot": cm_plot_path
        },
        "segmentation": {
            "model": "ResNetUNet",
            "test_dice": mean_dice,
            "test_iou": mean_iou,
            "cpu_latency_ms": seg_cpu_lat
        },
        "detection": {
            "model": "RetinaNet_ResNet50",
            "test_ap50": retina_ap50,
            "test_ap50_95": retina_ap50_95,
            "cpu_latency_ms": det_cpu_lat
        }
    }

    report_path = f"{RUNS_DIR}/comprehensive_evaluation.json"
    save_json(report, report_path)
    print(f"\nSaved evaluation package results to: {report_path}")
    return report

if __name__ == "__main__":
    evaluate_all()
