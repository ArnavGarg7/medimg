"""Triage and lesion analysis pipeline inference entry point:
python -m medimg infer --image <path> --out <dir>

Produces:
1. Triage decision (HIGH/LOW PRIORITY) at validation-selected threshold (Sens >= 95%).
2. Melanoma risk score (calibrated probability).
3. Lesion segmentation mask & contour overlay.
4. Bounding box detections with confidence scores.
5. Factual decision-support summary text.
6. Execution latency breakdown.
"""
import argparse, json, os, time
import numpy as np, torch
from PIL import Image, ImageDraw, ImageFont

from .common import device, seed_all, save_json
from .models import classifier, ResNetUNet, retinanet

# Default operating threshold selected on validation set to achieve Sensitivity >= 95%
DEFAULT_THRESHOLD = 0.518

def load_models(dev=None):
    if dev is None:
        dev = device()
    cls_net = classifier("resnet18", pretrained=True).to(dev)
    if os.path.exists("runs/checkpoints/cls_resnet18.pt"):
        cls_net.load_state_dict(torch.load("runs/checkpoints/cls_resnet18.pt", map_location=dev, weights_only=True))

    seg_net = ResNetUNet(pretrained=True).to(dev)
    if os.path.exists("runs/checkpoints/seg_resnetunet.pt"):
        seg_net.load_state_dict(torch.load("runs/checkpoints/seg_resnetunet.pt", map_location=dev, weights_only=True))

    det_net = retinanet(num_classes=2, pretrained_backbone=True).to(dev)
    if os.path.exists("runs/checkpoints/det_retinanet.pt"):
        det_net.load_state_dict(torch.load("runs/checkpoints/det_retinanet.pt", map_location=dev, weights_only=True))

    cls_net.eval()
    seg_net.eval()
    det_net.eval()
    return {"cls": cls_net, "seg": seg_net, "det": det_net, "device": dev}

def run_inference(image_path, out_dir=None, models=None, threshold=DEFAULT_THRESHOLD, size=128):
    t_start = time.perf_counter()
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Input image not found: {image_path}")
    
    dev = models["device"] if models else device()
    if models is None:
        models = load_models(dev)

    # Preprocessing
    t_pre0 = time.perf_counter()
    raw_img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = raw_img.size
    img_resized = raw_img.resize((size, size), Image.BILINEAR)
    img_np = np.asarray(img_resized, dtype=np.float32) / 255.0
    x_tensor = torch.from_numpy(img_np.transpose(2, 0, 1))[None].to(dev)
    t_pre = (time.perf_counter() - t_pre0) * 1e3

    # Inference
    t_inf0 = time.perf_counter()
    with torch.no_grad():
        cls_logits = models["cls"](x_tensor)
        risk_score = float(torch.sigmoid(cls_logits).item())

        seg_logits = models["seg"](x_tensor)
        mask_prob = torch.sigmoid(seg_logits)[0, 0].cpu().numpy()
        mask_binary = (mask_prob > 0.5).astype(np.uint8)

        det_out = models["det"](list(x_tensor))[0]
        boxes = det_out["boxes"].cpu().numpy()
        scores = det_out["scores"].cpu().numpy()
    t_inf = (time.perf_counter() - t_inf0) * 1e3

    # Postprocessing & Metrics
    t_post0 = time.perf_counter()
    priority = "HIGH PRIORITY" if risk_score >= threshold else "LOW PRIORITY"
    lesion_area_pct = float(mask_binary.sum() / mask_binary.size * 100.0)

    # Filter detection boxes above score threshold 0.20
    det_keep = scores > 0.20
    det_boxes = boxes[det_keep].tolist()
    det_scores = scores[det_keep].tolist()

    # Create composite overlay
    overlay = img_resized.copy()
    overlay_np = np.asarray(overlay).astype(np.float32)
    
    # Red mask overlay on segmented area
    mask_indices = mask_binary == 1
    overlay_np[mask_indices, 0] = overlay_np[mask_indices, 0] * 0.6 + 255.0 * 0.4
    overlay_np[mask_indices, 1] = overlay_np[mask_indices, 1] * 0.6
    overlay_np[mask_indices, 2] = overlay_np[mask_indices, 2] * 0.6
    overlay_img = Image.fromarray(np.clip(overlay_np, 0, 255).astype(np.uint8))
    
    draw = ImageDraw.Draw(overlay_img)
    # Draw detection bounding boxes
    for bx, sc in zip(det_boxes, det_scores):
        draw.rectangle([bx[0], bx[1], bx[2], bx[3]], outline=(255, 255, 0), width=2)
        draw.text((bx[0] + 2, bx[1] + 2), f"Lesion {sc*100:.1f}%", fill=(255, 255, 0))

    # Top status banner
    banner_color = (180, 20, 20) if priority == "HIGH PRIORITY" else (20, 140, 20)
    draw.rectangle([0, 0, size, 16], fill=banner_color)
    draw.text((3, 2), f"{priority} | Risk: {risk_score*100:.1f}%", fill=(255, 255, 255))
    
    t_post = (time.perf_counter() - t_post0) * 1e3
    total_ms = (time.perf_counter() - t_start) * 1e3

    # Structured clinical summary
    summary_text = (
        "CLINICAL TRIAGE & LESION ANALYSIS REPORT\n"
        "=========================================\n"
        "DISCLAIMER: This analysis is an AI decision-support aid and DOES NOT diagnose.\n"
        "All recommendations must be evaluated and confirmed by a qualified clinician.\n\n"
        f"Triage Decision:  {priority}\n"
        f"Melanoma Risk:    {risk_score * 100:.2f}%\n"
        f"Operating Point:  Threshold {threshold:.3f} (Val-calibrated target Sensitivity >= 95%)\n"
        f"Lesion Footprint: {lesion_area_pct:.2f}% of field of view ({mask_binary.sum()} px)\n"
        f"Suspicious Foci:  {len(det_boxes)} detected bounding box(es)\n"
        f"Total Latency:    {total_ms:.1f} ms (Pre: {t_pre:.1f}ms, Inf: {t_inf:.1f}ms, Post: {t_post:.1f}ms)\n\n"
        f"Clinical Pathway: {'Fast-track urgent dermatologist consultation and dermatoscopy review.' if priority == 'HIGH PRIORITY' else 'Routine clinical follow-up; monitor for changes.'}"
    )

    result = {
        "image_path": str(image_path),
        "priority": priority,
        "risk_score": risk_score,
        "threshold": threshold,
        "is_high_priority": priority == "HIGH PRIORITY",
        "lesion_area_pct": lesion_area_pct,
        "detection_count": len(det_boxes),
        "detection_boxes": det_boxes,
        "detection_scores": det_scores,
        "latency_ms": {
            "preprocess": t_pre,
            "inference": t_inf,
            "postprocess": t_post,
            "total": total_ms
        },
        "summary": summary_text
    }

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        overlay_path = os.path.join(out_dir, "triage_overlay.png")
        mask_path = os.path.join(out_dir, "segmentation_mask.png")
        report_path = os.path.join(out_dir, "triage_report.json")
        summary_path = os.path.join(out_dir, "triage_summary.txt")

        overlay_img.save(overlay_path)
        Image.fromarray(mask_binary * 255).save(mask_path)
        save_json(result, report_path)
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(summary_text + "\n")
        
        result["saved_files"] = {
            "overlay": overlay_path,
            "mask": mask_path,
            "report_json": report_path,
            "summary_txt": summary_path
        }

    return result, overlay_img

def main():
    parser = argparse.ArgumentParser(description="Medical Imaging Triage Inference Entry Point")
    parser.add_argument("--image", required=True, help="Path to input dermoscopy image")
    parser.add_argument("--out", default="runs/infer_out", help="Output directory for predictions and visualizations")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="Triage operating threshold")
    parser.add_argument("--size", type=int, default=128, help="Processing resolution")
    args = parser.parse_args()

    res, _ = run_inference(args.image, out_dir=args.out, threshold=args.threshold, size=args.size)
    print(res["summary"])
    print(f"\nSaved outputs to: {args.out}")

if __name__ == "__main__":
    main()
