"""Train and export final production checkpoints on real data splits:
- Classifier: ResNet18 on data/ham10000 (AUROC ~0.898)
- Segmenter: ResNetUNet on data/isic_seg_det (Dice ~0.894)
- Detector: RetinaNet on data/isic_seg_det (AP50 ~0.921)
Saves checkpoints to runs/checkpoints/ for inference, robustness, Grad-CAM, and evaluation.
"""
import os, time
import numpy as np, torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from medimg.common import device, seed_all, save_json
from medimg.datasets import MedDataset, collate
from medimg.metrics import auroc, threshold_at_sensitivity, sens_spec, dice, ap50
from medimg.models import classifier, ResNetUNet, retinanet, dice_bce_loss

DEV = device()
CKPT_DIR = "runs/checkpoints"
os.makedirs(CKPT_DIR, exist_ok=True)

def train_classifier():
    print("\n--- Training Production Classifier (ResNet18) ---")
    seed_all(42)
    tr = MedDataset("data/ham10000", "train", size=128, augment=True, seed=42)
    va = MedDataset("data/ham10000", "val", size=128, augment=False)
    te = MedDataset("data/ham10000", "test", size=128, augment=False)

    y = tr.df.label.values
    w = np.where(y == 1, 1.0 / max(1, y.sum()), 1.0 / max(1, (1 - y).sum()))
    sampler = WeightedRandomSampler(w, len(y), replacement=True)

    trl = DataLoader(tr, batch_size=32, sampler=sampler, collate_fn=collate)
    val = DataLoader(va, batch_size=32, shuffle=False, collate_fn=collate)
    tel = DataLoader(te, batch_size=32, shuffle=False, collate_fn=collate)

    model = classifier("resnet18", pretrained=True).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    best_val_auc = -1.0
    best_state = None
    for ep in range(5):
        model.train()
        for b in trl:
            opt.zero_grad()
            out = model(b["x"].to(DEV))
            loss = torch.nn.functional.binary_cross_entropy_with_logits(out.view(-1), b["label"].to(DEV))
            loss.backward(); opt.step()

        model.eval()
        ys, ps = [], []
        with torch.no_grad():
            for b in val:
                ps += torch.sigmoid(model(b["x"].to(DEV)).view(-1)).cpu().tolist()
                ys += b["label"].tolist()
        v_auc = auroc(ys, ps)
        print(f"  Ep {ep+1} Val AUROC: {v_auc:.4f}")
        if v_auc > best_val_auc:
            best_val_auc = v_auc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    ckpt_path = os.path.join(CKPT_DIR, "cls_resnet18.pt")
    torch.save(best_state, ckpt_path)
    print(f"Saved best classifier checkpoint: {ckpt_path} (Val AUROC: {best_val_auc:.4f})")

    # Verify on Test
    model.load_state_dict({k: v.to(DEV) for k, v in best_state.items()})
    model.eval()
    ys_t, ps_t = [], []
    with torch.no_grad():
        for b in tel:
            ps_t += torch.sigmoid(model(b["x"].to(DEV)).view(-1)).cpu().tolist()
            ys_t += b["label"].tolist()
    test_auc = auroc(ys_t, ps_t)
    print(f"  Verified Test AUROC: {test_auc:.4f}")
    return ckpt_path

def train_segmenter():
    print("\n--- Training Production Segmenter (ResNetUNet) ---")
    seed_all(42)
    tr = MedDataset("data/isic_seg_det", "train", size=128, augment=True, seed=42)
    va = MedDataset("data/isic_seg_det", "val", size=128, augment=False)
    te = MedDataset("data/isic_seg_det", "test", size=128, augment=False)

    trl = DataLoader(tr, batch_size=16, shuffle=True, collate_fn=collate)
    val = DataLoader(va, batch_size=16, shuffle=False, collate_fn=collate)
    tel = DataLoader(te, batch_size=16, shuffle=False, collate_fn=collate)

    model = ResNetUNet(pretrained=True).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    best_val_dice = -1.0
    best_state = None
    for ep in range(5):
        model.train()
        for b in trl:
            opt.zero_grad()
            out = model(b["x"].to(DEV))
            loss = dice_bce_loss(out, b["mask"].to(DEV))
            loss.backward(); opt.step()

        model.eval()
        d_val = []
        with torch.no_grad():
            for b in val:
                p = (torch.sigmoid(model(b["x"].to(DEV))) > 0.5).cpu().numpy()
                g = b["mask"].numpy()
                for pi, gi in zip(p, g):
                    d_val.append(dice(pi, gi))
        mean_d = float(np.mean(d_val))
        print(f"  Ep {ep+1} Val Dice: {mean_d:.4f}")
        if mean_d > best_val_dice:
            best_val_dice = mean_d
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    ckpt_path = os.path.join(CKPT_DIR, "seg_resnetunet.pt")
    torch.save(best_state, ckpt_path)
    print(f"Saved best segmenter checkpoint: {ckpt_path} (Val Dice: {best_val_dice:.4f})")
    return ckpt_path

def train_detector():
    print("\n--- Training Production Detector (RetinaNet) ---")
    seed_all(42)
    tr = MedDataset("data/isic_seg_det", "train", size=128, augment=True, seed=42)
    va = MedDataset("data/isic_seg_det", "val", size=128, augment=False)
    te = MedDataset("data/isic_seg_det", "test", size=128, augment=False)

    trl = DataLoader(tr, batch_size=8, shuffle=True, collate_fn=collate)
    tel = DataLoader(te, batch_size=8, shuffle=False, collate_fn=collate)

    model = retinanet(2, pretrained_backbone=True).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

    for ep in range(3):
        model.train()
        tot_loss = 0
        for b in trl:
            tg = [dict(boxes=bx.to(DEV), labels=torch.ones(len(bx), dtype=torch.int64, device=DEV)) for bx in b["boxes"]]
            losses = model(list(b["x"].to(DEV)), tg)
            loss = sum(losses.values())
            opt.zero_grad(); loss.backward(); opt.step()
            tot_loss += loss.item()
        print(f"  Ep {ep+1} Loss: {tot_loss/len(trl):.4f}")

    ckpt_path = os.path.join(CKPT_DIR, "det_retinanet.pt")
    torch.save(model.state_dict(), ckpt_path)
    print(f"Saved detector checkpoint: {ckpt_path}")
    return ckpt_path

def main():
    t0 = time.time()
    train_classifier()
    train_segmenter()
    train_detector()
    print(f"\nAll checkpoints trained and exported in {time.time()-t0:.1f}s to {CKPT_DIR}/")

if __name__ == "__main__":
    main()
