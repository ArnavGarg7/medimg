"""Train U-Net-style segmentation; reports Dice / IoU on val (model selection) and test."""
import argparse, os
import numpy as np, torch
from torch.utils.data import DataLoader
from .common import device, run_meta, save_json, seed_all
from .datasets import MedDataset, collate
from .metrics import dice, iou, time_fn
from .models import UNet, dice_bce_loss

@torch.no_grad()
def evaluate(model, loader, dev):
    model.eval(); d, j = [], []
    for b in loader:
        p = (torch.sigmoid(model(b["x"].to(dev))) > 0.5).cpu().numpy(); g = b["mask"].numpy()
        for pi, gi in zip(p, g): d.append(dice(pi, gi)); j.append(iou(pi, gi))
    return float(np.mean(d)), float(np.mean(j))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/synth"); ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--bs", type=int, default=16); ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--size", type=int, default=128); ap.add_argument("--base", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42); ap.add_argument("--out", default="runs/seg"); a = ap.parse_args()
    seed_all(a.seed); dev = device(); os.makedirs(a.out, exist_ok=True)
    L = lambda s, **k: DataLoader(MedDataset(a.root, s, a.size, **k), batch_size=a.bs, collate_fn=collate, shuffle=(s == "train"))
    trl, val, tel = L("train", augment=True), L("val"), L("test")
    model = UNet(3, a.base).to(dev); opt = torch.optim.AdamW(model.parameters(), lr=a.lr); best, hist = -1, []
    for ep in range(a.epochs):
        model.train(); tl = 0
        for b in trl:
            opt.zero_grad(); l = dice_bce_loss(model(b["x"].to(dev)), b["mask"].to(dev))
            l.backward(); opt.step(); tl += l.item() * len(b["mask"])
        vd, vi = evaluate(model, val, dev); hist.append(dict(epoch=ep + 1, train_loss=tl / len(trl.dataset), val_dice=vd, val_iou=vi)); print(hist[-1])
        if vd > best: best = vd; torch.save(model.state_dict(), f"{a.out}/best.pt")
    model.load_state_dict(torch.load(f"{a.out}/best.pt", map_location=dev, weights_only=True)); td, ti = evaluate(model, tel, dev)
    x1 = next(iter(tel))["x"][:1].to(dev)
    rep = dict(test_dice=td, test_iou=ti, latency_ms_median_p95=time_fn(lambda: model(x1)), history=hist, meta=run_meta(a))
    save_json(rep, f"{a.out}/report.json"); print(dict(test_dice=td, test_iou=ti))

if __name__ == "__main__":
    main()
