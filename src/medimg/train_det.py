"""Fine-tune torchvision RetinaNet for lesion detection; reports AP@0.5 (simple, VOC-style) on test."""
import argparse, os
import numpy as np, torch
from torch.utils.data import DataLoader
from .common import device, run_meta, save_json, seed_all
from .datasets import MedDataset, collate
from .metrics import ap50
from .models import retinanet

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/synth"); ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--bs", type=int, default=8); ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--size", type=int, default=128); ap.add_argument("--pretrained_backbone", action="store_true")
    ap.add_argument("--seed", type=int, default=42); ap.add_argument("--out", default="runs/det"); a = ap.parse_args()
    seed_all(a.seed); dev = device(); os.makedirs(a.out, exist_ok=True)
    mk = lambda s, **k: DataLoader(MedDataset(a.root, s, a.size, **k), batch_size=a.bs, collate_fn=collate, shuffle=(s == "train"))
    trl, tel = mk("train", augment=True), mk("test")
    model = retinanet(2, a.pretrained_backbone).to(dev); opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
    for ep in range(a.epochs):
        model.train(); tot = 0
        for b in trl:
            tg = [dict(boxes=bx.to(dev), labels=torch.ones(len(bx), dtype=torch.int64, device=dev)) for bx in b["boxes"]]
            loss = sum(model(list(b["x"].to(dev)), tg).values()); opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
        print(dict(epoch=ep + 1, loss=tot / len(trl)))
    torch.save(model.state_dict(), f"{a.out}/last.pt"); model.eval(); preds, gts = [], []
    with torch.no_grad():
        for b in tel:
            for o, g in zip(model(list(b["x"].to(dev))), b["boxes"]):
                preds.append((o["boxes"].cpu().numpy(), o["scores"].cpu().numpy())); gts.append(g.numpy())
    rep = dict(test_ap50=ap50(preds, gts), meta=run_meta(a)); save_json(rep, f"{a.out}/report.json"); print(rep["test_ap50"])

if __name__ == "__main__":
    main()
