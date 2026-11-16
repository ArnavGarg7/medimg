"""Fine-tune torchvision RetinaNet for lesion detection; selects best epoch on VAL AP50, reports TEST once at the end."""
import argparse, os
import numpy as np, torch
from torch.utils.data import DataLoader
from .common import device, run_meta, save_json, seed_all
from .datasets import MedDataset, collate
from .metrics import ap50, time_fn
from .models import retinanet

@torch.no_grad()
def evaluate_det(model, loader, dev):
    model.eval(); preds, gts = [], []
    for b in loader:
        for o, g in zip(model(list(b["x"].to(dev))), b["boxes"]):
            preds.append((o["boxes"].cpu().numpy(), o["scores"].cpu().numpy())); gts.append(g.numpy())
    return ap50(preds, gts)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/synth"); ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--bs", type=int, default=8); ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--size", type=int, default=128); ap.add_argument("--pretrained_backbone", action="store_true")
    ap.add_argument("--seed", type=int, default=42); ap.add_argument("--out", default="runs/det"); a = ap.parse_args()
    seed_all(a.seed); dev = device(); os.makedirs(a.out, exist_ok=True)
    mk = lambda s, **k: DataLoader(MedDataset(a.root, s, a.size, **k), batch_size=a.bs, collate_fn=collate, shuffle=(s == "train"))
    trl, val, tel = mk("train", augment=True), mk("val"), mk("test")
    model = retinanet(2, a.pretrained_backbone).to(dev); opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
    best_ap, hist = -1.0, []
    for ep in range(a.epochs):
        model.train(); tot = 0
        for b in trl:
            tg = [dict(boxes=bx.to(dev), labels=torch.ones(len(bx), dtype=torch.int64, device=dev)) for bx in b["boxes"]]
            loss = sum(model(list(b["x"].to(dev)), tg).values()); opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
        v_ap = evaluate_det(model, val, dev)
        hist.append(dict(epoch=ep + 1, train_loss=tot / len(trl), val_ap50=v_ap))
        print(hist[-1])
        if v_ap > best_ap:
            best_ap = v_ap
            torch.save(model.state_dict(), f"{a.out}/best.pt")
    # Touch test set once at the end using best model selected on val
    ckpt_path = f"{a.out}/best.pt" if os.path.exists(f"{a.out}/best.pt") else f"{a.out}/last.pt"
    if not os.path.exists(ckpt_path):
        torch.save(model.state_dict(), ckpt_path)
    model.load_state_dict(torch.load(ckpt_path, map_location=dev, weights_only=True))
    test_ap = evaluate_det(model, tel, dev)
    x1 = next(iter(tel))["x"][:1].to(dev)
    lat = time_fn(lambda: model(list(x1)), warmup=3, runs=20)
    rep = dict(test_ap50=test_ap, latency_ms_median_p95=lat, history=hist, meta=run_meta(a))
    save_json(rep, f"{a.out}/report.json")
    print(dict(test_ap50=test_ap, latency_ms_median_p95=lat))

if __name__ == "__main__":
    main()
