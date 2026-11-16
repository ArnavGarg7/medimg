"""Train ResNet classifier (lesion present / absent). Selects best epoch on VAL AUROC; reports TEST once at the end."""
import argparse, os
import numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from .common import device, run_meta, save_json, seed_all
from .datasets import MedDataset, collate
from .metrics import auroc, bootstrap_ci, sens_spec, threshold_at_sensitivity, time_fn, confusion
from .models import resnet_classifier

@torch.no_grad()
def predict(model, loader, dev):
    model.eval(); ys, ps = [], []
    for b in loader:
        ps += torch.sigmoid(model(b["x"].to(dev)).squeeze(1)).cpu().tolist(); ys += b["label"].tolist()
    return np.array(ys), np.array(ps)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/synth"); ap.add_argument("--arch", default="resnet18")
    ap.add_argument("--epochs", type=int, default=10); ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--pretrained", action="store_true"); ap.add_argument("--balance", action="store_true", help="oversample minority class")
    ap.add_argument("--no_aug", action="store_true"); ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="runs/cls"); a = ap.parse_args()
    seed_all(a.seed); dev = device()
    tr = MedDataset(a.root, "train", a.size, augment=not a.no_aug); va = MedDataset(a.root, "val", a.size); te = MedDataset(a.root, "test", a.size)
    sampler = None
    if a.balance:
        y = tr.df.label.values; w = np.where(y == 1, 1.0 / max(1, y.sum()), 1.0 / max(1, (1 - y).sum()))
        sampler = WeightedRandomSampler(w, len(y), replacement=True)
    L = lambda d, **k: DataLoader(d, batch_size=a.bs, collate_fn=collate, **k)
    trl, val, tel = L(tr, shuffle=sampler is None, sampler=sampler), L(va), L(te)
    model = resnet_classifier(a.arch, a.pretrained).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4); lossf = nn.BCEWithLogitsLoss()
    best, hist = -1, []
    os.makedirs(a.out, exist_ok=True)
    for ep in range(a.epochs):
        model.train(); tl = 0
        for b in trl:
            opt.zero_grad(); l = lossf(model(b["x"].to(dev)).squeeze(1), b["label"].to(dev)); l.backward(); opt.step(); tl += l.item() * len(b["label"])
        yv, pv = predict(model, val, dev); au = auroc(yv, pv)
        hist.append(dict(epoch=ep + 1, train_loss=tl / len(tr), val_auroc=au)); print(hist[-1])
        if au > best: best = au; torch.save(model.state_dict(), f"{a.out}/best.pt")
    model.load_state_dict(torch.load(f"{a.out}/best.pt", map_location=dev, weights_only=True)); yv, pv = predict(model, val, dev)
    thr = threshold_at_sensitivity(yv, pv, 0.95)               # operating point chosen on VAL only
    yt, pt = predict(model, tel, dev); se, sp = sens_spec(yt, pt >= thr)
    x1 = te[0]["x"][None].to(dev)
    lat = time_fn(lambda: model(x1), warmup=3, runs=20)
    rep = dict(test_auroc=auroc(yt, pt), test_auroc_ci95=bootstrap_ci(auroc, yt, pt), threshold=thr, sensitivity=se, specificity=sp,
               confusion=confusion(yt, pt >= thr), latency_ms_median_p95=lat, history=hist, meta=run_meta(a))
    save_json(rep, f"{a.out}/report.json"); print({k: v for k, v in rep.items() if k not in ("history", "meta")})

if __name__ == "__main__":
    main()
