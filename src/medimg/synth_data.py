"""Synthetic 'lesion' dataset so the whole pipeline runs end-to-end before real data is downloaded.
Creates images, masks, boxes, patient ids, and deliberate duplicates (to exercise dedup + leakage checks).
NOT medical data - for plumbing and unit tests only."""
import argparse, json, os
import numpy as np
from PIL import Image

def make_image(rng, size, positive):
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    fx, fy, ph = rng.uniform(8, 30, 2).tolist() + [rng.uniform(0, 6.28)]
    img = 0.55 + 0.12 * np.sin(xx / fx + ph) * np.cos(yy / fy - ph) + 0.1 * (xx / size) * rng.uniform(-1, 1) \
          + rng.normal(0, 0.03, (size, size)).astype(np.float32)
    mask = np.zeros((size, size), np.uint8)
    boxes = []
    if positive:
        for _ in range(rng.integers(1, 3)):
            cx, cy = rng.integers(size // 4, 3 * size // 4, 2)
            ax, ay = rng.integers(size // 12, size // 5, 2)
            blob = ((xx - cx) / ax) ** 2 + ((yy - cy) / ay) ** 2 <= 1.0
            img[blob] -= 0.3 + 0.1 * rng.random()
            mask[blob] = 1
            boxes.append([int(cx - ax), int(cy - ay), int(cx + ax), int(cy + ay)])
    return np.clip(img, 0, 1), mask, boxes

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synth")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--pos_frac", type=float, default=0.3)   # class imbalance (minority = lesion)
    ap.add_argument("--dup_frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    for d in ("images", "masks"):
        os.makedirs(os.path.join(a.out, d), exist_ok=True)
    rows, n_patients = [], max(1, a.n // 3)
    for i in range(a.n):
        pos = rng.random() < a.pos_frac
        img, mask, boxes = make_image(rng, a.size, pos)
        rows.append((f"img_{i:05d}", f"pat_{rng.integers(n_patients):04d}", int(pos), img, mask, boxes))
    for k in range(int(a.n * a.dup_frac)):  # near-duplicates under a *different* patient id (worst case for leakage)
        s = rows[rng.integers(a.n)]
        noisy = np.clip(s[3] + rng.normal(0, 0.005, s[3].shape).astype(np.float32), 0, 1)
        rows.append((f"dup_{k:04d}", f"pat_{rng.integers(n_patients):04d}", s[2], noisy, s[4], s[5]))
    meta = ["image_id,patient_id,label,image_path,mask_path,boxes_json"]
    for iid, pid, lab, img, mask, boxes in rows:
        ip, mp = f"images/{iid}.png", f"masks/{iid}.png"
        Image.fromarray((img * 255).astype(np.uint8)).save(os.path.join(a.out, ip))
        Image.fromarray(mask * 255).save(os.path.join(a.out, mp))
        meta.append(f'{iid},{pid},{lab},{ip},{mp},"{json.dumps(boxes).replace(chr(34), chr(34)*2)}"')
    with open(os.path.join(a.out, "metadata.csv"), "w") as f:
        f.write("\n".join(meta) + "\n")
    print(f"wrote {len(rows)} samples to {a.out} (positives: {sum(r[2] for r in rows)})")

if __name__ == "__main__":
    main()
