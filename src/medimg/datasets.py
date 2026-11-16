"""Shared dataset for classification / segmentation / detection from splits.csv (columns: image_id,label,image_path,mask_path,boxes_json,split)."""
import json, os
import numpy as np, pandas as pd, torch
from PIL import Image
from torch.utils.data import Dataset

class MedDataset(Dataset):
    def __init__(self, root, split, size=128, augment=False, rgb=True, seed=0):
        df = pd.read_csv(os.path.join(root, "splits.csv"))
        self.df, self.root, self.size, self.augment, self.rgb = df[df.split == split].reset_index(drop=True), root, size, augment, rgb
        self.rng = np.random.default_rng(seed)

    def __len__(self): return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = Image.open(os.path.join(self.root, r.image_path)).convert("RGB" if self.rgb else "L")
        msk_raw = Image.open(os.path.join(self.root, r.mask_path)).convert("L")
        mask_w, mask_h = msk_raw.size
        img = img.resize((self.size, self.size), Image.BILINEAR)
        msk = msk_raw.resize((self.size, self.size), Image.NEAREST)
        x = np.asarray(img, np.float32) / 255.0
        x = x.transpose(2, 0, 1) if self.rgb else x[None]
        m = (np.asarray(msk, np.float32) > 127)[None].astype(np.float32)
        raw_boxes = np.array(json.loads(r.boxes_json) or [], np.float32).reshape(-1, 4)
        if len(raw_boxes):
            raw_boxes[:, [0, 2]] *= (self.size / max(1, mask_w))
            raw_boxes[:, [1, 3]] *= (self.size / max(1, mask_h))
            raw_boxes = np.clip(raw_boxes, 0, self.size)
            valid = (raw_boxes[:, 2] - raw_boxes[:, 0] >= 1.0) & (raw_boxes[:, 3] - raw_boxes[:, 1] >= 1.0)
            boxes = raw_boxes[valid]
        else:
            boxes = np.zeros((0, 4), dtype=np.float32)
        if self.augment:   # train-only augmentation: flips (applied identically to image, mask, boxes)
            if self.rng.random() < 0.5:
                x, m = x[..., ::-1].copy(), m[..., ::-1].copy()
                if len(boxes):
                    boxes[:, [0, 2]] = self.size - boxes[:, [2, 0]]
                    boxes = np.clip(boxes, 0, self.size)
            if self.rng.random() < 0.5:
                x, m = x[:, ::-1, :].copy(), m[:, ::-1, :].copy()
                if len(boxes):
                    boxes[:, [1, 3]] = self.size - boxes[:, [3, 1]]
                    boxes = np.clip(boxes, 0, self.size)
        return dict(x=torch.from_numpy(x), mask=torch.from_numpy(m), label=torch.tensor(float(r.label)),
                    boxes=torch.from_numpy(boxes), image_id=r.image_id)

def collate(batch):
    return dict(x=torch.stack([b["x"] for b in batch]), mask=torch.stack([b["mask"] for b in batch]),
                label=torch.stack([b["label"] for b in batch]), boxes=[b["boxes"] for b in batch],
                image_id=[b["image_id"] for b in batch])
