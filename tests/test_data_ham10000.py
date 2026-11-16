import json, os
import numpy as np, pandas as pd, pytest
from PIL import Image

def test_mask_to_boxes():
    from medimg.data_ham10000 import mask_to_boxes

    # Empty mask
    empty = np.zeros((100, 100), dtype=np.uint8)
    assert mask_to_boxes(empty) == []

    # Single connected component
    single = np.zeros((100, 100), dtype=np.uint8)
    single[20:50, 30:70] = 255
    boxes = mask_to_boxes(single)
    assert len(boxes) == 1
    # Note: 0-indexed bounding coordinates [x_min, y_min, x_max, y_max]
    assert boxes[0] == [30, 20, 69, 49]

    # Two disjoint components
    two = np.zeros((100, 100), dtype=np.uint8)
    two[10:20, 10:20] = 255
    two[50:60, 50:60] = 255
    boxes2 = mask_to_boxes(two)
    assert len(boxes2) == 2

def test_ham10000_pipeline_mock(tmp_path):
    from medimg.data_ham10000 import build_ham10000_metadata
    from medimg.__main__ import run_module

    raw_dir = str(tmp_path / "raw_ham")
    out_dir = str(tmp_path / "proc_ham")
    os.makedirs(os.path.join(raw_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(raw_dir, "masks"), exist_ok=True)

    # Create mock metadata with 6 lesions (including one multi-image lesion)
    meta_rows = [
        {"lesion_id": "HAM_L01", "image_id": "ISIC_0001", "dx": "mel", "dx_type": "histo", "age": 55, "sex": "male", "localization": "back"},
        {"lesion_id": "HAM_L01", "image_id": "ISIC_0002", "dx": "mel", "dx_type": "histo", "age": 55, "sex": "male", "localization": "back"},
        {"lesion_id": "HAM_L02", "image_id": "ISIC_0003", "dx": "nv", "dx_type": "follow_up", "age": 30, "sex": "female", "localization": "face"},
        {"lesion_id": "HAM_L03", "image_id": "ISIC_0004", "dx": "bkl", "dx_type": "consensus", "age": 60, "sex": "male", "localization": "chest"},
        {"lesion_id": "HAM_L04", "image_id": "ISIC_0005", "dx": "bcc", "dx_type": "histo", "age": 70, "sex": "male", "localization": "ear"},
        {"lesion_id": "HAM_L05", "image_id": "ISIC_0006", "dx": "akiec", "dx_type": "histo", "age": 65, "sex": "female", "localization": "hand"},
    ]
    pd.DataFrame(meta_rows).to_csv(os.path.join(raw_dir, "HAM10000_metadata.csv"), index=False)

    # Create mock images and masks with distinct random noise so dedup doesn't merge them
    rng = np.random.default_rng(42)
    for i, r in enumerate(meta_rows):
        arr = rng.integers(50, 200, (64, 64, 3), dtype=np.uint8)
        img = Image.fromarray(arr)
        img.save(os.path.join(raw_dir, "images", f"{r['image_id']}.jpg"))

        msk = Image.new("L", (64, 64), color=0)
        if r["dx"] == "mel":
            msk_arr = np.array(msk)
            msk_arr[16:48, 16:48] = 255
            msk = Image.fromarray(msk_arr)
        msk.save(os.path.join(raw_dir, "masks", f"{r['image_id']}_segmentation.png"))

    df_out, summary = build_ham10000_metadata(raw_dir, out_dir)
    assert len(df_out) == 6
    assert summary["melanoma_count"] == 2
    assert summary["other_count"] == 4
    assert df_out["patient_id"].tolist()[:2] == ["HAM_L01", "HAM_L01"]

    # Verify boxes derived
    b0 = json.loads(df_out.iloc[0]["boxes_json"])
    assert len(b0) == 1
    assert b0[0] == [16, 16, 47, 47]

    # Verify compatibility with prepare_splits.py
    run_module("medimg.prepare_splits", "--root", out_dir, "--ratios", "0.5,0.25,0.25", "--seed", "42")
    splits_csv = os.path.join(out_dir, "splits.csv")
    assert os.path.exists(splits_csv)
    df_splits = pd.read_csv(splits_csv)
    # Check that lesion HAM_L01 did not cross splits
    l1_splits = df_splits[df_splits.image_id.isin(["ISIC_0001", "ISIC_0002"])].split.unique()
    assert len(l1_splits) == 1
