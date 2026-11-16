import json
import numpy as np, pandas as pd, pytest
from PIL import Image
from medimg import metrics as M
from medimg.__main__ import run_module

def test_auroc_known_and_ties_vs_sklearn():
    y = np.array([0, 0, 1, 1, 1, 0]); s = np.array([.1, .4, .35, .8, .9, .4])
    assert M.auroc(y, s) == pytest.approx(7 / 9)
    sk = pytest.importorskip("sklearn.metrics"); r = np.random.default_rng(0)
    y = r.integers(0, 2, 500); s = np.round(r.random(500) + .3 * y, 1)       # many ties
    assert M.auroc(y, s) == pytest.approx(sk.roc_auc_score(y, s))
    assert np.isnan(M.auroc([1, 1], [.2, .3]))                                # single class -> undefined

def test_sens_spec_threshold_and_ci():
    y = np.array([0, 0, 1, 1, 1, 0]); s = np.array([.1, .4, .35, .8, .9, .4])
    assert M.sens_spec(y, s >= .5) == (2 / 3, 1.0)
    t = M.threshold_at_sensitivity(y, s, 1.0); assert M.sens_spec(y, s >= t)[0] == 1.0
    lo, hi = M.bootstrap_ci(M.auroc, y, s, n=200); assert 0 <= lo <= hi <= 1

def test_dice_iou_box_ap():
    a = np.zeros((8, 8)); a[:4] = 1; b = np.zeros((8, 8)); b[2:6] = 1
    assert M.dice(a, b) == pytest.approx(.5, abs=1e-6) and M.iou(a, b) == pytest.approx(1 / 3, abs=1e-6)
    assert M.dice(np.zeros((4, 4)), np.zeros((4, 4))) == pytest.approx(1.0)       # both empty = perfect
    assert M.box_iou([0, 0, 2, 2], [1, 1, 3, 3]) == pytest.approx(1 / 7)
    gts = [np.array([[0, 0, 10, 10.]]), np.array([[20, 20, 30, 30.]])]
    perfect = [(np.array([[0, 0, 10, 10.]]), np.array([.9])), (np.array([[20, 20, 30, 30.]]), np.array([.8]))]
    miss = [perfect[0], (np.array([[50, 50, 60, 60.]]), np.array([.8]))]
    assert M.ap50(perfect, gts) == pytest.approx(1.0) and M.ap50(miss, gts) == pytest.approx(0.5)

def test_split_pipeline_no_leakage(tmp_path):
    root = str(tmp_path / "d")
    run_module("medimg.synth_data", "--out", root, "--n", "240", "--seed", "3")
    run_module("medimg.prepare_splits", "--root", root)
    d = pd.read_csv(f"{root}/splits.csv"); rep = json.load(open(f"{root}/split_report.json"))
    assert rep["leakage_audit"] == "passed" and set(d.split) == {"train", "val", "test"}
    assert d.groupby("patient_id").split.nunique().max() == 1                  # patient-level split
    # every planted near-duplicate must sit in the same split as its closest original
    split = dict(zip(d.image_id, d.split)); img = lambda p: np.asarray(Image.open(f"{root}/{p}"), float)
    orig = d[~d.image_id.str.startswith("dup_")]; O = np.stack([img(p) for p in orig.image_path])
    dups = d[d.image_id.str.startswith("dup_")]; assert len(dups) > 5
    for _, r in dups.iterrows():
        j = np.abs(O - img(r.image_path)).mean((1, 2)).argmin()
        assert split[orig.image_id.iloc[j]] == r.split
    assert all(0.1 < v["n"] / len(d) < 0.8 for v in rep["per_split"].values())
    assert all(abs(v["pos_rate"] - d.label.mean()) < 0.15 for v in rep["per_split"].values())   # stratification
