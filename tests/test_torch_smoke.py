import json, os, sys
import numpy as np
import pytest

torch = pytest.importorskip("torch")
torchvision = pytest.importorskip("torchvision")

def test_unet_output_spatial_shape():
    """U-Net output spatial shape must strictly equal input spatial shape."""
    from medimg.models import UNet

    model = UNet(in_ch=3, base=16, depth=4)
    model.eval()

    for size in (64, 128):
        x = torch.randn(2, 3, size, size)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 1, size, size), f"Expected {(2, 1, size, size)} but got {out.shape}"

def test_export_onnx_matches_pytorch_logits(tmp_path):
    """export_onnx output matches PyTorch logits within 1e-4 (via onnxruntime)."""
    ort = pytest.importorskip("onnxruntime")
    from medimg.models import resnet_classifier, UNet
    from medimg.export_onnx import main as export_main

    # 1. Classifier test
    cls_ckpt = str(tmp_path / "cls.pt")
    cls_onnx = str(tmp_path / "cls.onnx")
    m_cls = resnet_classifier("resnet18")
    torch.save(m_cls.state_dict(), cls_ckpt)

    old_argv = sys.argv
    sys.argv = ["export_onnx", "--task", "cls", "--ckpt", cls_ckpt, "--out", cls_onnx, "--size", "128"]
    try:
        export_main()
    finally:
        sys.argv = old_argv

    assert os.path.exists(cls_onnx)

    m_cls.eval()
    x_cls = torch.randn(1, 3, 128, 128)
    with torch.no_grad():
        pt_cls_out = m_cls(x_cls).numpy()

    sess_cls = ort.InferenceSession(cls_onnx)
    ort_cls_out = sess_cls.run(None, {"image": x_cls.numpy()})[0]
    max_cls_diff = float(np.max(np.abs(pt_cls_out - ort_cls_out)))
    assert max_cls_diff < 1e-4, f"Classifier ONNX discrepancy {max_cls_diff:.2e} exceeds 1e-4"

    # 2. U-Net test
    seg_ckpt = str(tmp_path / "seg.pt")
    seg_onnx = str(tmp_path / "seg.onnx")
    m_seg = UNet(3, 16)
    torch.save(m_seg.state_dict(), seg_ckpt)

    sys.argv = ["export_onnx", "--task", "seg", "--ckpt", seg_ckpt, "--out", seg_onnx, "--size", "128"]
    try:
        export_main()
    finally:
        sys.argv = old_argv

    assert os.path.exists(seg_onnx)

    m_seg.eval()
    x_seg = torch.randn(1, 3, 128, 128)
    with torch.no_grad():
        pt_seg_out = m_seg(x_seg).numpy()

    sess_seg = ort.InferenceSession(seg_onnx)
    ort_seg_out = sess_seg.run(None, {"image": x_seg.numpy()})[0]
    max_seg_diff = float(np.max(np.abs(pt_seg_out - ort_seg_out)))
    assert max_seg_diff < 1e-4, f"UNet ONNX discrepancy {max_seg_diff:.2e} exceeds 1e-4"

def test_single_epoch_trainers_smoke(tmp_path):
    """1-epoch run of each trainer (train_cls, train_seg, train_det) on a tiny generated dataset."""
    from medimg.__main__ import run_module

    root = str(tmp_path / "tiny_data")
    run_module("medimg.synth_data", "--out", root, "--n", "80", "--seed", "7")
    run_module("medimg.prepare_splits", "--root", root)

    # 1. train_cls 1 epoch
    out_cls = str(tmp_path / "cls_run")
    run_module("medimg.train_cls", "--root", root, "--epochs", "1", "--bs", "16", "--size", "64", "--out", out_cls)
    assert os.path.exists(os.path.join(out_cls, "report.json"))
    assert os.path.exists(os.path.join(out_cls, "best.pt"))
    rep_cls = json.load(open(os.path.join(out_cls, "report.json")))
    assert "test_auroc" in rep_cls

    # 2. train_seg 1 epoch
    out_seg = str(tmp_path / "seg_run")
    run_module("medimg.train_seg", "--root", root, "--epochs", "1", "--bs", "16", "--size", "64", "--base", "8", "--out", out_seg)
    assert os.path.exists(os.path.join(out_seg, "report.json"))
    assert os.path.exists(os.path.join(out_seg, "best.pt"))
    rep_seg = json.load(open(os.path.join(out_seg, "report.json")))
    assert "test_dice" in rep_seg

    # 3. train_det 1 epoch
    out_det = str(tmp_path / "det_run")
    run_module("medimg.train_det", "--root", root, "--epochs", "1", "--bs", "8", "--size", "64", "--out", out_det)
    assert os.path.exists(os.path.join(out_det, "report.json"))
    rep_det = json.load(open(os.path.join(out_det, "report.json")))
    assert "test_ap50" in rep_det
