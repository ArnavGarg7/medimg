import json, os, numpy as np, pytest, torch
from PIL import Image

def test_datasets_and_collate(tmp_path):
    from medimg.datasets import MedDataset, collate
    from medimg.__main__ import run_module
    from torch.utils.data import DataLoader

    root = str(tmp_path / "data")
    run_module("medimg.synth_data", "--out", root, "--n", "120", "--seed", "1")
    run_module("medimg.prepare_splits", "--root", root)

    ds_rgb = MedDataset(root, "train", size=64, augment=True, rgb=True, seed=42)
    assert len(ds_rgb) > 0
    item = ds_rgb[0]
    assert item["x"].shape == (3, 64, 64)
    assert item["mask"].shape == (1, 64, 64)
    assert isinstance(item["label"].item(), float)
    assert item["boxes"].ndim == 2 and item["boxes"].shape[1] == 4

    ds_gray = MedDataset(root, "train", size=64, augment=False, rgb=False, seed=42)
    item_gray = ds_gray[0]
    assert item_gray["x"].shape == (1, 64, 64)

    # Collate check
    loader = DataLoader(ds_rgb, batch_size=4, collate_fn=collate)
    b = next(iter(loader))
    assert b["x"].shape == (4, 3, 64, 64)
    assert b["mask"].shape == (4, 1, 64, 64)
    assert b["label"].shape == (4,)
    assert len(b["boxes"]) == 4

def test_models_architectures_and_losses():
    from medimg.models import resnet_classifier, UNet, retinanet, dice_bce_loss

    x = torch.randn(2, 3, 64, 64)
    
    # ResNet Classifier
    cls_model = resnet_classifier("resnet18", pretrained=False)
    y_cls = cls_model(x)
    assert y_cls.shape == (2, 1)
    loss_cls = y_cls.sum()
    loss_cls.backward()

    # UNet
    unet = UNet(in_ch=3, base=8, depth=3)
    y_seg = unet(x)
    assert y_seg.shape == (2, 1, 64, 64)
    target_mask = torch.randint(0, 2, (2, 1, 64, 64)).float()
    loss_seg = dice_bce_loss(y_seg, target_mask)
    assert loss_seg.ndim == 0 and not torch.isnan(loss_seg)
    loss_seg.backward()

    # RetinaNet forward with mixed and empty targets
    det_model = retinanet(num_classes=2, pretrained_backbone=False)
    det_model.train()
    targets = [
        {"boxes": torch.tensor([[5., 5., 20., 20.]]), "labels": torch.tensor([1], dtype=torch.int64)},
        {"boxes": torch.zeros((0, 4)), "labels": torch.zeros(0, dtype=torch.int64)}
    ]
    det_losses = det_model(list(x), targets)
    assert "classification" in det_losses and "bbox_regression" in det_losses
    total_det_loss = sum(det_losses.values())
    total_det_loss.backward()

def test_export_onnx_and_ort_parity(tmp_path):
    ort = pytest.importorskip("onnxruntime")
    from medimg.models import resnet_classifier, UNet
    from medimg.export_onnx import main as export_main
    import sys

    ckpt_path = str(tmp_path / "model.pt")
    onnx_path = str(tmp_path / "model.onnx")

    m = resnet_classifier("resnet18")
    torch.save(m.state_dict(), ckpt_path)

    old_argv = sys.argv
    sys.argv = ["export_onnx", "--task", "cls", "--ckpt", ckpt_path, "--out", onnx_path, "--size", "64"]
    try:
        export_main()
    finally:
        sys.argv = old_argv

    assert os.path.exists(onnx_path)

    # Parity check
    m.eval()
    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        pt_out = m(x).numpy()

    sess = ort.InferenceSession(onnx_path)
    ort_out = sess.run(None, {"image": x.numpy()})[0]
    np.testing.assert_allclose(pt_out, ort_out, rtol=1e-3, atol=1e-3)

def test_training_pipelines_execution(tmp_path):
    from medimg.__main__ import run_module

    root = str(tmp_path / "data")
    run_module("medimg.synth_data", "--out", root, "--n", "80", "--seed", "1")
    run_module("medimg.prepare_splits", "--root", root)

    # 1. train_cls
    out_cls = str(tmp_path / "runs_cls")
    run_module("medimg.train_cls", "--root", root, "--epochs", "1", "--bs", "16", "--size", "64", "--out", out_cls)
    assert os.path.exists(os.path.join(out_cls, "report.json"))
    assert os.path.exists(os.path.join(out_cls, "best.pt"))
    rep_cls = json.load(open(os.path.join(out_cls, "report.json")))
    assert "test_auroc" in rep_cls and "latency_ms_median_p95" in rep_cls

    # 2. train_seg
    out_seg = str(tmp_path / "runs_seg")
    run_module("medimg.train_seg", "--root", root, "--epochs", "1", "--bs", "16", "--size", "64", "--base", "8", "--out", out_seg)
    assert os.path.exists(os.path.join(out_seg, "report.json"))
    assert os.path.exists(os.path.join(out_seg, "best.pt"))
    rep_seg = json.load(open(os.path.join(out_seg, "report.json")))
    assert "test_dice" in rep_seg and "test_iou" in rep_seg

    # 3. train_det
    out_det = str(tmp_path / "runs_det")
    run_module("medimg.train_det", "--root", root, "--epochs", "1", "--bs", "8", "--size", "64", "--out", out_det)
    assert os.path.exists(os.path.join(out_det, "report.json"))
    rep_det = json.load(open(os.path.join(out_det, "report.json")))
    assert "test_ap50" in rep_det and "latency_ms_median_p95" in rep_det

