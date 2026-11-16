"""Smoke and integration tests for Task D inference, evaluation, and interpretability pipelines."""
import os, pytest, numpy as np, torch
from PIL import Image

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() and not os.environ.get("RUN_SLOW_TESTS"),
                                reason="Skipped in fast test mode without GPU or slow flag")

def test_calibration_curve():
    from medimg.evaluate import compute_calibration_curve
    y_true = np.array([0, 0, 0, 1, 1, 1, 0, 1, 1, 0])
    y_prob = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9, 0.4, 0.6, 0.85, 0.15])
    calib = compute_calibration_curve(y_true, y_prob, n_bins=5)
    assert "ece" in calib
    assert 0.0 <= calib["ece"] <= 1.0
    assert len(calib["bin_centers"]) == 5

def test_gradcam_hook_and_shape():
    from medimg.gradcam import GradCAM
    from medimg.models import classifier
    model = classifier("resnet18", pretrained=False)
    target_layer = model.layer4[-1]
    gcam = GradCAM(model, target_layer)
    x = torch.randn(1, 3, 128, 128)
    cam, prob = gcam.generate(x)
    gcam.close()
    assert cam.shape == (128, 128)
    assert 0.0 <= cam.min() <= cam.max() <= 1.0 + 1e-6
    assert 0.0 <= prob <= 1.0

def test_infer_pipeline_runs(tmp_path):
    from medimg.infer import run_inference
    test_img = Image.new("RGB", (128, 128), color=(180, 120, 100))
    img_path = os.path.join(tmp_path, "sample.png")
    test_img.save(img_path)
    out_dir = os.path.join(tmp_path, "out")

    res, overlay = run_inference(img_path, out_dir=out_dir, threshold=0.518, size=128)
    assert res["priority"] in ("HIGH PRIORITY", "LOW PRIORITY")
    assert 0.0 <= res["risk_score"] <= 1.0
    assert res["threshold"] == 0.518
    assert "latency_ms" in res
    assert "total" in res["latency_ms"]
    assert os.path.exists(os.path.join(out_dir, "triage_overlay.png"))
    assert os.path.exists(os.path.join(out_dir, "triage_report.json"))
    assert os.path.exists(os.path.join(out_dir, "triage_summary.txt"))
