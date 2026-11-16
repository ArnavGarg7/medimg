"""CLI:  python -m medimg <command> [options]"""
import argparse, json, runpy, sys

def run_module(name, *args):
    """Run `python -m name args` in-process (works whether or not the package is pip-installed)."""
    old = sys.argv; sys.argv = [name, *args]
    try: runpy.run_module(name, run_name="__main__")
    finally: sys.argv = old

def main(argv=None):
    ap = argparse.ArgumentParser(prog="medimg", description="Medical imaging triage workstation")
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fundamentals", help="steps 3-6 demos (NumPy)"); f.add_argument("demo", choices=["tensor", "svd", "neurons", "optimizers", "ablation", "learning", "all"]); f.add_argument("--out", default="out")
    d = sub.add_parser("data", help="synthetic data + dedup + leakage-safe split"); d.add_argument("--root", default="data/synth"); d.add_argument("--n", type=int, default=300)
    
    # Task D CLI Entry points
    inf = sub.add_parser("infer", help="Run multi-task inference pipeline on a single image")
    inf.add_argument("--image", required=True, help="Input image path")
    inf.add_argument("--out", default="runs/infer_out", help="Output directory")
    inf.add_argument("--threshold", type=float, default=0.518, help="Triage operating threshold")
    inf.add_argument("--size", type=int, default=128, help="Inference resolution")

    ev = sub.add_parser("evaluate", help="Run comprehensive clinical evaluation package")
    
    rob = sub.add_parser("robustness", help="Run clinical robustness stress tests")
    
    gc = sub.add_parser("gradcam", help="Run Grad-CAM interpretability and error analysis")
    
    gui = sub.add_parser("gui", help="Launch interactive Gradio triage workstation web UI")
    gui.add_argument("--port", type=int, default=7860, help="Server port")

    a, rest = ap.parse_known_args(argv)
    if a.cmd == "fundamentals":
        from .fundamentals import experiments as E
        fn = {"tensor": E.demo_tensor, "svd": E.demo_svd, "neurons": E.demo_neurons, "optimizers": E.run_optimizers, "ablation": E.run_ablation, "learning": E.demo_learning, "all": E.run_all}[a.demo]
        if a.demo == "all" or a.demo == "neurons": E.demo_activations(a.out)
        print(json.dumps(fn(a.out), indent=2, default=str))
    elif a.cmd == "data":
        run_module("medimg.synth_data", "--out", a.root, "--n", str(a.n)); run_module("medimg.prepare_splits", "--root", a.root)
    elif a.cmd == "infer":
        from .infer import run_inference
        res, _ = run_inference(a.image, out_dir=a.out, threshold=a.threshold, size=a.size)
        print(res["summary"])
        print(f"\nSaved outputs to: {a.out}")
    elif a.cmd == "evaluate":
        from .evaluate import evaluate_all
        evaluate_all()
    elif a.cmd == "robustness":
        from .robustness import run_robustness_suite
        run_robustness_suite()
    elif a.cmd == "gradcam":
        from .gradcam import run_error_analysis
        run_error_analysis()
    elif a.cmd == "gui":
        from .app_gradio import launch
        launch(port=a.port)

if __name__ == "__main__":
    main()
