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
    a, rest = ap.parse_known_args(argv)
    if a.cmd == "fundamentals":
        from .fundamentals import experiments as E
        fn = {"tensor": E.demo_tensor, "svd": E.demo_svd, "neurons": E.demo_neurons, "optimizers": E.run_optimizers, "ablation": E.run_ablation, "learning": E.demo_learning, "all": E.run_all}[a.demo]
        if a.demo == "all" or a.demo == "neurons": E.demo_activations(a.out)
        print(json.dumps(fn(a.out), indent=2, default=str))
    elif a.cmd == "data":
        run_module("medimg.synth_data", "--out", a.root, "--n", str(a.n)); run_module("medimg.prepare_splits", "--root", a.root)

if __name__ == "__main__":
    main()
