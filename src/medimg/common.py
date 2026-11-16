import json, os, random, subprocess, sys, time
import numpy as np, torch

def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

def device(): return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_meta(args):
    """Reproducibility record saved next to every checkpoint."""
    try: git = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception: git = None
    return dict(args=vars(args), torch=torch.__version__, python=sys.version.split()[0], git=git, time=time.strftime("%F %T"))

def save_json(obj, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True); json.dump(obj, open(path, "w"), indent=2)
