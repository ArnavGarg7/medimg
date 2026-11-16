"""Runnable demonstrations for steps 3-6. Each returns a dict of results and writes figures/JSON to `out`."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from . import tensor_ops as T, learning_rules as LR
from .neurons import McCullochPitts, SigmoidNeuron, ACTIVATIONS
from .mlp import MLP, HyperParams, grad_check

SEEDS = (0, 1, 2, 3, 4)

def _fig(out, name):
    os.makedirs(os.path.join(out, "figures"), exist_ok=True)
    p = os.path.join(out, "figures", name); plt.tight_layout(); plt.savefig(p, dpi=110); plt.close(); return p

def make_circles(n, rng, noise=0.1):
    y = np.arange(n) % 2; r = np.where(y == 1, 0.4, 1.0) + rng.uniform(-noise, noise, n); t = rng.uniform(0, 2 * np.pi, n)
    return np.c_[r * np.cos(t), r * np.sin(t)], y.reshape(-1, 1).astype(float)

def _ms(v): return f"{np.mean(v):.3f} +/- {np.std(v):.3f}"

# ---------------------------------------------------------------- step 3
def demo_tensor(out="out", seed=0):
    rng = np.random.default_rng(seed); res = {}
    A, row, col = np.arange(1., 7).reshape(2, 3), np.array([10., 20, 30]), np.array([[100.], [200.]])
    res["A+row"] = T.broadcast_apply(A, row, lambda x, y: x + y).tolist(); res["A*col"] = T.broadcast_apply(A, col, lambda x, y: x * y).tolist()
    res["inner([1,2,3],[4,5,6])"] = T.inner([1, 2, 3], [4, 5, 6])
    res["reshape(6)->(2,3) is a view"] = T.reshape_view_demo(np.arange(6), (2, 3))[1]
    x = rng.normal(size=5); f = lambda t: np.sum(np.sin(t) * t ** 2)
    res["numeric_vs_analytic_grad_rel_err"] = T.rel_error(T.numeric_grad(f, x), np.cos(x) * x ** 2 + 2 * x * np.sin(x))
    fig, ax = plt.subplots(1, 4, figsize=(11, 2.6))
    for a, (m, t) in zip(ax, [(A, "A (2x3)"), (row[None], "row (3,)->(1,3)"), (np.array(res["A+row"]), "A + row"), (np.array(res["A*col"]), "A * col(2,1)")]):
        a.imshow(m, cmap="viridis"); a.set_title(t, fontsize=9)
        for (i, j), v in np.ndenumerate(m): a.text(j, i, f"{v:g}", ha="center", va="center", color="w", fontsize=8)
        a.set_xticks([]); a.set_yticks([])
    _fig(out, "broadcasting.png")
    # 2D and 3D convolution visualisation
    yy, xx = np.mgrid[0:64, 0:64]; img = np.exp(-((yy - 30) ** 2 + (xx - 34) ** 2) / 150) + 0.05 * np.sin(0.7 * xx)
    sobel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1.]]); blur = np.ones((5, 5)) / 25
    vol = np.exp(-((np.arange(16)[:, None, None] - 8) ** 2 + (yy[None, :32, :32] - 16) ** 2 + (xx[None, :32, :32] - 16) ** 2) / 40)
    k3 = np.ones((3, 3, 3)) / 27; v3 = T.conv3d_valid(vol, k3)
    fig, ax = plt.subplots(1, 5, figsize=(14, 2.9))
    for a, (m, t) in zip(ax, [(img, "2D input"), (T.conv2d_valid(img, sobel), "Sobel-x (edges)"), (T.conv2d_valid(img, blur), "5x5 box blur"),
                              (vol[8], "3D volume, slice 8"), (v3[7], "3x3x3 mean filter, slice 7")]):
        a.imshow(m, cmap="gray"); a.set_title(t, fontsize=9); a.axis("off")
    _fig(out, "convolution_2d_3d.png"); res["conv3d_out_shape"] = list(v3.shape)
    return res

def demo_svd(out="out", seed=0):
    rng = np.random.default_rng(seed); yy, xx = np.mgrid[0:64, 0:64]
    img = rng.normal(0, 0.05, (64, 64)) + np.exp(-((yy - 32) ** 2 + (xx - 32) ** 2) / 200) + 0.5 * ((yy > 20) & (yy < 44) & (xx > 10) & (xx < 22))
    U, S, V = T.jacobi_svd(img); res = {"top_singular_values": S[:6].round(3).tolist(), "max_abs_diff_vs_lapack": float(np.abs(S - np.linalg.svd(img, compute_uv=False)).max())}
    ks = [1, 3, 8, 20]; fig, ax = plt.subplots(1, 6, figsize=(16, 3))
    ax[0].imshow(img, cmap="gray"); ax[0].set_title("original"); ax[0].axis("off")
    for a, k in zip(ax[1:5], ks):
        rec = T.svd_reconstruct(U, S, V, k); e = np.linalg.norm(img - rec) / np.linalg.norm(img)
        a.imshow(rec, cmap="gray"); a.set_title(f"rank {k}  (err {e:.2f})", fontsize=9); a.axis("off"); res[f"rank{k}_rel_err"] = float(e)
    ax[5].semilogy(S, "o-", ms=3); ax[5].set_title("singular spectrum"); ax[5].set_xlabel("index")
    _fig(out, "svd.png"); return res

# ---------------------------------------------------------------- step 4
def demo_neurons(out="out", seed=0):
    res = {}; X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], float)
    AND, OR, NOT = McCullochPitts.AND(), McCullochPitts.OR(), McCullochPitts.NOT()
    res["MP_AND"] = [AND(x) for x in X]; res["MP_OR"] = [OR(x) for x in X]; res["MP_NOT"] = [NOT([0]), NOT([1])]
    for name, y in [("AND", [0, 0, 0, 1]), ("XOR", [0, 1, 1, 0])]:
        res[f"single_sigmoid_{name}_correct_of_4"] = int(((SigmoidNeuron(2, seed).fit(X, y, 5000).forward(X) >= .5) == y).sum())
    m = MLP([2, 4, 1], ["tanh", "sigmoid"], seed=seed); Y = np.array([[0], [1], [1], [0.]])
    m.fit(X, Y, HyperParams(optimizer="momentum", lr=.3, batch_size=4, epochs=2000)); res["MLP_2-4-1_XOR_accuracy"] = m.accuracy(X, Y)
    rng = np.random.default_rng(seed); Xg, Yg = rng.normal(size=(6, 4)), (np.arange(6) % 2).reshape(-1, 1).astype(float)
    res["backprop_gradcheck_max_rel_err"] = grad_check(MLP([4, 5, 3, 1], ["tanh", "sigmoid", "sigmoid"], seed=seed), Xg, Yg, 1e-3)
    return res

def demo_activations(out="out"):
    z = np.linspace(-5, 5, 400); fig, ax = plt.subplots(1, 2, figsize=(10, 3.2))
    for n in ("step", "sigmoid", "tanh", "relu"):
        f, df = ACTIVATIONS[n]; a = f(z); ax[0].plot(z, a, label=n)
        if n != "step": ax[1].plot(z, df(z, a), label=n)
    ax[0].set_title("activation"); ax[1].set_title("derivative (step: 0 a.e. -> cannot learn by gradient)"); ax[0].legend(); ax[1].legend()
    _fig(out, "activations.png")

# ---------------------------------------------------------------- step 5
def run_optimizers(out="out", epochs=150):
    cfg = {"batch_gd": dict(lr=0.5, batch_size=400), "sgd": dict(lr=0.05), "momentum": dict(lr=0.05), "nesterov": dict(lr=0.05)}
    res, curves = {}, {}
    for name, kw in cfg.items():
        fin, acc, e50, cs = [], [], [], []
        for s in SEEDS:
            rng = np.random.default_rng(100 + s); X, Y = make_circles(400, rng)
            m = MLP([2, 16, 16, 1], ["tanh", "tanh", "sigmoid"], seed=s)
            h = m.fit(X, Y, HyperParams(optimizer=name, epochs=epochs, seed=s, **kw))
            tl = np.array(h["train_loss"]); fin.append(tl[-1]); acc.append(m.accuracy(X, Y)); cs.append(tl)
            hit = np.where(tl < 0.05)[0]; e50.append(int(hit[0]) + 1 if len(hit) else epochs + 1)
        curves[name] = np.mean(cs, 0)
        res[name] = {"final_loss": _ms(fin), "train_acc": _ms(acc), "epochs_to_loss<0.05 (151=never)": _ms(e50), "hyperparams": {**HyperParams(optimizer=name, epochs=epochs, **kw).to_dict()}}
    plt.figure(figsize=(6.5, 4))
    for n, c in curves.items(): plt.semilogy(range(1, epochs + 1), c, label=n)
    plt.xlabel("epoch"); plt.ylabel("train loss (mean of 5 seeds)"); plt.legend(); plt.title("Optimizer comparison (concentric rings)")
    _fig(out, "optimizers.png"); return res

def _overfit_trial(seed, hp_kw, standardize=False, augment=None, scale2=1.0, init="xavier", acts=("tanh", "tanh", "sigmoid")):
    rng = np.random.default_rng(1000 + seed); Xt, Yt = make_circles(40, rng, 0.45); Xv, Yv = make_circles(2000, rng, 0.45)
    Xt = Xt * [1, scale2]; Xv = Xv * [1, scale2]
    m = MLP([2, 32, 32, 1], list(acts), init=init, seed=seed)
    m.fit(Xt, Yt, HyperParams(optimizer="momentum", lr=0.05, batch_size=8, epochs=400, seed=seed, **hp_kw), standardize=standardize, augment=augment)
    return m.accuracy(Xt, Yt), m.accuracy(Xv, Yv)

def _rot_aug(Xb, rng):                       # label-preserving for rings; applied to the TRAIN batches only
    t = rng.uniform(0, 2 * np.pi, len(Xb)); c, s = np.cos(t), np.sin(t)
    return np.c_[c * Xb[:, 0] - s * Xb[:, 1], s * Xb[:, 0] + c * Xb[:, 1]]

def run_ablation(out="out"):
    res = {}
    # activation (hidden) comparison
    rows = {}
    for act in ("sigmoid", "tanh", "relu", "step"):
        accs, err = [], None
        for s in SEEDS:
            rng = np.random.default_rng(100 + s); X, Y = make_circles(400, rng)
            m = MLP([2, 16, 1], [act, "sigmoid"], seed=s)
            try: m.fit(X, Y, HyperParams(optimizer="nesterov", epochs=150, seed=s)); accs.append(m.accuracy(X, Y))
            except ValueError as e: err = str(e); break
        rows[act] = _ms(accs) if err is None else f"cannot train: {err}"
    res["hidden_activation_train_acc"] = rows
    # initialisation
    rows = {}
    for init in ("zeros", "small", "xavier", "he"):
        fin = []
        for s in SEEDS:
            rng = np.random.default_rng(100 + s); X, Y = make_circles(400, rng)
            m = MLP([2, 16, 16, 1], ["tanh", "tanh", "sigmoid"], init=init, seed=s)
            fin.append(m.fit(X, Y, HyperParams(optimizer="momentum", epochs=150, seed=s))["train_loss"][-1])
        rows[init] = _ms(fin)
    res["init_final_train_loss (0.693 = no learning)"] = rows
    # generalisation: regularisation / augmentation / normalisation (train 40 noisy pts, test 2000)
    def row(**kw):
        r = np.array([_overfit_trial(s, **kw) for s in SEEDS]); return {"train_acc": _ms(r[:, 0]), "heldout_acc": _ms(r[:, 1])}
    res["generalisation"] = {"baseline": row(hp_kw={}), "L2=1e-2": row(hp_kw={"l2": 1e-2}), "L2=5e-2": row(hp_kw={"l2": 5e-2}),
                             "rotation augmentation": row(hp_kw={}, augment=_rot_aug)}
    res["normalisation (feature 2 scaled x50)"] = {"raw inputs": row(hp_kw={}, scale2=50.0), "standardised inputs": row(hp_kw={}, scale2=50.0, standardize=True)}
    return res

# ---------------------------------------------------------------- step 6
def demo_learning(out="out", seed=0):
    rng = np.random.default_rng(seed); n = 300; s, e = 2 * rng.normal(size=n), 0.3 * rng.normal(size=n)
    X = np.c_[0.866 * s - 0.5 * e, 0.5 * s + 0.866 * e]; res = {}
    _, norms = LR.hebbian(X); w = LR.oja(X); _, _, V = T.jacobi_svd(X)
    res["hebb_weight_norm_epoch1_to_8"] = [round(norms[0], 3), round(norms[-1], 1)]
    res["oja_norm"] = float(np.linalg.norm(w)); res["oja_|cos|_with_SVD_PC1"] = float(abs(w @ V[:, 0]))
    C = np.r_[rng.normal(-2, .4, (100, 2)), rng.normal(2, .4, (100, 2))]; P = LR.competitive(C, 2, seed=seed)
    res["competitive_prototypes"] = P.round(2).tolist()
    OR = np.array([[0, 0], [0, 1], [1, 0], [1, 1.]]); res["delta_rule_epochs_OR"] = LR.delta_rule(OR, [0, 1, 1, 1])[2]; res["delta_rule_epochs_XOR"] = LR.delta_rule(OR, [0, 1, 1, 0])[2]
    Vb = np.array([[1, 1, 1, 0, 0, 0], [0, 0, 0, 1, 1, 1]] * 10, float); r = LR.RBM(6, 4, seed); err, en = [], []
    for _ in range(200): err.append(r.train_epoch(Vb)); en.append(r.mean_energy(Vb))
    res["rbm_recon_error_first_last"] = [round(err[0], 4), round(err[-1], 6)]; res["rbm_mean_energy_first_last"] = [round(en[0], 3), round(en[-1], 3)]
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.5))
    ax[0].plot(norms, "o-"); ax[0].set_yscale("log"); ax[0].set_title("Hebbian |w| (unbounded)")
    ax[1].scatter(*C.T, s=5, c="gray"); ax[1].scatter(*P.T, c="r", marker="x", s=120); ax[1].set_title("competitive prototypes")
    ax[2].plot(err, label="recon. error"); ax[2].set_xlabel("epoch"); a2 = ax[2].twinx(); a2.plot(en, "r", label="mean energy"); ax[2].set_title("RBM (CD-1)")
    _fig(out, "learning_rules.png"); return res

def run_all(out="out"):
    demo_activations(out); allres = {"tensor": demo_tensor(out), "svd": demo_svd(out), "neurons": demo_neurons(out), "optimizers": run_optimizers(out),
                                    "ablation": run_ablation(out), "learning": demo_learning(out)}
    os.makedirs(out, exist_ok=True); json.dump(allres, open(os.path.join(out, "fundamentals_results.json"), "w"), indent=2, default=str); return allres
