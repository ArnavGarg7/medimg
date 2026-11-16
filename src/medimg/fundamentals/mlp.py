"""Step 4b/5: multilayer feed-forward network with hand-written backprop, four optimizers, L2, input standardisation,
and explicit parameter vs. hyper-parameter tracking."""
from dataclasses import dataclass, asdict
import numpy as np
from .neurons import ACTIVATIONS

@dataclass
class HyperParams:
    """Everything that is *chosen* (not learned). Saved with every experiment for reproducibility."""
    optimizer: str = "sgd"        # batch_gd | sgd | momentum | nesterov
    lr: float = 0.05
    momentum: float = 0.9
    batch_size: int = 16          # ignored by batch_gd
    epochs: int = 100
    l2: float = 0.0
    init: str = "xavier"          # zeros | small | xavier | he
    seed: int = 0
    def to_dict(self): return asdict(self)

class Optimizer:
    """batch_gd / sgd: plain step (they differ only in batch size). momentum: v=mu v - lr g; p+=v.
    nesterov (look-ahead, 'Sutskever form'): p += -mu v_prev + (1+mu) v."""
    def __init__(self, kind, lr, mu=0.9):
        if kind not in ("batch_gd", "sgd", "momentum", "nesterov"): raise ValueError(kind)
        self.kind, self.lr, self.mu, self.v = kind, lr, mu, None
    def update(self, params, grads):
        if self.v is None: self.v = [np.zeros_like(p) for p in params]
        for p, g, v in zip(params, grads, self.v):
            if self.kind in ("batch_gd", "sgd"): p -= self.lr * g
            elif self.kind == "momentum": v *= self.mu; v -= self.lr * g; p += v
            else:
                vp = v.copy(); v *= self.mu; v -= self.lr * g; p += -self.mu * vp + (1 + self.mu) * v

class MLP:
    def __init__(self, sizes, acts, init="xavier", seed=0, loss="bce"):
        assert len(acts) == len(sizes) - 1 and loss in ("bce", "mse")
        if loss == "bce" and acts[-1] != "sigmoid":
            raise ValueError("bce loss needs a sigmoid output unit (output must be a probability); use mse otherwise")
        rng = np.random.default_rng(seed)
        self.acts, self.loss_name = list(acts), loss
        self.W, self.b = [], []
        for i, o in zip(sizes[:-1], sizes[1:]):
            sd = {"zeros": 0.0, "small": 0.01, "xavier": np.sqrt(2 / (i + o)), "he": np.sqrt(2 / i)}[init]
            self.W.append(rng.normal(0, sd, (i, o)) if sd > 0 else np.zeros((i, o))); self.b.append(np.zeros(o))
        self.mu_x = self.sd_x = None                       # optional input normalisation (fit on TRAIN only)

    # --- parameters as a flat vector (for gradient checks / saving) ---
    def params(self): return [p for pair in zip(self.W, self.b) for p in pair]
    def get_flat(self): return np.concatenate([p.ravel() for p in self.params()])
    def set_flat(self, v):
        k = 0
        for p in self.params(): p[...] = v[k:k + p.size].reshape(p.shape); k += p.size

    def _norm(self, X): return X if self.mu_x is None else (X - self.mu_x) / self.sd_x

    def forward(self, X, cache=False):
        a, zs, As = self._norm(np.asarray(X, float)), [], []
        As.append(a)
        for W, b, name in zip(self.W, self.b, self.acts):
            z = a @ W + b; a = ACTIVATIONS[name][0](z); zs.append(z); As.append(a)
        return (a, zs, As) if cache else a

    def loss(self, X, Y, l2=0.0):
        P = self.forward(X); Y = np.asarray(Y, float)
        if self.loss_name == "mse": L = np.mean((P - Y) ** 2)
        else: P = np.clip(P, 1e-12, 1 - 1e-12); L = -np.sum(Y * np.log(P) + (1 - Y) * np.log(1 - P)) / len(X)
        return L + 0.5 * l2 * sum(np.sum(W * W) for W in self.W)

    def grads(self, X, Y, l2=0.0):
        P, zs, As = self.forward(X, cache=True); Y = np.asarray(Y, float); N = len(X)
        f_last, df_last = ACTIVATIONS[self.acts[-1]]
        if self.loss_name == "bce": delta = (P - Y) / N        # sigmoid + cross-entropy fused: dL/dz = (a - y)/N
        else: delta = 2 * (P - Y) / P.size * df_last(zs[-1], P)
        gW, gb = [None] * len(self.W), [None] * len(self.W)
        for l in reversed(range(len(self.W))):
            gW[l] = As[l].T @ delta + l2 * self.W[l]; gb[l] = delta.sum(0)
            if l > 0: delta = (delta @ self.W[l].T) * ACTIVATIONS[self.acts[l - 1]][1](zs[l - 1], As[l])
        return [g for pair in zip(gW, gb) for g in pair]

    def grads_flat(self, X, Y, l2=0.0): return np.concatenate([g.ravel() for g in self.grads(X, Y, l2)])

    def fit(self, X, Y, hp: HyperParams, X_val=None, Y_val=None, standardize=False, augment=None):
        """Mini-batch training. `augment(Xb, rng)` -> new Xb is applied to each training batch only. Returns history dict."""
        X, Y = np.asarray(X, float), np.asarray(Y, float)
        if standardize: self.mu_x, self.sd_x = X.mean(0), X.std(0) + 1e-8
        rng = np.random.default_rng(hp.seed); N = len(X)
        bs = N if hp.optimizer == "batch_gd" else min(hp.batch_size, N)
        opt = Optimizer(hp.optimizer, hp.lr, hp.momentum)
        hist = {"train_loss": [], "val_loss": [], "hp": hp.to_dict()}
        for _ in range(hp.epochs):
            idx = rng.permutation(N) if bs < N else np.arange(N)
            for s in range(0, N, bs):
                b = idx[s:s + bs]; xb = X[b]
                if augment is not None: xb = augment(xb, rng)
                opt.update(self.params(), self.grads(xb, Y[b], hp.l2))
            hist["train_loss"].append(self.loss(X, Y))
            if X_val is not None: hist["val_loss"].append(self.loss(X_val, Y_val))
        return hist

    def accuracy(self, X, Y): return float(np.mean((self.forward(X) >= 0.5) == (np.asarray(Y) >= 0.5)))

def grad_check(model, X, Y, l2=0.0, h=1e-5):
    """Max relative error between backprop and central-difference gradients."""
    analytic, theta = model.grads_flat(X, Y, l2), model.get_flat().copy(); worst = 0.0
    for i in range(len(theta)):
        t = theta.copy(); t[i] += h; model.set_flat(t); lp = model.loss(X, Y, l2)
        t[i] -= 2 * h; model.set_flat(t); lm = model.loss(X, Y, l2)
        num = (lp - lm) / (2 * h)
        worst = max(worst, abs(num - analytic[i]) / max(1e-8, abs(num) + abs(analytic[i])))
    model.set_flat(theta)
    return worst
