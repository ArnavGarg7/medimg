"""Step 6: Hebbian, modified Hebbian (Oja), competitive, error-correction (delta/perceptron) and a Boltzmann-style RBM."""
import numpy as np
from .neurons import sigmoid

def hebbian(X, epochs=8, eta=1e-3, seed=0):
    """Plain Hebb: dw = eta*y*x, y=w.x  -> |w| grows without bound. Returns (w, norm history)."""
    w = np.random.default_rng(seed).normal(0, 0.1, X.shape[1]); norms = []
    for _ in range(epochs):
        for x in X: w += eta * (w @ x) * x
        norms.append(float(np.linalg.norm(w)))
    return w, norms

def oja(X, epochs=30, eta=2e-3, seed=0):
    """Oja's modified Hebb: dw = eta*y*(x - y*w). Converges to the unit-norm first principal component."""
    w = np.random.default_rng(seed).normal(0, 0.1, X.shape[1])
    for _ in range(epochs):
        for x in X:
            y = w @ x; w += eta * y * (x - y * w)
    return w

def competitive(X, k=2, epochs=10, eta=0.05, seed=0):
    """Winner-take-all: only the nearest prototype moves toward the sample."""
    rng = np.random.default_rng(seed); P = X[rng.choice(len(X), k, replace=False)].copy()
    for _ in range(epochs):
        for x in X:
            j = np.argmin(((P - x) ** 2).sum(1)); P[j] += eta * (x - P[j])
    return P

def delta_rule(X, y, eta=0.1, max_epochs=100):
    """Error-correction (perceptron) learning. Returns (w, b, epochs_to_converge or None)."""
    w, b = np.zeros(X.shape[1]), 0.0
    for e in range(1, max_epochs + 1):
        errs = 0
        for x, t in zip(X, y):
            err = t - int(w @ x + b >= 0)
            if err: errs += 1; w += eta * err * x; b += eta * err
        if not errs: return w, b, e
    return w, b, None

class RBM:
    """Binary restricted Boltzmann machine trained with 1-step contrastive divergence. Energy:
    E(v,h) = -a.v - b.h - v^T W h."""
    def __init__(self, nv, nh, seed=0):
        self.rng = np.random.default_rng(seed)
        self.W = self.rng.normal(0, 0.1, (nv, nh)); self.a = np.zeros(nv); self.b = np.zeros(nh)
    def hidden_p(self, v): return sigmoid(v @ self.W + self.b)
    def visible_p(self, h): return sigmoid(h @ self.W.T + self.a)
    def energy(self, v, h): return -(v @ self.a) - (h @ self.b) - np.einsum("ni,ij,nj->n", v, self.W, h)
    def train_epoch(self, V, lr=0.05):
        err = 0.0
        for v0 in V:
            h0p = self.hidden_p(v0); h0 = (self.rng.random(h0p.shape) < h0p).astype(float)
            v1p = self.visible_p(h0); h1p = self.hidden_p(v1p)
            self.W += lr * (np.outer(v0, h0p) - np.outer(v1p, h1p))
            self.a += lr * (v0 - v1p); self.b += lr * (h0p - h1p); err += np.mean((v0 - v1p) ** 2)
        return err / len(V)
    def mean_energy(self, V): return float(self.energy(V, self.hidden_p(V)).mean())
