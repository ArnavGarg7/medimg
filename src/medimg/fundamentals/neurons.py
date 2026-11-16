"""Step 4a: activations, McCulloch-Pitts neuron, sigmoid neuron."""
import numpy as np

def step(z): return (z >= 0).astype(float)
def sigmoid(z): return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
def tanh(z): return np.tanh(z)
def relu(z): return np.maximum(0.0, z)
def linear(z): return z

# derivative as a function of (z, a=act(z))
def _no_grad(z, a): raise ValueError("step activation has zero/undefined gradient: cannot backpropagate")
ACTIVATIONS = {
    "step": (step, _no_grad),
    "sigmoid": (sigmoid, lambda z, a: a * (1 - a)),
    "tanh": (tanh, lambda z, a: 1 - a * a),
    "relu": (relu, lambda z, a: (z > 0).astype(float)),
    "linear": (linear, lambda z, a: np.ones_like(z)),
}

class McCullochPitts:
    """Binary threshold unit: fires iff sum(w*x) >= theta, with fixed integer weights (+1 excitatory, -1 inhibitory)."""
    def __init__(self, weights, theta): self.w, self.theta = np.asarray(weights), theta
    def __call__(self, x): return int(np.dot(self.w, x) >= self.theta)
    @classmethod
    def AND(cls, n=2): return cls([1] * n, n)
    @classmethod
    def OR(cls, n=2): return cls([1] * n, 1)
    @classmethod
    def NOT(cls): return cls([-1], 0)

class SigmoidNeuron:
    """Single logistic unit trained by gradient descent on cross-entropy (dL/dz = a - y)."""
    def __init__(self, n_in, seed=0):
        self.w = np.random.default_rng(seed).normal(0, 0.1, n_in); self.b = 0.0
    def forward(self, X): return sigmoid(np.atleast_2d(X) @ self.w + self.b)
    def fit(self, X, y, epochs=2000, lr=0.5):
        X, y = np.asarray(X, float), np.asarray(y, float)
        for _ in range(epochs):
            err = self.forward(X) - y
            self.w -= lr * X.T @ err / len(y); self.b -= lr * err.mean()
        return self
