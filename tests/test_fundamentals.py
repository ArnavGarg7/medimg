import numpy as np, pytest
from medimg.fundamentals import tensor_ops as T
from medimg.fundamentals.neurons import McCullochPitts, SigmoidNeuron, ACTIVATIONS
from medimg.fundamentals.mlp import MLP, HyperParams, Optimizer, grad_check
from medimg.fundamentals import learning_rules as LR

rng = np.random.default_rng(0)

# ---------------- tensor sandbox ----------------
@pytest.mark.parametrize("sa,sb", [((2, 3), (3,)), ((2, 3), (2, 1)), ((3, 1), (1, 4)), ((5, 1, 4), (3, 1)), ((), (2, 2))])
def test_broadcast_matches_numpy(sa, sb):
    a, b = rng.normal(size=sa), rng.normal(size=sb)
    assert T.broadcast_shape(sa, sb) == np.broadcast(a, b).shape
    np.testing.assert_allclose(T.broadcast_apply(a, b, lambda x, y: x + 2 * y), a + 2 * b)

def test_broadcast_incompatible():
    with pytest.raises(ValueError): T.broadcast_shape((2, 3), (4,))

def test_inner_matmul_reshape():
    u, v = np.arange(1., 4), np.arange(4., 7)
    assert T.inner(u, v) == 32.0
    A, B = rng.normal(size=(4, 5)), rng.normal(size=(5, 3))
    np.testing.assert_allclose(T.matmul_loops(A, B), A @ B)
    y, shared = T.reshape_view_demo(np.arange(6), (2, 3)); assert shared and y.shape == (2, 3)
    with pytest.raises(ValueError): np.arange(6).reshape(4, 2)

def test_conv_shapes_and_values():
    img = np.arange(9.).reshape(3, 3); k = np.array([[1., 0], [0, 1]])
    np.testing.assert_allclose(T.conv2d_valid(img, k), [[4, 6], [10, 12]])
    vol = rng.normal(size=(6, 7, 8)); k3 = rng.normal(size=(3, 3, 3))
    out = T.conv3d_valid(vol, k3); assert out.shape == (4, 5, 6)
    np.testing.assert_allclose(out[1, 2, 3], np.sum(vol[1:4, 2:5, 3:6] * k3))

@pytest.mark.parametrize("shape", [(8, 5), (5, 8), (6, 6), (20, 3)])
def test_jacobi_svd(shape):
    A = rng.normal(size=shape); U, S, V = T.jacobi_svd(A)
    np.testing.assert_allclose(S, np.linalg.svd(A, compute_uv=False), atol=1e-10)     # vs LAPACK
    np.testing.assert_allclose(T.svd_reconstruct(U, S, V, len(S)), A, atol=1e-9)
    np.testing.assert_allclose(U.T @ U, np.eye(len(S)), atol=1e-8)
    np.testing.assert_allclose(V.T @ V, np.eye(len(S)), atol=1e-8)
    assert np.all(np.diff(S) <= 1e-12)

def test_svd_rank_deficient():
    A = np.outer(rng.normal(size=6), rng.normal(size=4)); U, S, V = T.jacobi_svd(A)
    assert S[0] > 1 and np.all(S[1:] < 1e-10)
    np.testing.assert_allclose(T.svd_reconstruct(U, S, V, 1), A, atol=1e-9)

def test_numeric_grad():
    x = rng.normal(size=(3, 2)); f = lambda t: np.sum(np.sin(t) * t ** 2)
    analytic = np.cos(x) * x ** 2 + 2 * x * np.sin(x)
    assert T.rel_error(T.numeric_grad(f, x), analytic) < 1e-6

# ---------------- neurons ----------------
def test_mcculloch_pitts():
    AND, OR, NOT = McCullochPitts.AND(), McCullochPitts.OR(), McCullochPitts.NOT()
    for a in (0, 1):
        assert NOT([a]) == 1 - a
        for b in (0, 1): assert AND([a, b]) == (a & b) and OR([a, b]) == (a | b)

def test_sigmoid_neuron_and_xor_limit():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], float)
    n = SigmoidNeuron(2).fit(X, [0, 0, 0, 1]); assert ((n.forward(X) >= .5) == [0, 0, 0, 1]).all()
    n = SigmoidNeuron(2).fit(X, [0, 1, 1, 0], epochs=5000); assert ((n.forward(X) >= .5) == [0, 1, 1, 0]).sum() < 4  # not separable

def test_step_not_trainable():
    with pytest.raises(ValueError): ACTIVATIONS["step"][1](np.zeros(2), np.zeros(2))

# ---------------- MLP ----------------
def _data(n=6, d=4):
    X = rng.normal(size=(n, d)); Y = (np.arange(n) % 2).reshape(-1, 1).astype(float); return X, Y

@pytest.mark.parametrize("acts,loss,l2", [
    (["tanh", "sigmoid", "sigmoid"], "bce", 0.0), (["tanh", "sigmoid", "sigmoid"], "bce", 1e-2),
    (["relu", "tanh", "linear"], "mse", 0.0), (["tanh", "relu", "tanh"], "mse", 1e-2)])
def test_backprop_gradcheck(acts, loss, l2):
    X, Y = _data(); m = MLP([4, 5, 3, 1], acts, "xavier", seed=1, loss=loss)
    assert grad_check(m, X, Y, l2) < 1e-6

def test_bce_requires_sigmoid_output():
    with pytest.raises(ValueError): MLP([4, 3, 1], ["tanh", "tanh"], loss="bce")

def test_gradcheck_detects_wrong_gradient():
    X, Y = _data(); m = MLP([4, 5, 1], ["tanh", "sigmoid"], seed=1)
    good = m.grads_flat(X, Y); m.grads_flat = lambda *a, **k: good * 1.05     # corrupt on purpose
    assert grad_check(m, X, Y) > 1e-3

def test_gradients_vs_torch_autograd():
    try: import torch
    except (ImportError, OSError): pytest.skip("torch not usable in this environment")
    X, Y = _data(); m = MLP([4, 5, 1], ["tanh", "sigmoid"], seed=2)
    Ws = [torch.tensor(p, dtype=torch.float64, requires_grad=True) for p in m.params()]
    x, y = torch.tensor(X), torch.tensor(Y)
    out = torch.sigmoid(torch.tanh(x @ Ws[0] + Ws[1]) @ Ws[2] + Ws[3])
    torch.nn.functional.binary_cross_entropy(out, y).backward()
    ref = np.concatenate([w.grad.numpy().ravel() for w in Ws])
    np.testing.assert_allclose(m.grads_flat(X, Y), ref, atol=1e-10)

XOR_X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], float); XOR_Y = np.array([[0], [1], [1], [0]], float)

@pytest.mark.parametrize("opt", ["momentum", "nesterov"])
def test_xor_learns(opt):
    m = MLP([2, 4, 1], ["tanh", "sigmoid"], seed=42)
    h = m.fit(XOR_X, XOR_Y, HyperParams(optimizer=opt, lr=0.3, batch_size=4, epochs=3000))
    assert h["train_loss"][-1] < 0.05 < h["train_loss"][0] and m.accuracy(XOR_X, XOR_Y) == 1.0

def test_zero_init_symmetry():
    m = MLP([2, 3, 1], ["tanh", "sigmoid"], init="zeros")
    m.fit(XOR_X, XOR_Y, HyperParams(optimizer="sgd", lr=.3, batch_size=4, epochs=50))
    np.testing.assert_allclose(m.W[0][:, 0], m.W[0][:, 1], atol=1e-12)    # hidden units never differentiate

def test_optimizer_update_rules():
    p, g = [np.array([1.0])], [np.array([2.0])]
    o = Optimizer("momentum", 0.1, 0.9); o.update(p, g); assert np.isclose(p[0][0], 0.8)
    o.update(p, g); assert np.isclose(p[0][0], 0.8 + (0.9 * -0.2 - 0.2))
    p = [np.array([1.0])]; o = Optimizer("nesterov", 0.1, 0.9); o.update(p, g)
    assert np.isclose(p[0][0], 1 + (1.9) * -0.2)       # v=-0.2 ; p += -mu*0 + (1+mu)*v
    with pytest.raises(ValueError): Optimizer("adam", 0.1)

def test_batch_gd_is_deterministic_across_seeds():
    r = []
    for s in (1, 2):
        m = MLP([2, 4, 1], ["tanh", "sigmoid"], seed=7)
        r.append(m.fit(XOR_X, XOR_Y, HyperParams(optimizer="batch_gd", lr=.3, epochs=20, seed=s))["train_loss"][-1])
    assert r[0] == r[1]

# ---------------- learning rules ----------------
def _corr_data():
    n = 300; s, e = 2 * rng.normal(size=n), 0.3 * rng.normal(size=n)
    return np.c_[0.866 * s - 0.5 * e, 0.5 * s + 0.866 * e]

def test_hebb_unbounded_oja_matches_pca():
    X = _corr_data(); _, norms = LR.hebbian(X); assert norms[-1] > 5 * norms[0]
    w = LR.oja(X); _, _, V = T.jacobi_svd(X)
    assert abs(np.linalg.norm(w) - 1) < 0.05 and abs(w @ V[:, 0]) > 0.98

def test_competitive_and_delta_and_rbm():
    C = np.r_[rng.normal(-2, .4, (100, 2)), rng.normal(2, .4, (100, 2))]
    P = LR.competitive(C, 2); assert sorted(np.round(P[:, 0]))[0] == -2 and sorted(np.round(P[:, 0]))[1] == 2
    X = XOR_X; assert LR.delta_rule(X, [0, 1, 1, 1])[2] is not None and LR.delta_rule(X, [0, 1, 1, 0])[2] is None
    V = np.array([[1, 1, 1, 0, 0, 0], [0, 0, 0, 1, 1, 1]] * 10, float); r = LR.RBM(6, 4)
    e0 = r.train_epoch(V, lr=0.0); e0m = r.mean_energy(V)
    for _ in range(200): e1 = r.train_epoch(V)
    assert e1 < 0.5 * e0 and r.mean_energy(V) < e0m
