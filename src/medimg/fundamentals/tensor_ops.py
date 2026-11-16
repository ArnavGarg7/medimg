"""Step 3: tensor sandbox. Explicit (loop-based) broadcasting, inner product, reshape, convolution, Jacobi SVD,
and numerical gradient checking. Each function is verified against NumPy in tests/."""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

# ---------- broadcasting ----------
def broadcast_shape(a, b):
    """NumPy broadcasting rule: align shapes on the right; dims must match or be 1."""
    out = []
    for i in range(1, max(len(a), len(b)) + 1):
        da = a[-i] if i <= len(a) else 1
        db = b[-i] if i <= len(b) else 1
        if da != db and da != 1 and db != 1:
            raise ValueError(f"cannot broadcast {tuple(a)} with {tuple(b)}")
        out.append(max(da, db))
    return tuple(out[::-1])

def broadcast_apply(a, b, fn):
    """Element-wise fn(a, b) with explicit index mapping (stride 0 on stretched dims)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    shape = broadcast_shape(a.shape, b.shape)
    out = np.empty(shape)
    for idx in np.ndindex(*shape):
        ia = tuple(0 if a.shape[-k] == 1 else idx[-k] for k in range(len(a.shape), 0, -1))
        ib = tuple(0 if b.shape[-k] == 1 else idx[-k] for k in range(len(b.shape), 0, -1))
        out[idx] = fn(a[ia], b[ib])
    return out

# ---------- products / reshape ----------
def inner(u, v):
    u, v = np.asarray(u, float), np.asarray(v, float)
    if u.shape != v.shape or u.ndim != 1: raise ValueError("inner: need equal-length 1D vectors")
    return float(sum(x * y for x, y in zip(u, v)))

def matmul_loops(A, B):
    A, B = np.asarray(A, float), np.asarray(B, float)
    if A.shape[1] != B.shape[0]: raise ValueError(f"matmul: {A.shape} x {B.shape}")
    C = np.zeros((A.shape[0], B.shape[1]))
    for i in range(A.shape[0]):
        for k in range(A.shape[1]):
            C[i] += A[i, k] * B[k]
    return C

def reshape_view_demo(x, shape):
    """Reshape returns a view when memory layout allows; returns (array, shares_memory)."""
    y = np.reshape(x, shape)
    return y, bool(np.shares_memory(x, y))

# ---------- convolution (cross-correlation, as in CNN libraries) ----------
def conv2d_valid(img, ker):
    win = sliding_window_view(img, ker.shape)          # (H-kh+1, W-kw+1, kh, kw)
    return np.einsum("ijuv,uv->ij", win, ker)

def conv3d_valid(vol, ker):
    win = sliding_window_view(vol, ker.shape)          # (D',H',W',kd,kh,kw)
    return np.einsum("ijkuvw,uvw->ijk", win, ker)

# ---------- SVD (one-sided Jacobi / Hestenes) ----------
def jacobi_svd(A, max_sweeps=60, eps=1e-12):
    """Returns U (m,r), S (r,), V (n,r) with A = U diag(S) V^T, S descending."""
    A = np.array(A, float)
    m, n = A.shape
    if m < n:
        U, S, V = jacobi_svd(A.T, max_sweeps, eps)
        return V, S, U
    U, V = A.copy(), np.eye(n)
    for _ in range(max_sweeps):
        rotated = False
        for p in range(n - 1):
            for q in range(p + 1, n):
                alpha, beta, gamma = U[:, p] @ U[:, p], U[:, q] @ U[:, q], U[:, p] @ U[:, q]
                if gamma == 0 or abs(gamma) <= eps * np.sqrt(alpha * beta): continue
                rotated = True
                zeta = (beta - alpha) / (2 * gamma)
                t = (1.0 if zeta >= 0 else -1.0) / (abs(zeta) + np.sqrt(1 + zeta ** 2))
                c = 1 / np.sqrt(1 + t * t); s = c * t
                for M in (U, V):
                    mp, mq = M[:, p].copy(), M[:, q].copy()
                    M[:, p], M[:, q] = c * mp - s * mq, s * mp + c * mq
        if not rotated: break
    S = np.linalg.norm(U, axis=0)
    order = np.argsort(-S)
    S, U, V = S[order], U[:, order], V[:, order]
    U = np.where(S > 1e-14, U / np.where(S > 1e-14, S, 1), 0.0)
    return U, S, V

def svd_reconstruct(U, S, V, k):
    return (U[:, :k] * S[:k]) @ V[:, :k].T

# ---------- numerical gradient checking ----------
def numeric_grad(f, x, h=1e-6):
    """Central differences of scalar f at array x."""
    x = np.array(x, float); g = np.zeros_like(x)
    for idx in np.ndindex(*x.shape):
        old = x[idx]
        x[idx] = old + h; fp = f(x)
        x[idx] = old - h; fm = f(x)
        x[idx] = old; g[idx] = (fp - fm) / (2 * h)
    return g

def rel_error(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.max(np.abs(a - b) / np.maximum(1e-8, np.abs(a) + np.abs(b))))
