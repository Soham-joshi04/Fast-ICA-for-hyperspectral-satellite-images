import numpy as np
from numpy.linalg import svd, pinv

def fast_ica(
        X,                       # shape (n_signals, n_samples)
        n_components=None,       # ≤ n_signals
        fun="tanh",              # 'tanh' (log-cosh), 'cube', 'exp'
        alpha=1.0,               # scaling for tanh
        max_iter=1000,
        tol=1e-8,
        random_state=None,
        whiten=True,
):
    """
    Lightweight FastICA (deflation) inspired by your snippet.

    Returns
    -------
    S : (n_components, n_samples) – estimated independent sources
    W : (n_components, n_components) – un-mixing matrix
    A : (n_components, n_components) – mixing matrix  (≈ pinv(W))
    """
    rng = np.random.default_rng(random_state)
    X = np.asarray(X, dtype=np.float64)
    m, n = X.shape                      # m = observed mixtures

    if n_components is None:
        n_components = m
    if n_components > m:
        raise ValueError("n_components cannot exceed number of signals (rows).")

    # ------------------------------------------------------------------
    # 1.  center
    X -= X.mean(axis=1, keepdims=True)

    # ------------------------------------------------------------------
    # 2.  whiten  (optional)
    if whiten:
        U, s, _ = svd(X, full_matrices=False)
        K = (U[:, :n_components] / s[:n_components]).T      # whitening mat
        Xw = K @ X                                          # shape (k, n)
    else:
        K = np.eye(n_components, m)
        Xw = X.copy()

    # precompute for efficiency
    n_inv = 1.0 / n

    # ------------------------------------------------------------------
    # 3.  choose non-linearity
    if fun == "tanh":
        def g(u):  return np.tanh(alpha * u), alpha * (1.0 - np.tanh(alpha * u)**2)
    elif fun == "cube":
        def g(u):  return u**3, 3 * u**2
    elif fun == "exp":
        def g(u):  exp = np.exp(-(u**2) / 2);  return u * exp, (1 - u**2) * exp
    else:
        raise ValueError("fun must be 'tanh', 'cube' or 'exp'")

    # ------------------------------------------------------------------
    # 4.  deflationary FastICA
    W = np.zeros((n_components, n_components))
    for p in range(n_components):
        w = rng.standard_normal(n_components)
        w /= np.linalg.norm(w)

        for _ in range(max_iter):
            wx = w @ Xw                        # shape (n,)
            gwx, g_wx = g(wx)

            # update rule
            w_new = (gwx @ Xw.T) * n_inv - g_wx.mean() * w

            # Gram–Schmidt against previous components
            if p:
                w_new -= (W[:p] @ w_new) @ W[:p]

            w_new /= np.linalg.norm(w_new)

            # check convergence (|cos(angle)| → 1)
            if abs(np.dot(w_new, w)) > 1 - tol:
                break
            w = w_new

        W[p] = w_new

    # ------------------------------------------------------------------
    # 5.  sources and mixing
    S = W @ Xw                            # (k, n)

    return S
