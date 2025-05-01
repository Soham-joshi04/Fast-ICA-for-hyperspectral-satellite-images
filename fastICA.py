import numpy as np

def fast_ica(X, num_components, max_iter=200, tol=1e-5):
    """
    FastICA algorithm from scratch
    X: B x N numpy array (bands x pixels)
    num_components: how many ICs to extract
    """
    B, N = X.shape

    # 1. Centering
    X = X - np.mean(X, axis=1, keepdims=True)

    # 2. Whitening
    cov = np.cov(X)
    eig_vals, eig_vecs = np.linalg.eigh(cov)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(eig_vals))
    whitening_matrix = D_inv_sqrt @ eig_vecs.T
    X_white = whitening_matrix @ X

    # 3. FastICA
    def g(u):
        return np.tanh(u)

    def g_prime(u):
        return 1 - np.tanh(u) ** 2

    W = np.zeros((num_components, B))

    for i in range(num_components):
        w = np.random.randn(B)
        w = w / np.linalg.norm(w)

        for _ in range(max_iter):
            wx = w @ X_white
            w_new = np.mean(X_white * g(wx), axis=1) - np.mean(g_prime(wx)) * w

            # Orthogonalize against previous components
            if i > 0:
                w_new -= W[:i] @ (W[:i] @ w_new)

            w_new = w_new / np.linalg.norm(w_new)

            if np.abs(np.abs(np.dot(w, w_new)) - 1.0) < tol:
                break

            w = w_new

        W[i, :] = w

    S = W @ X_white  # Independent components
    return S
