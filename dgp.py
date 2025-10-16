import numpy as np
import pandas as pd
import scipy

def generate_design_matrix(n, p, rng=None):
    """
    X in R^{n x p} with iid N(0,1) entries.
    """
    rng = np.random.default_rng(rng)
    X = rng.normal(loc=0.0, scale=1.0, size=(n, p))
    return pd.DataFrame(X, columns=[f"X{j+1}" for j in range(p)])

def generate_beta(p, r2=5):
    """
    Beta = sqrt(r^2 / p) * 1_p (no intercept).
    """
    if p <= 0:
        raise ValueError("p must be positive.")
    if r2 < 0:
        raise ValueError("r2 must be nonnegative.")
    return np.full(p, np.sqrt(r2 / p), dtype=float)

def generate_response(X, beta, sigma2=1, rng=None) -> pd.Series:
    """
    y = X beta + sigma * eps,  eps ~ N(0, I_n); no intercept term.
    """
    rng = np.random.default_rng(rng)
    n = X.shape[0]
    if sigma2 < 0:
        raise ValueError("sigma2 must be nonnegative.")
    sigma = float(np.sqrt(sigma2))
    eps = rng.normal(loc=0.0, scale=1.0, size=n)
    y = X.to_numpy() @ beta + sigma * eps
    return pd.Series(y, name="y")

def generate_data(n=200, gamma=1, r2=5, sigma2=1.0, seed= 1):
    """
    Experimental setup (no intercept) for benign overfitting:
      - p = floor(gamma * n)
      - X_ij ~ N(0,1), eps_i ~ N(0,1)
      - beta = sqrt(r^2 / p) * 1_p
      - y = X beta + sigma * eps, with sigma^2 = sigma2

    Returns
    -------
    data : DataFrame with columns ['y', 'X1', ..., 'Xp']
    beta : np.ndarray, shape (p,)
    """
    if n <= 0:
        raise ValueError("n must be a positive integer.")
    if gamma <= 0:
        raise ValueError("gamma must be positive.")
    p = int(np.floor(gamma * n))
    if p < 1:
        p = 1  

    rng = np.random.default_rng(seed)
    X = generate_design_matrix(n=n, p=p, rng=rng)
    beta = generate_beta(p=p, r2=r2)
    y = generate_response(X=X, beta=beta, sigma2=sigma2, rng=rng)

    data = pd.concat([y, X], axis=1)
    return data, beta
