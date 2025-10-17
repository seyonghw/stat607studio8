import numpy as np
import pandas as pd
from typing import Tuple


def _as_generator(rng):
    if rng is None:
        return np.random.default_rng()
    if isinstance(rng, np.random.Generator):
        return rng
    if isinstance(rng, np.random.BitGenerator):
        return np.random.Generator(rng)
    return np.random.default_rng(rng)


def generate_design_matrix(n: int, p: int, rng=None) -> pd.DataFrame:
    """
    X in R^{n x p} with iid N(0,1) entries.
    """
    if n <= 0 or p <= 0:
        raise ValueError("n and p must be positive.")
    rng = _as_generator(rng)
    X = rng.normal(loc=0.0, scale=1.0, size=(n, p))
    return pd.DataFrame(X, columns=[f"X{j+1}" for j in range(p)])


def generate_beta(p: int, r2: float = 5.0) -> np.ndarray:
    """
    Beta = sqrt(r^2 / p) * 1_p (no intercept).
    """
    if p <= 0:
        raise ValueError("p must be positive.")
    if r2 < 0:
        raise ValueError("r2 must be non-negative.")
    return np.full(p, np.sqrt(r2 / p), dtype=float)


def generate_response(X: pd.DataFrame, beta: np.ndarray, sigma2: float = 1.0, rng=None) -> pd.Series:
    """
    y = X beta + sigma * eps,  eps ~ N(0, I_n); no intercept term.
    """
    # rng = np.random.default_rng(rng)

    if sigma2 < 0:
        raise ValueError("sigma2 must be non-negative.")
    if X.shape[1] != beta.shape[0]:
        raise ValueError(f"X has {X.shape[1]} columns but beta has length {beta.shape[0]}.")

    rng = _as_generator(rng)
    sigma = float(np.sqrt(sigma2))
    n = X.shape[0]
    eps = rng.normal(loc=0.0, scale=1.0, size=n)
    y = X.to_numpy() @ beta + sigma * eps
    return pd.Series(y, name="y")


def generate_data(n: int = 200, gamma: float = 1.0, r2: float = 5.0, sigma2: float = 1.0, seed: int | None = 1) -> Tuple[pd.DataFrame, np.ndarray]:
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

    p = max(int(np.floor(gamma * n)), 1)
    rng = _as_generator(seed)
    X = generate_design_matrix(n=n, p=p, rng=rng)
    beta = generate_beta(p=p, r2=r2)
    y = generate_response(X=X, beta=beta, sigma2=sigma2, rng=rng)

    data = pd.concat([y, X], axis=1)
    return data, beta
