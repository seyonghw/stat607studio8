import numpy as np
from typing import Tuple

def evaluate(beta_true: np.ndarray, beta_hat: np.ndarray) -> Tuple[float, float]:
    """
    Compute Monte Carlo mean squared error (MSE)
    and its Monte Carlo standard error (MCSE)
    between true beta and repeated estimated betas.

    Accepts beta_hat with shape (R, p) OR (R, p+1) if an intercept is present
    in the first column. In the latter case, the intercept column is ignored.

    Parameters
    ----------
    beta_true : np.ndarray, shape (p, )
        True coefficient vector (fixed across repetitions) ((no intercept))
    beta_hat : np.ndarray, shape (R, p) or (R, p+1) (may include an intercept
        in column 0 (will be ignored))
        Estimated coefficients across Monte Carlo repetitions.

    Returns
    -------
    tuple of floats
        (mse, mcse)
        mse  : mean MSE over all repetitions
        mcse : Monte Carlo standard error of the MSE estimate
    """

    # diff = beta_hat[:,1:] - beta_true  # shape (R, p)
    # n_sim = np.shape(beta_hat)[0]
    # if n_sim == 1:
    #     mse = diff
    #     mcse = 0
    # else:
    #     mse_per_rep = np.mean(diff ** 2, axis=1)  # MSE per repetition, shape (R,)
    #     mse = np.mean(mse_per_rep)                # overall mean MSE
    #     mcse = np.std(mse_per_rep, ddof=1) / np.sqrt(len(mse_per_rep))  # Monte Carlo SE
    # return mse, mcse
    beta_true = np.asarray(beta_true, dtype=float).reshape(-1)
    beta_hat = np.asarray(beta_hat, dtype=float)

    if beta_hat.ndim != 2:
        raise ValueError("beta_hat must be 2D: (R, p) or (R, p+1).")

    R, q = beta_hat.shape
    p = beta_true.shape[0]

    # Allow an intercept in the first column
    if q == p + 1:
        bh = beta_hat[:, 1:]
    elif q == p:
        bh = beta_hat
    else:
        raise ValueError(f"beta_hat has {q} columns but beta_true has length {p} "
                         "(expected p or p+1 columns).")

    if not np.isfinite(beta_true).all() or not np.isfinite(bh).all():
        raise ValueError("beta_true and beta_hat must contain only finite values.")

    # Per-repetition MSE over coefficients
    diff = bh - beta_true  # (R, p)
    sq_err_per_rep = np.sum(diff**2, axis=1)

    if R == 1:
        return float(sq_err_per_rep.item()), 0.0
    mse  = float(np.mean(sq_err_per_rep))
    mcse = float(np.std(sq_err_per_rep, ddof=1) / np.sqrt(R))

    return mse, mcse


def ground_truth_mse(aspect_ratio: float, variance: float, beta_norm: float):
    """
    Compute the theoretical MSE of the OLS estimator
    based on the aspect ratio γ = p/n.

    Parameters
    ----------
    aspect_ratio : float
        Ratio p/n of covariates to observations (γ).
    variance : float
        Variance σ² of the noise.
    beta_norm : float
        Norm r = ||β|| of the true parameter.

    Returns
    -------
    float
        Theoretical expected MSE E[||β̂ - β||²].
        Returns np.nan if aspect_ratio == 1.
    """
    gamma = float(aspect_ratio)
    sigma2 = float(variance)
    r = float(beta_norm)

    if sigma2 < 0:
        raise ValueError("variance (sigma²) must be nonnegative.")
    if r < 0:
        raise ValueError("beta_norm (||β||) must be nonnegative.")

    if np.isclose(gamma, 1.0, atol=1e-12):
        return np.nan
    if gamma < 1.0:
        return sigma2 * gamma / (1.0 - gamma)
    else:
        return r ** 2 * (1.0 - 1.0 / gamma) + sigma2 * (1.0 / (gamma - 1.0))

