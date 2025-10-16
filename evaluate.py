import numpy as np

def evaluate(beta_true: np.ndarray, beta_hat: np.ndarray) -> tuple[float, float]:
    """
    Compute Monte Carlo mean squared error (MSE)
    and its Monte Carlo standard error (MCSE)
    between true beta and repeated estimated betas.

    Parameters
    ----------
    beta_true : np.ndarray, shape (p,)
        True coefficient vector (fixed across repetitions).
    beta_hat : np.ndarray, shape (R, p)
        Estimated coefficients across Monte Carlo repetitions.

    Returns
    -------
    tuple of floats
        (mse, mcse)
        mse  : mean MSE over all repetitions
        mcse : Monte Carlo standard error of the MSE estimate
    """
    diff = beta_hat[:,1:] - beta_true  # shape (R, p)
    n_sim = np.shape(beta_hat)[0]
    if n_sim == 1:
        mse = diff
        mcse = 0
    else: 
        mse_per_rep = np.mean(diff ** 2, axis=1)  # MSE per repetition, shape (R,)
        mse = np.mean(mse_per_rep)                # overall mean MSE
        mcse = np.std(mse_per_rep, ddof=1) / np.sqrt(len(mse_per_rep))  # Monte Carlo SE
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
    if aspect_ratio < 1:
        return variance * aspect_ratio / (1 - aspect_ratio)
    elif aspect_ratio > 1:
        return beta_norm**2 * (1 - 1/aspect_ratio) + variance * (1 / (aspect_ratio - 1))
    else:
        return np.nan