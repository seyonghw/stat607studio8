import numpy as np

def evaluate(beta_true: np.ndarray, beta_hat: np.ndarray) -> float:
    """
    Compute Monte Carlo mean squared error (MSE)
    between true beta and repeated estimated betas.

    Parameters
    ----------
    beta_true : np.ndarray, shape (p,)
        True coefficient vector (fixed across repetitions).
    beta_hat : np.ndarray, shape (R, p)
        Estimated coefficients across Monte Carlo repetitions.

    Returns
    -------
    float
        Mean MSE over all repetitions: average(||beta_hat_r - beta_true||^2)
    """
    diff = beta_hat - beta_true  # shape (R, p)
    mse_per_rep = np.mean(diff ** 2, axis=1)  # shape (R,)
    mse = np.mean(mse_per_rep)                # scalar
    return mse