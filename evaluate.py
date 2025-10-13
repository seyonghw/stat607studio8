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
    diff = beta_hat - beta_true  # shape (R, p)
    mse_per_rep = np.mean(diff ** 2, axis=1)  # MSE per repetition, shape (R,)
    mse = np.mean(mse_per_rep)                # overall mean MSE
    mcse = np.std(mse_per_rep, ddof=1) / np.sqrt(len(mse_per_rep))  # Monte Carlo SE
    return mse, mcse