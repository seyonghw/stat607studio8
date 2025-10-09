import pandas as pd
import numpy as np
import scipy


def generate_design_matrix(n, aspect_ratio, correlation_structure="identity", rho=0.5, seed=1):
    """
    Generate a sample design matrix for demonstration purposes.

    Parameters
    ----------
    n : int
        Number of samples.
    aspect_ratio : int
        Ratio of features to samples.
    correlation_structure : str, optional
        Type of correlation structure. Default is "identity".
    rho : float, optional
        Correlation coefficient for autoregressive structure. Default is 0.5.
    seed : int, optional
        Random seed for reproducibility. Default is 1.

    Returns
    -------
    pd.DataFrame
        A DataFrame representing the design matrix with n samples and p features.
    """
    np.random.seed(seed)

    p = int(aspect_ratio * n)
    if(correlation_structure == "identity"):
        X = np.random.normal(0, 1, (n, p))
    elif(correlation_structure == "autoregressive"):
        cov = rho ** np.abs(np.subtract.outer(np.arange(p), np.arange(p)))
        X = np.random.multivariate_normal(np.zeros(p), cov, size=n)
    else:
        raise ValueError("Unsupported correlation structure")

    return pd.DataFrame(X, columns=[f'X{i+1}' for i in range(p)])

def generate_coefficients(p, seed=1):
    """
    Generate a coefficient vector for the linear model.

    Parameters
    ----------
    p : int
        Number of features.
    """
    np.random.seed(seed)
    ints = range(-3, 3)
    beta0 = np.random.choice(ints, 1)
    beta = np.random.choice(ints, p, replace=True)
    return beta0, beta

def generate_response(X, beta0, beta, seed=1, snr=1, distribution="normal", df=1):
    """
    Generate a response variable based on the design matrix and coefficients.

    Parameters
    ----------
    X : pd.DataFrame
        Design matrix.
    beta0 : float
        Intercept term.
    beta : np.ndarray
        Coefficient vector of length p.
    seed : int, optional
        Random seed for reproducibility. Default is 1.
    snr : float, optional
        Signal-to-noise ratio. Default is 1.
    distribution : str, optional
        Type of error distribution ("normal" or "t"). Default is "normal".
    """
    np.random.seed(seed)
    n = X.shape[0]
    p = X.shape[1]
    sigma = np.sqrt(np.transpose(beta)@X.T@X@beta / (snr))
    if distribution == "t":
        errors = scipy.stats.t.rvs(df=df, size=n) * sigma
    elif distribution == "normal":
        errors = np.random.normal(0, sigma, n)
    else:
        raise ValueError("Unsupported distribution type")
    
    y = beta0 + X.values @ beta + errors
    return pd.Series(y, name='y')

def generate_data(n, aspect_ratio, correlation_structure="identity", rho=0.5, seed=1, snr=1, distribution="normal", df_t=1):
    """
    Generate a complete dataset including design matrix and response variable.

    Parameters
    ----------
    n : int
        Number of samples.
    aspect_ratio : int
        Ratio of features to samples.
    correlation_structure : str, optional
        Type of correlation structure. Default is "identity".
    rho : float, optional
        Correlation coefficient for autoregressive structure. Default is 0.5.
    seed : int, optional
        Random seed for reproducibility. Default is 1.
    snr : float, optional
        Signal-to-noise ratio. Default is 1.
    distribution : str, optional
        Type of error distribution ("normal" or "t"). Default is "normal".
    df_t : int, optional
        Degrees of freedom for t-distribution if used. Default is 1.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the response variable and design matrix.
    """
    X = generate_design_matrix(n, aspect_ratio, correlation_structure, rho, seed)
    p = X.shape[1]
    beta0, beta = generate_coefficients(p, seed)
    y = generate_response(X, beta0, beta, seed, snr, distribution, df_t)
    data = pd.concat([y, X], axis=1)
    return data, beta0, beta