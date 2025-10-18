import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, HuberRegressor, QuantileRegressor

def fit_regression(
    df: pd.DataFrame,
    method: str = "ols",
    *,
    # shared
    fit_intercept: bool = True,
    # LAD (QuantileRegressor) params
    lad_quantile: float = 0.5,
    lad_alpha: float = 1e-8,
    lad_solver: str = "highs",
    # Huber params
    huber_epsilon: float = 1.35,
    huber_alpha: float = 0.0,
    huber_max_iter: int = 5_000,
):
    """
    Fit OLS, LAD (median/quantile regression), or Huber on a DataFrame.
    - df: first column = Y, remaining columns = X1..Xp
    - method: 'ols' | 'lad' | 'huber'

    Tunables:
      fit_intercept (all)
      lad_quantile, lad_alpha, lad_solver (LAD)
      huber_epsilon, huber_alpha, huber_max_iter (Huber)
    """
    y = df.iloc[:, 0].to_numpy()
    X = df.iloc[:, 1:].to_numpy()
    feature_names = df.columns[1:]
    m = method.lower()

    if m == "ols":
        model = LinearRegression(fit_intercept=fit_intercept)
    elif m == "lad":
        model = QuantileRegressor(
            quantile=lad_quantile,
            alpha=lad_alpha,
            solver=lad_solver,
            fit_intercept=fit_intercept,
        )
    elif m == "huber":
        model = HuberRegressor(
            epsilon=huber_epsilon,
            alpha=huber_alpha,
            max_iter=huber_max_iter,
            fit_intercept=fit_intercept,
            warm_start=False,
        )
    else:
        raise ValueError("method must be one of: 'ols', 'lad', 'huber'")

    model.fit(X, y)

    intercept = float(getattr(model, "intercept_", 0.0))
    coef = np.asarray(model.coef_, dtype=float).ravel()

    # concatenate to one (p+1)-dimensional vector
    beta = np.concatenate(([intercept], coef))

    return beta