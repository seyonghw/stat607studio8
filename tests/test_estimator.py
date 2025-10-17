import numpy as np
import pandas as pd
from numpy.testing import assert_allclose

from estimator import fit_regression
from dgp import generate_data


def _make_df(n=80, gamma=0.5, r2=9.0, sigma2=0.0, seed=11):
    data, beta_true = generate_data(n=n, gamma=gamma, r2=r2, sigma2=sigma2, seed=seed)
    return data, beta_true


def test_ols_matches_true_beta_when_sigma_zero_no_intercept():
    data, beta_true = _make_df(n=100, gamma=0.6, r2=4.0, sigma2=0.0, seed=7)
    est = fit_regression(data, method="ols", fit_intercept=False)
    # est: [intercept, b1..bp]
    assert est.shape[0] == beta_true.shape[0] + 1
    assert est[0] == 0.0
    assert_allclose(est[1:], beta_true, rtol=1e-10, atol=1e-10)


def test_huber_closest_to_ols_under_gaussian_noise():
    data, beta_true = _make_df(n=120, gamma=0.8, r2=5.0, sigma2=1.0, seed=3)
    est_ols = fit_regression(data, method="ols", fit_intercept=False)
    est_hub = fit_regression(data, method="huber", fit_intercept=False)
    # shapes
    assert est_hub.shape == est_ols.shape == (beta_true.shape[0] + 1,)
    # huber should be reasonably close to ols under N noise
    diff = np.linalg.norm(est_hub[1:] - est_ols[1:]) / np.sqrt(beta_true.size)
    assert diff < 0.2  # loose but informative bound


def test_lad_runs_and_returns_correct_length():
    data, beta_true = _make_df(n=90, gamma=1.2, r2=5.0, sigma2=1.0, seed=5)
    est_lad = fit_regression(data, method="lad", fit_intercept=False)
    assert est_lad.shape[0] == beta_true.shape[0] + 1
    assert np.isfinite(est_lad).all()


def test_collinearity_smoke_test_no_crash():
    # Perfectly collinear X columns to exercise retry/conditioning
    n = 60
    x = np.linspace(-1, 1, n)
    X = np.c_[x, 2*x, 3*x]
    y = x + 0.1*np.random.default_rng(0).standard_normal(n)
    df = pd.DataFrame(np.c_[y, X], columns=["y", "X1", "X2", "X3"])
    est = fit_regression(df, method="lad", fit_intercept=True, lad_solver="highs")
    assert est.shape == (4,)  # [intercept, 3 coefs]
    assert np.isfinite(est).all()