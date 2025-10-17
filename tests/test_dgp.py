import numpy as np
import pandas as pd
import pytest

from dgp import generate_design_matrix, generate_beta, generate_response, generate_data


def test_generate_design_matrix_shape_and_basic_stats():
    n, p, seed = 200, 37, 1610
    X = generate_design_matrix(n, p, rng=seed)
    assert isinstance(X, pd.DataFrame)
    assert X.shape == (n, p)
    assert list(X.columns[:3]) == ["X1", "X2", "X3"]

    # sanity: roughly zero mean / unit variance (LLN)
    col_means = X.mean().to_numpy()
    col_vars = X.var(ddof=0).to_numpy()

    thr = 2.0 * np.sqrt(np.log(p) / n)  # ~0.27 here
    assert np.max(np.abs(col_means)) < thr

    assert np.all(np.abs(col_vars - 1.0) < thr)


def test_generate_beta_shape_and_basic_stats():
    p, r2 = 50, 5.0
    beta = generate_beta(p, r2=r2)
    assert beta.shape == (p,)
    # all entries equal and positive
    assert np.allclose(beta, beta[0])
    # L2-norm should be sqrt(r2)
    assert np.isclose(np.linalg.norm(beta), np.sqrt(r2), rtol=1e-12, atol=1e-12)
    # per-entry value
    assert np.isclose(beta[0], np.sqrt(r2 / p), rtol=1e-12, atol=1e-12)


def test_generate_beta_invalid():
    with pytest.raises(ValueError):
        generate_beta(0, r2=1.0)
    with pytest.raises(ValueError):
        generate_beta(5, r2=-1.0)


def test_generate_response_dimensions_and_reproducibility():
    n, p, seed = 120, 10, 999
    X = generate_design_matrix(n, p, rng=seed)
    beta = generate_beta(p, r2=4.0)  # ||beta||^2 = 4
    y1 = generate_response(X, beta, sigma2=1.0, rng=seed)
    y2 = generate_response(X, beta, sigma2=1.0, rng=seed)
    assert isinstance(y1, pd.Series)
    assert y1.shape == (n,)
    assert y1.name == "y"
    # reproducible for the same rng/seed
    assert np.allclose(y1.to_numpy(), y2.to_numpy(), rtol=0, atol=0)


def test_generate_response_invalid_sigma2():
    n, p = 30, 5
    X = generate_design_matrix(n, p, rng=0)
    beta = generate_beta(p, r2=1.0)
    with pytest.raises(ValueError):
        generate_response(X, beta, sigma2=-0.1, rng=0)


def test_generate_data_shapes_gamma_and_reproducibility():
    n, gamma, r2, sigma2, seed = 200, 1.3, 5.0, 1.0, 42
    # p = floor(gamma * n)
    p_expected = int(np.floor(gamma * n))
    data1, beta1 = generate_data(n=n, gamma=gamma, r2=r2, sigma2=sigma2, seed=seed)
    data2, beta2 = generate_data(n=n, gamma=gamma, r2=r2, sigma2=sigma2, seed=seed)
    assert isinstance(data1, pd.DataFrame)
    assert data1.shape == (n, p_expected + 1)  # y + p features
    assert data1.columns[0] == "y"
    assert beta1.shape == (p_expected,)
    # reproducible
    pd.testing.assert_frame_equal(data1, data2, check_exact=True)
    assert np.allclose(beta1, beta2, rtol=0, atol=0)


def test_generate_data_gamma_small_caps_at_one_feature():
    n, gamma = 50, 1e-6
    data, beta = generate_data(n=n, gamma=gamma, r2=2.0, sigma2=1.0, seed=7)
    # floor(gamma*n) == 0 -> coerced to 1
    assert data.shape[1] == 2   # y + X1
    assert beta.shape == (1,)


def test_generate_data_invalid_inputs():
    with pytest.raises(ValueError):
        generate_data(n=0, gamma=1.0, r2=1.0, sigma2=1.0, seed=1)
    with pytest.raises(ValueError):
        generate_data(n=10, gamma=0.0, r2=1.0, sigma2=1.0, seed=1)