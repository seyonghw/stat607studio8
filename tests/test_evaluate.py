import numpy as np
import pytest
from evaluate import evaluate, ground_truth_mse


def test_evaluate_shapes_without_intercept_R_gt_1():
    p, R = 5, 10
    beta_true = np.arange(1, p+1, dtype=float) / 10.0
    rng = np.random.default_rng(0)
    # unbiased noisy estimates
    bh = beta_true + 0.1 * rng.standard_normal((R, p))
    mse, mcse = evaluate(beta_true, bh)
    assert isinstance(mse, float) and isinstance(mcse, float)
    assert mse > 0 and mcse > 0


def test_evaluate_with_intercept_column_dropped():
    p, R = 4, 7
    beta_true = np.ones(p)
    rng = np.random.default_rng(1)
    intercept = rng.normal(size=(R, 1))
    slopes = beta_true + 0.01 * rng.standard_normal((R, p))
    bh = np.concatenate([intercept, slopes], axis=1)  # (R, p+1)
    mse, mcse = evaluate(beta_true, bh)
    # with tiny noise, mse should be very small
    assert mse < 1e-3


def test_evaluate_R_eq_1_scalar_outputs():
    p, R = 3, 1
    beta_true = np.zeros(p)
    bh = np.array([[0.0, 0.0, 1.0]])  # shape (1,3)
    mse, mcse = evaluate(beta_true, bh)
    assert np.isclose(mse, 1.0 / 3.0)
    assert mcse == 0.0


def test_evaluate_bad_shapes():
    beta_true = np.zeros(3)
    with pytest.raises(ValueError):
        evaluate(beta_true, np.zeros((5, 5)))  # 5 != 3 nor 4


def test_ground_truth_mse_pieces_and_nan():
    sigma2 = 1.0
    r = 2.0
    # gamma < 1
    g = 0.5
    assert np.isclose(ground_truth_mse(g, sigma2, r), sigma2 * g / (1-g))
    # gamma > 1
    g = 2.0
    expect = r**2 * (1 - 1/g) + sigma2 * (1/(g - 1))
    assert np.isclose(ground_truth_mse(g, sigma2, r), expect)
    # gamma ~ 1 -> nan
    assert np.isnan(ground_truth_mse(1.0, sigma2, r))