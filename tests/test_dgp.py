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