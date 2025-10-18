import numpy as np
import pandas as pd
import pytest

import analyze
from analyze import analyze_data, simulation_scenarios, simulate, run_all


# -------- stubs (fakes) --------

class Calls:
    """Simple collector for inspecting how analyze_data calls its deps."""
    def __init__(self):
        self.generate_data = []

calls = Calls()

def fake_generate_data(*, n, gamma, r2, sigma2, seed):
    """
    Return a tiny synthetic dataset and a known beta, and record args.
    p = max(floor(gamma*n), 1) to mirror DGP.
    """
    calls.generate_data.append(dict(n=n, gamma=gamma, r2=r2, sigma2=sigma2, seed=seed))
    p = max(int(np.floor(gamma * n)), 1)
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    beta = np.full(p, 1.0 / np.sqrt(p))          # ||beta||^2 = 1
    y = X @ beta + rng.normal(size=n)            # not used by our fake estimator
    cols = ["y"] + [f"X{j+1}" for j in range(p)]
    data = pd.DataFrame(np.c_[y, X], columns=cols)
    return data, beta

def fake_fit_regression(data, method="ols", fit_intercept=False, **kwargs):
    """
    Return perfect estimates so true MSE==0.
    If fit_intercept=False, we still return [0, beta] (as analyze_data expects).
    """
    p = data.shape[1] - 1
    beta = np.full(p, 1.0 / np.sqrt(p))
    return np.concatenate(([0.0], beta))  # [intercept, coef...]

def fake_evaluate(beta_true, beta_hat):
    """
    Intercept-agnostic evaluate: drop col0 if present, compute exact MSE/MCSE.
    """
    bt = np.asarray(beta_true, float)
    bh = np.asarray(beta_hat, float)
    if bh.shape[1] == bt.shape[0] + 1:
        bh = bh[:, 1:]
    diff = bh - bt
    mse_per_rep = np.mean(diff**2, axis=1)
    R = mse_per_rep.shape[0]
    if R == 1:
        return float(mse_per_rep.item()), 0.0
    return float(mse_per_rep.mean()), float(mse_per_rep.std(ddof=1) / np.sqrt(R))


# -------- tests for analyze_data ONLY --------

def test_analyze_data_returns_scalar_mse_mcse_and_zero_with_perfect_fit(monkeypatch):
    """
    With perfect estimator/evaluate stubs, (mse, mcse) should be (0, 0).
    Also verify shapes handed to evaluate: (R, p) vs (R, p+1) via our stubs.
    """
    calls.generate_data.clear()
    monkeypatch.setattr(analyze, "generate_data", fake_generate_data)
    monkeypatch.setattr(analyze, "fit_regression", fake_fit_regression)
    monkeypatch.setattr(analyze, "evaluate", fake_evaluate)

    params = dict(
        n=20,
        aspect_ratio=0.5,
        seed=7,
        snr=5.0,                    # will be mapped to r2
        sigma2=2.0,
        distribution="normal",
        correlation_structure="identity",
        df_t=None,
    )
    mse, mcse = analyze.analyze_data(n_sim=3, parameters=params)
    assert mse == 0.0
    assert mcse == 0.0

def test_analyze_data_maps_snr_to_r2_and_uses_sigma2_and_seeds(monkeypatch):
    """
    Check that analyze_data:
      - maps 'snr' -> r2 when 'r2' not provided,
      - passes sigma2 through,
      - advances seed as seed+sim.
    """
    calls.generate_data.clear()
    monkeypatch.setattr(analyze, "generate_data", fake_generate_data)
    monkeypatch.setattr(analyze, "fit_regression", fake_fit_regression)
    monkeypatch.setattr(analyze, "evaluate", fake_evaluate)

    n_sim = 3
    base_seed = 100
    snr = 9.0
    sigma2 = 1.5
    params = dict(n=10, aspect_ratio=0.7, seed=base_seed, snr=snr, sigma2=sigma2)

    _ = analyze.analyze_data(n_sim=n_sim, parameters=params)

    # We should have exactly n_sim calls
    assert len(calls.generate_data) == n_sim

    # All calls used the expected mapped/forwarded params
    for s, call in enumerate(calls.generate_data):
        assert call["n"] == 10
        assert call["gamma"] == pytest.approx(0.7)
        assert call["r2"] == pytest.approx(snr)       # snr mapped to r2
        assert call["sigma2"] == pytest.approx(sigma2)
        assert call["seed"] == base_seed + s

def test_analyze_data_prefers_explicit_r2_over_snr(monkeypatch):
    """
    If both r2 and snr are present, r2 must win.
    """
    calls.generate_data.clear()
    monkeypatch.setattr(analyze, "generate_data", fake_generate_data)
    monkeypatch.setattr(analyze, "fit_regression", fake_fit_regression)
    monkeypatch.setattr(analyze, "evaluate", fake_evaluate)

    params = dict(n=15, aspect_ratio=1.2, seed=5, snr=4.0, r2=7.0, sigma2=1.0)
    _ = analyze.analyze_data(n_sim=2, parameters=params)

    assert len(calls.generate_data) == 2
    for call in calls.generate_data:
        assert call["r2"] == pytest.approx(7.0)       # explicit r2 overrides snr
        assert call["gamma"] == pytest.approx(1.2)

def test_analyze_data_handles_nsim_1_scalar_outputs(monkeypatch):
    """
    R=1 should still return floats, not arrays.
    """
    calls.generate_data.clear()
    monkeypatch.setattr(analyze, "generate_data", fake_generate_data)
    monkeypatch.setattr(analyze, "fit_regression", fake_fit_regression)
    monkeypatch.setattr(analyze, "evaluate", fake_evaluate)

    params = dict(n=12, aspect_ratio=0.3, seed=0, snr=5.0, sigma2=1.0)
    mse, mcse = analyze.analyze_data(n_sim=1, parameters=params)
    assert isinstance(mse, float) and isinstance(mcse, float)
    assert mse == 0.0 and mcse == 0.0


import numpy as np
import pytest
import analyze  # module that defines simulation_scenarios

def test_counts_and_keys_for_each_bucket():
    dist = "normal"
    corr = "identity"
    snr  = 5

    # n == 1 → 5000 scenarios
    sc1 = analyze.simulation_scenarios([1], dist, corr, snr)
    assert len(sc1) == 5000

    # n == 50 → 100 scenarios
    sc50 = analyze.simulation_scenarios([50], dist, corr, snr)
    assert len(sc50) == 100

    # n == other → 5 scenarios
    sc_other = analyze.simulation_scenarios([999], dist, corr, snr)
    assert len(sc_other) == 5

    # keys present and values wired through
    for s in (sc1[0], sc50[0], sc_other[0]):
        for k in ("number iterations", "distribution", "correlation_structure", "snr", "aspect_ratio"):
            assert k in s
        assert s["distribution"] == dist
        assert s["correlation_structure"] == corr
        assert s["snr"] == snr

def test_logspace_ranges_and_monotonicity():
    dist = "normal"
    corr = "identity"
    snr  = 5

    sc1  = analyze.simulation_scenarios([1], dist, corr, snr)
    sc50 = analyze.simulation_scenarios([50], dist, corr, snr)

    # extract gammas
    g1   = np.array([s["aspect_ratio"] for s in sc1])
    g50  = np.array([s["aspect_ratio"] for s in sc50])

    # endpoints (allow tiny FP wiggle)
    assert g1.min()  == pytest.approx(0.1, abs=1e-12)
    assert g1.max()  == pytest.approx(10.0, abs=1e-12)
    assert g50.min() == pytest.approx(0.1, abs=1e-12)
    assert g50.max() == pytest.approx(10.0, abs=1e-12)

    # strictly increasing
    assert np.all(np.diff(g1)  > 0)
    assert np.all(np.diff(g50) > 0)

def test_else_bucket_exact_values_and_tagged_n():
    dist = "normal"
    corr = "identity"
    snr  = 5

    sc = analyze.simulation_scenarios([1000], dist, corr, snr)
    gammas = [s["aspect_ratio"] for s in sc]
    assert gammas == [0.2, 0.5, 0.8, 2, 5]
    # make sure "number iterations" tags each scenario with the n we passed
    assert all(s["number iterations"] == 1000 for s in sc)

def test_multiple_ns_concatenate_in_order():
    dist = "normal"
    corr = "identity"
    snr  = 5

    sc = analyze.simulation_scenarios([1, 50, 7], dist, corr, snr)
    # beginning should be the 5000 γ for n=1, then 100 for n=50, then 5 for n=7
    assert len(sc) == 5000 + 100 + 5
    assert sc[0]["number iterations"] == 1
    assert sc[4999]["number iterations"] == 1
    assert sc[5000]["number iterations"] == 50
    assert sc[5000 + 99]["number iterations"] == 50
    assert sc[-5]["number iterations"] == 7
    assert sc[-1]["number iterations"] == 7


def test_simulate_aggregates_rows_and_order(monkeypatch):
    """
    simulate() should:
      - iterate scenarios in order,
      - call analyze_data once per scenario,
      - return a DataFrame with the expected columns and values.
    """
    scenarios = [
        {"number iterations": 1, "distribution": "normal", "correlation_structure": "identity", "snr": 5, "aspect_ratio": 0.2},
        {"number iterations": 50, "distribution": "normal", "correlation_structure": "identity", "snr": 5, "aspect_ratio": 2.0},
    ]
    monkeypatch.setattr(analyze, "simulation_scenarios", lambda *args, **kwargs: scenarios)

    def fake_analyze_data(n_sim, params):
        # Return (MSE, MCSE) deterministically from gamma to simplify assertions.
        g = float(params["aspect_ratio"])
        return g, g / 10.0

    monkeypatch.setattr(analyze, "analyze_data", fake_analyze_data)

    out = analyze.simulate([1, 50], "normal", "identity", 5, n=200, seed=1)
    assert list(out.columns) == ["n_sim", "aspect_ratio", "MSE", "MCSE"]
    assert out.shape[0] == 2

    # Row 0: n_sim=1, gamma=0.2
    r0 = out.iloc[0]
    assert r0["n_sim"] == 1
    assert r0["aspect_ratio"] == pytest.approx(0.2)
    assert r0["MSE"] == pytest.approx(0.2)
    assert r0["MCSE"] == pytest.approx(0.02)

    # Row 1: n_sim=50, gamma=2.0
    r1 = out.iloc[1]
    assert r1["n_sim"] == 50
    assert r1["aspect_ratio"] == pytest.approx(2.0)
    assert r1["MSE"] == pytest.approx(2.0)
    assert r1["MCSE"] == pytest.approx(0.2)


def test_simulate_seeding_progression_and_forwarded_params(monkeypatch):
    """
    Check that simulate() forwards the correct parameters to analyze_data
    and uses seed + i per scenario.
    """
    scenarios = [
        {"number iterations": 1, "distribution": "normal", "correlation_structure": "identity", "snr": 7, "aspect_ratio": 0.5},
        {"number iterations": 1000, "distribution": "t", "correlation_structure": "ar1", "snr": 3, "aspect_ratio": 5.0},
    ]
    monkeypatch.setattr(analyze, "simulation_scenarios", lambda *args, **kwargs: scenarios)

    seen = []

    def spy_analyze_data(n_sim, params):
        # capture the call for inspection and return dummy (MSE, MCSE)
        seen.append((n_sim, params.copy()))
        return (0.0, 0.0)

    monkeypatch.setattr(analyze, "analyze_data", spy_analyze_data)

    base_seed = 42
    n = 200
    _ = analyze.simulate([1, 1000], "ignored", "ignored", "ignored", n=n, seed=base_seed)

    # two calls made
    assert len(seen) == 2

    # First scenario
    n_sim0, params0 = seen[0]
    assert n_sim0 == 1
    assert params0["n"] == n
    assert params0["aspect_ratio"] == pytest.approx(0.5)
    assert params0["seed"] == base_seed + 0
    assert params0["snr"] == 7
    assert params0["distribution"] == "normal"
    assert params0["correlation_structure"] == "identity"
    assert "df_t" in params0  # present (None by default)

    # Second scenario
    n_sim1, params1 = seen[1]
    assert n_sim1 == 1000
    assert params1["n"] == n
    assert params1["aspect_ratio"] == pytest.approx(5.0)
    assert params1["seed"] == base_seed + 1
    assert params1["snr"] == 3
    assert params1["distribution"] == "t"
    assert params1["correlation_structure"] == "ar1"


def test_run_all_delegates_and_returns(monkeypatch):
    # Sentinel DF to prove run_all returns exactly what simulate returns
    sentinel = pd.DataFrame(
        {"n_sim": [1, 100, 1000], "aspect_ratio": [0.5, 1.0, 2.0], "MSE": [0.0, 0.1, 0.2], "MCSE": [0.0, 0.01, 0.02]}
    )

    captured_args = {}
    def fake_simulate(n_sim, distributions, correlation_structures, snrs, n, seed):
        # capture the call for assertions
        captured_args["n_sim"] = n_sim
        captured_args["distributions"] = distributions
        captured_args["correlation_structures"] = correlation_structures
        captured_args["snrs"] = snrs
        captured_args["n"] = n
        captured_args["seed"] = seed
        return sentinel

    monkeypatch.setattr(analyze, "simulate", fake_simulate)

    out = analyze.run_all()

    # 1) Delegation arguments are exactly as specified in run_all
    assert captured_args == {
        "n_sim": [1, 100, 1000],
        "distributions": "normal",
        "correlation_structures": "identity",
        "snrs": 5,
        "n": 200,
        "seed": 1,
    }

    # 2) Return value is exactly simulate(...)'s return
    pd.testing.assert_frame_equal(out, sentinel)
