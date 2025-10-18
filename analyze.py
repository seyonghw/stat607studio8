import pandas as pd
import numpy as np
import scipy


from dgp import generate_data
from estimator import fit_regression
from evaluate import evaluate


def analyze_data(n_sim, parameters):
    """
    Analyze the generated data using different regression methods and evaluate their performance.

    Parameters
    ----------
    n_sim: int
        Number of simulations to run.
    parameters : dict
        A dictionary containing parameters for data generation.

    Returns
    -------
    np.ndarray
        An array of evaluation metrics for each regression method.
    """
    n = parameters.get('n', 100)
    aspect_ratio = parameters.get('aspect_ratio', 1.0)
    correlation_structure = parameters.get('correlation_structure', 'identity')  # unused (kept for compat)
    rho = parameters.get('rho', 0.5)                                            # unused (kept for compat)
    seed = parameters.get('seed', 1)
    snr = parameters.get('snr', 1)
    distribution = parameters.get('distribution', 'normal')                      # unused (kept for compat)
    df_t = parameters.get('df_t', 1)                                          # unused (kept for compat)
    sigma2 = parameters.get('sigma2', 1.0)  # allow override; default stays 1.0

    r2 = parameters.get('r2', snr)

    ols_estimates = []
    coefficients = []

    p_expected = None  # track p to ensure consistent shapes across reps

    for sim in range(n_sim):
        np.random.seed(seed + sim)

        # Generate data
        data, beta = generate_data(
            n=n,
            gamma=float(aspect_ratio),
            r2=float(r2),
            sigma2=float(sigma2),
            seed=int(seed + sim)
        )
        # record true β (no intercept)
        coefficients.append(beta)

        # Fit models
        # --- fit OLS (old estimator may return [intercept, β...] or just β) ---
        est = fit_regression(data, method='ols')  # keep original call style

        est = np.asarray(est, dtype=float).ravel()
        # If estimator returned an intercept as first element, drop it
        if est.size == beta.size + 1:
            est = est[1:]
        # If estimator returned only coefs, keep as-is
        elif est.size == beta.size:
            pass
        else:
            # As a last resort, try to coerce by dropping/padding to match β length
            if est.size > beta.size:
                est = est[-beta.size:]  # take the last p entries (defensive)
            else:
                # pad with zeros (shouldn't normally happen)
                est = np.pad(est, (0, beta.size - est.size))

        # track/validate p consistency
        if p_expected is None:
            p_expected = beta.size
        elif est.size != p_expected or beta.size != p_expected:
            raise ValueError(
                f"Inconsistent p across reps: expected {p_expected}, "
                f"got est={est.size}, beta={beta.size}"
            )

        ols_estimates.append(est)

    coefficients = np.asarray(coefficients, dtype=float)  # (R, p)
    ols_estimates = np.asarray(ols_estimates, dtype=float)  # maybe (R, p+1) or (R, p)

    # drop intercept if present to get (R, p)
    if ols_estimates.shape[1] == coefficients.shape[1] + 1:
        ols_estimates = ols_estimates[:, 1:]
    elif ols_estimates.shape[1] != coefficients.shape[1]:
        raise ValueError(f"Shape mismatch: estimates have {ols_estimates.shape[1]} cols, "
                         f"but beta has {coefficients.shape[1]}.")

    # use a single true beta vector (same across reps)
    beta_true = coefficients[0]  # (p,)
    # (optional) sanity check they are all identical
    # assert np.allclose(coefficients, beta_true)

    mse, mcse = evaluate(beta_true, ols_estimates)  # beta_true: (p,), beta_hat: (R, p)
    return mse, mcse
    # # --- stack to (R, p) and evaluate ---
    # coefficients = np.asarray(coefficients, dtype=float)  # (R, p)
    # ols_estimates = np.asarray(ols_estimates, dtype=float)  # (R, p)
    #
    # # Now both have matching shape (R, p), so evaluate won't broadcast error
    # mse, mcse = evaluate(coefficients, ols_estimates)
    # return mse, mcse


def simulation_scenarios(n_sim, distribution, correlation_structure, snr):
    """
    Generate a list of simulation scenarios based on combinations of parameters.
    """
    scenarios = []
    for n in n_sim:
        if n == 1:
            aspect_ratios = np.logspace(-1, 1, 5000)
        elif n == 50:
            aspect_ratios = np.logspace(-1, 1, 100)
        else:
            aspect_ratios = [0.2, 0.5, 0.8, 2, 5]
        for ar in aspect_ratios:
            scenario = {
                "number iterations": n,
                "distribution": distribution,
                "correlation_structure": correlation_structure,
                "snr": snr,
                "aspect_ratio": ar
            }
            scenarios.append(scenario)
    return scenarios


# def simulate(n_sim, distributions, df_ts, correlation_structures, snrs, aspect_ratios, n=100, seed=1):
#     scenarios = simulation_scenarios(n_sim, distributions, df_ts, correlation_structures, snrs, aspect_ratios)
#     simulation_results = []
#     for i, scenario in enumerate(scenarios):
#         scenario_seed = seed + i
#         results = analyze_data(scenario["number iterations"], {
#             "n": n,
#             "aspect_ratio": scenario["aspect_ratio"],
#             "correlation_structure": scenario["correlation_structure"],
#             "rho": 0.5,
#             "seed": scenario_seed,
#             "snr": scenario["snr"],
#             "distribution": scenario["distribution"],
#             "df_t": scenario.get("df_t", None)
#         })
#         simulation_results.append([scenario, results])
#     return simulation_results

def simulate(n_sim, distributions, correlation_structures, snrs, n=100, seed=1):
    """
    Run Monte Carlo simulations across scenarios and return results as a DataFrame.

    Parameters
    ----------
    n_sim : int
        Number of simulations per scenario.
    distributions, df_ts, correlation_structures, snrs, aspect_ratios : list
        Lists of parameter values to iterate over.
    n : int
        Sample size.
    seed : int
        Base random seed.

    Returns
    -------
    pd.DataFrame
        A DataFrame summarizing results with columns:
        ['model', 'distribution', 'df_t', 'correlation_structure', 'snr',
         'aspect_ratio', 'n_sim', 'MSE', 'MCSE']
    """
    scenarios = simulation_scenarios(n_sim, distributions, correlation_structures, snrs)
    rows = []
    #model_names = ["OLS", "Huber", "LAD"]

    for i, scenario in enumerate(scenarios):
        scenario_seed = seed + i
        results = analyze_data(scenario["number iterations"], {
            "n": n,
            "aspect_ratio": scenario["aspect_ratio"],
            "correlation_structure": scenario["correlation_structure"],
            "rho": 0.5,
            "seed": scenario_seed,
            "snr": scenario["snr"],
            "distribution": scenario["distribution"],
            "df_t": scenario.get("df_t", None)
        })

        # results = [(mse, mcse), (mse, mcse), (mse, mcse)]
        mse, mcse = results
        rows.append({
            "n_sim": scenario["number iterations"],
            "aspect_ratio": scenario["aspect_ratio"],
            "MSE": mse,
            "MCSE": mcse
        })

    df = pd.DataFrame(rows)
    return df


def run_all():
    """
    Run the canonical set of scenarios and return a DataFrame of results.

    Fixed settings:
      n_sim = [1, 50, 1000], distributions = "normal",
      correlation_structures = "identity", snrs = 5,
      n = 200, seed = 1
    """
    n_sim = [1, 50, 1000]
    distributions = "normal"
    correlation_structures = "identity"
    snrs = 5
    all_results = simulate(n_sim, distributions, correlation_structures, snrs, n=200, seed=1)
    return all_results