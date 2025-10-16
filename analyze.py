import pandas as pd
import numpy as np
import scipy
from sklearn.linear_model import LinearRegression, HuberRegressor, QuantileRegressor

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
    aspect_ratio = parameters.get('aspect_ratio', 1)
    correlation_structure = parameters.get('correlation_structure', 'identity')
    rho = parameters.get('rho', 0.5)
    seed = parameters.get('seed', 1)
    snr = parameters.get('snr', 1)
    distribution = parameters.get('distribution', 'normal')
    df_t = parameters.get('df_t', 1)

    ols_estimates = []
    coefficients = []
    for sim in range(n_sim):
        np.random.seed(seed + sim)

        # Generate data
        data, beta0, beta = generate_data(n, aspect_ratio, snr, sigma2=1, seed=(seed+sim))
    
        # Fit models
        ols_estimates.append(fit_regression(data, method='ols'))

        coefficients.append(np.concatenate((beta0, beta)))
    

    coefficients = np.array(coefficients)
    ols_estimates = np.array(ols_estimates)

    ols_results = evaluate(coefficients, ols_estimates)
    results = ols_results

    return results

def simulation_scenarios(n_sim, distribution, correlation_structure, snr):
    """
    Generate a list of simulation scenarios based on combinations of parameters.
    """
    scenarios = []
    for n in n_sim:
        if n == 1:
            aspect_ratios = np.logspace(0.1, 10, 5000)
        elif n==50:
            aspect_ratios = np.logspace(0.1, 10, 50)
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
    model_names = ["OLS", "Huber", "LAD"]

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
        for model_name, (mse, mcse) in zip(model_names, results):
            rows.append({
                "n_sim": scenario["number iterations"],
                "aspect_ratio": scenario["aspect_ratio"],
                "MSE": mse,
                "MCSE": mcse
            })

    df = pd.DataFrame(rows)
    return df

def run_all():
    n_sim = [1, 100, 1000]
    distributions = "normal"
    correlation_structures = "identity"
    snrs = 5
    all_results = simulate(n_sim, distributions, correlation_structures, snrs, n=200, seed=1)
    return all_results
