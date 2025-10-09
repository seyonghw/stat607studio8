import pandas as pd
import numpy as np
import scipy
from sklearn.linear_model import LinearRegression, HuberRegressor, QuantileRegressor

from dgp import generate_design_matrix, generate_coefficients, generate_response, generate_data
from estimator import fit_regression
from evaluate import evaluate

def analyze_data(parameters):
    """
    Analyze the generated data using different regression methods and evaluate their performance.

    Parameters
    ----------
    parameters : dict
        A dictionary containing parameters for data generation and analysis.

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

    # Generate data
    data, beta0, beta = generate_data(n, aspect_ratio, correlation_structure, rho, seed, snr, distribution, df_t)
    
    # Fit different regression models
    methods = {
        'OLS': LinearRegression(),
        'Huber': HuberRegressor(),
        'Quantile_0.5': QuantileRegressor(quantile=0.5)
    }

    ols_estimates = fit_regression(data, method='ols')
    huber_estimates = fit_regression(data, method='huber')
    lad_estimates = fit_regression(data, method='lad')

    results = [evaluate(ols_estimates, [beta0, beta]),
               evaluate(huber_estimates, [beta0, beta]),
               evaluate(lad_estimates, [beta0, beta])]

    return np.ndarray(results)

def simulation_scenarios(distributions, df_ts, correlation_structures, snrs, aspect_ratios):
    """
    Generate a list of simulation scenarios based on combinations of parameters.
    """
    scenarios = []
    for distribution in distributions:
        if distribution == "normal":
            for correlation_structure in correlation_structures:
                for snr in snrs:
                    for aspect_ratio in aspect_ratios:
                        scenario = {
                            "distribution": distribution,
                            "correlation_structure": correlation_structure,
                            "snr": snr,
                            "aspect_ratio": aspect_ratio
                        }
                        scenarios.append(scenario)
        elif distribution == "t":
            for df_t in df_ts:
                for correlation_structure in correlation_structures:
                    for snr in snrs:
                        for aspect_ratio in aspect_ratios:
                            scenario = {
                                "distribution": distribution,
                                "df_t": df_t,
                                "correlation_structure": correlation_structure,
                                "snr": snr,
                                "aspect_ratio": aspect_ratio
                            }
                            scenarios.append(scenario)
    return scenarios


def simulate(distributions, df_ts, correlation_structures, snrs, aspect_ratios, n=100, seed=1):
    scenarios = simulation_scenarios(distributions, df_ts, correlation_structures, snrs, aspect_ratios)
    simulation_results = []
    for i, scenario in enumerate(scenarios):
        scenario_seed = seed + i
        results = analyze_data({
            "n": n,
            "aspect_ratio": scenario["aspect_ratio"],
            "correlation_structure": scenario["correlation_structure"],
            "rho": 0.5,
            "seed": scenario_seed,
            "snr": scenario["snr"],
            "distribution": scenario["distribution"],
            "df_t": scenario.get("df_t", None)
        })
    simulation_results.append(scenario, results)
    return simulation_results

def run_all()
    distributions = ["normal", "t"]
    df_ts = [1, 2, 3, 20]
    correlation_structures = ["identity", "autoregressive"]
    snrs = [1, 5, 10]
    aspect_ratios = [0.5, 1, 2]
    all_results = simulate(distributions, df_ts, correlation_structures, snrs, aspect_ratios, n=100, seed=1)
    return all_results
