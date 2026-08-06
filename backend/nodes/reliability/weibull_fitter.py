"""
backend/nodes/reliability/weibull_fitter.py
============================================
Stratified 2-Parameter Weibull Fitter & Kaplan-Meier Estimator.

Implements Maximum Likelihood Estimation (MLE) for right-censored life data
across the four failure mode groupings requested by your mentor:

  1. Immediate-trigger failures
  2. Fast accumulation failures
  3. Progressive resource degradation
  4. Slow or latent degradation
"""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize

# Path bootstrap
_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Mentor's exact 4-group mapping definition
GROUPS: dict[str, list[str]] = {
    "Immediate trigger": [
        "BAD_DEPLOY",
        "CACHE_STAMPEDE",
        "CASCADING_FAILURE",
        "CPU_SATURATION",
        "DEPENDENCY_TIMEOUT",
        "ERROR_STORM",
    ],
    "Fast accumulation": [
        "QUEUE_BACKUP",
        "RETRY_STORM",
    ],
    "Progressive resource degradation": [
        "DISK_IO_SATURATION",
    ],
    "Slow or latent degradation": [
        "DB_SLOWDOWN",
        "MEMORY_LEAK",
        "LATENCY_SPIKE",
    ],
}


def fit_weibull_censored(time: np.ndarray, event: np.ndarray) -> dict:
    """
    Fit a 2-Parameter Weibull distribution to right-censored data via MLE.

    Primary Method: `reliability` (Reid) or `lifelines` (Davidson-Pilon)
    Fallback Method: `scipy.optimize.minimize` (Native MLE)

    Parameters
    ----------
    time  : array-like of float, shape (N,)
        Observed durations (TTF or censoring time).
    event : array-like of int, shape (N,)
        1 = failure occurred, 0 = right-censored.

    Returns
    -------
    dict
        Fitted parameters and fit statistics:
        {'beta', 'eta', 'n', 'events', 'censored', 'log_likelihood', 'method_used'}
    """
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=int)

    if np.any(time <= 0):
        raise ValueError("All durations must be positive.")

    # -----------------------------------------------------------------------
    # Primary Method 1: Reid's `reliability` library
    # -----------------------------------------------------------------------
    try:
        from reliability.Fitters import Fit_Weibull_2P
        failures = time[event == 1]
        right_censored = time[event == 0]

        fit = Fit_Weibull_2P(
            failures=failures if len(failures) > 0 else None,
            right_censored=right_censored if len(right_censored) > 0 else None,
            show_probability_plot=False,
            print_results=False,
        )
        return {
            "beta": float(fit.beta),
            "eta": float(fit.alpha),
            "n": int(len(time)),
            "events": int(event.sum()),
            "censored": int((event == 0).sum()),
            "log_likelihood": float(fit.loglik),
            "method_used": "reliability_reid",
        }
    except Exception:
        pass

    # -----------------------------------------------------------------------
    # Primary Method 2: Cameron Davidson-Pilon's `lifelines` library
    # -----------------------------------------------------------------------
    try:
        from lifelines import WeibullFitter
        wf = WeibullFitter()
        wf.fit(durations=time, event_observed=event)
        return {
            "beta": float(wf.rho_),
            "eta": float(wf.lambda_),
            "n": int(len(time)),
            "events": int(event.sum()),
            "censored": int((event == 0).sum()),
            "log_likelihood": float(wf.log_likelihood_),
            "method_used": "lifelines",
        }
    except Exception:
        pass

    # -----------------------------------------------------------------------
    # Fallback Method: Native SciPy MLE Optimization
    # -----------------------------------------------------------------------
    def negative_log_likelihood(log_params):
        log_beta, log_eta = log_params

        beta = np.exp(log_beta)
        eta = np.exp(log_eta)

        log_t_eta = np.log(time) - log_eta
        # Clip exponent to prevent overflow for very large beta (e.g. Group 1)
        cumulative_hazard = np.exp(np.clip(beta * log_t_eta, -500.0, 500.0))

        # Observed failure contributes log hazard + log survival.
        # Censored observation contributes only log survival.
        log_likelihood = np.sum(
            event
            * (
                log_beta
                - log_eta
                + (beta - 1.0) * log_t_eta
            )
            - cumulative_hazard
        )

        return -log_likelihood

    observed = time[event == 1]

    initial_eta = (
        float(np.median(observed))
        if len(observed) > 0
        else float(np.median(time))
    )

    result = minimize(
        negative_log_likelihood,
        x0=np.log([2.0, max(0.1, initial_eta)]),
        method="L-BFGS-B",
        bounds=[
            (np.log(0.05), np.log(1000.0)),
            (np.log(0.1), np.log(10000.0)),
        ],
    )

    if not result.success:
        # Fallback to simple moment-matching initial guess if optimization hit bound
        result = minimize(
            negative_log_likelihood,
            x0=np.log([1.0, float(np.mean(time))]),
            method="Nelder-Mead",
        )

    beta, eta = np.exp(result.x)

    return {
        "beta": float(beta),
        "eta": float(eta),
        "n": int(len(time)),
        "events": int(event.sum()),
        "censored": int((event == 0).sum()),
        "log_likelihood": float(-result.fun),
        "method_used": "scipy_mle",
    }


def kaplan_meier(time: np.ndarray, event: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute non-parametric Kaplan–Meier survival estimate S(t) and 95% Greenwood CI bounds.

    Returns
    -------
    t_steps  : 1D array of time steps starting from 0.0
    s_steps  : 1D array of survival probabilities S(t)
    se_steps : 1D array of Greenwood standard errors
    """
    df = pd.DataFrame({"time": time, "event": event}).sort_values("time")
    
    unique_times = df["time"].unique()
    n_at_risk = len(df)
    
    t_list = [0.0]
    s_list = [1.0]
    var_sum = 0.0
    se_list = [0.0]

    current_s = 1.0

    for t in unique_times:
        sub = df[df["time"] == t]
        d_i = sub["event"].sum()  # number of failures at t
        c_i = (sub["event"] == 0).sum()  # number of censored at t
        n_i = n_at_risk  # number at risk just before t

        if n_i > 0 and d_i > 0:
            current_s *= (1.0 - d_i / n_i)
            if n_i > d_i:
                var_sum += d_i / (n_i * (n_i - d_i))

        se = current_s * np.sqrt(var_sum)

        t_list.append(float(t))
        s_list.append(float(current_s))
        se_list.append(float(se))

        n_at_risk -= (d_i + c_i)

    return np.array(t_list), np.array(s_list), np.array(se_list)


def weibull_survival(t: np.ndarray, beta: float, eta: float) -> np.ndarray:
    """Evaluate Weibull survival function R(t) = exp(-(t/eta)^beta)."""
    t = np.asarray(t, dtype=float)
    return np.exp(- (t / eta) ** beta)


def fit_all_groups(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fit 2-Parameter Weibull models to all 4 failure groups.

    Parameters
    ----------
    df : pd.DataFrame
        Extracted life data containing failure_mode, ttf_seconds, and event columns.

    Returns
    -------
    pd.DataFrame
        Summary results table containing beta, eta, log_likelihood, and counts per group.
    """
    results = []

    for group_name, failure_modes in GROUPS.items():
        subset = df[df["failure_mode"].isin(failure_modes)].copy()

        if subset.empty:
            continue

        fit = fit_weibull_censored(
            subset["ttf_seconds"].values,
            subset["event"].values,
        )

        results.append({
            "group": group_name,
            **fit,
        })

    return pd.DataFrame(results)
