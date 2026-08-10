"""
backend/nodes/reliability/plot_weibull_km.py
=============================================
Generates 2x2 grid figure displaying:
  1. Kaplan-Meier empirical step curves with 95% Greenwood confidence bands
  2. Overlaid 2-Parameter Weibull theoretical fitted curves

Visualizes exactly where Weibull fits well (Progressive degradation)
and where it fails (Immediate point mass & heavily censored latent failures).
"""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Path bootstrap
_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from nodes.reliability.weibull_fitter import (
    GROUPS,
    fit_weibull_censored,
    kaplan_meier,
    weibull_survival,
)

_OUTPUT_PNG = _HERE / "output" / "weibull_km_groups.png"


def plot_weibull_km_overlay(df: pd.DataFrame, save_path: Path | None = None) -> Path:
    """
    Generate and save the 2x2 Kaplan-Meier vs Weibull overlay figure.

    Parameters
    ----------
    df : pd.DataFrame
        Life data containing failure_mode, ttf_seconds, and event.
    save_path : Path, optional
        Target PNG output path.

    Returns
    -------
    Path
        Absolute path to generated PNG figure.
    """
    out_path = Path(save_path or _OUTPUT_PNG)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=300)
    axes = axes.flatten()

    group_annotations = {
        "Immediate trigger": (
            "Point-mass behavior concentrated near ~10s.\n"
            "Weibull β is unnaturally large (poor continuous model fit)."
        ),
        "Fast accumulation": (
            "Infant/early accumulation (12–22s range).\n"
            "Discrete step artifacts present in empirical data."
        ),
        "Progressive resource degradation": (
            "Classic wear-out failure mechanism (12–84s continuous).\n"
            "BEST Weibull fit — KM and Weibull track closely."
        ),
        "Slow or latent degradation": (
            "Heavy right-censoring at 238s (~85% censored).\n"
            "High confidence interval spread; requires cure fraction model."
        ),
    }

    colors = ["#e63946", "#f4a261", "#2a9d8f", "#457b9d"]

    for idx, (group_name, failure_modes) in enumerate(GROUPS.items()):
        ax = axes[idx]
        subset = df[df["failure_mode"].isin(failure_modes)].copy()

        if subset.empty:
            ax.text(0.5, 0.5, f"No data for {group_name}", ha="center", va="center")
            continue

        time = subset["ttf_seconds"].values
        event = subset["event"].values

        # 1. Fit Weibull MLE
        fit = fit_weibull_censored(time, event)
        beta, eta = fit["beta"], fit["eta"]

        # 2. Compute Kaplan-Meier curve
        t_km, s_km, se_km = kaplan_meier(time, event)

        # Plot Kaplan-Meier step function
        ax.step(
            t_km,
            s_km,
            where="post",
            color=colors[idx],
            linewidth=2.2,
            label=f"Kaplan–Meier (n={fit['n']}, events={fit['events']})",
        )

        # Plot 95% CI band for KM
        upper_ci = np.clip(s_km + 1.96 * se_km, 0, 1)
        lower_ci = np.clip(s_km - 1.96 * se_km, 0, 1)
        ax.fill_between(t_km, lower_ci, upper_ci, color=colors[idx], alpha=0.18, step="post")

        # 3. Compute Weibull theoretical curve
        t_smooth = np.linspace(0.1, max(240.0, time.max()), 300)
        s_weibull = weibull_survival(t_smooth, beta, eta)

        ax.plot(
            t_smooth,
            s_weibull,
            "--",
            color="#1d3557",
            linewidth=2.0,
            label=f"Weibull Fit (β={beta:.2f}, η={eta:.1f}s)",
        )

        # Labels & Annotations
        ax.set_title(f"Group {idx+1}: {group_name}", fontsize=12, fontweight="bold", pad=8)
        ax.set_xlabel("Time-To-Failure t (seconds)", fontsize=10)
        ax.set_ylabel("Reliability S(t) = P(T > t)", fontsize=10)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="upper right", fontsize=9, framealpha=0.9)

        # Commentary Box
        note_text = group_annotations.get(group_name, "")
        ax.text(
            0.03,
            0.06,
            note_text,
            transform=ax.transAxes,
            fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#ffffff", edgecolor="#cccccc", alpha=0.85),
        )

    plt.suptitle(
        "AIOps Reliability Analysis: Stratified Kaplan–Meier vs. 2-Parameter Weibull Fits",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"[Plotter] KM + Weibull 4-Group plot saved -> {out_path}")
    return out_path
