"""
app_simulator/offline/visualize_forecasting.py
===============================================
Rich visualization dashboard for the Forecasting Node output.

Generates 6 plots saved to pipeline/output/forecasting_plots/:
  1. Confidence Distribution       — histogram of forecast_confidence per mode
  2. Algorithm Distribution        — pie chart of which algorithm ran most
  3. TTF Distribution per Mode     — boxplot of time_to_failure (seconds)
  4. Confidence vs TTF Scatter     — scatter: does higher confidence = better TTF?
  5. Feature Forecast Trajectory   — line chart: current + predicted for one episode
  6. Threshold Crossing Rate       — bar: % of episodes that crossed threshold per mode

Usage:
    python app_simulator/offline/visualize_forecasting.py
    python app_simulator/offline/visualize_forecasting.py --episode ep_RETRY_STORM_001
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")  # non-interactive backend (saves to file)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

from app_data_generator.config import FORECASTING_OUTPUT_CSV, PIPELINE_OUTPUT_DIR

PLOT_DIR = PIPELINE_OUTPUT_DIR / "forecasting_plots"

# ── Colour palette per failure mode ──────────────────────────────────────────
MODE_COLORS = {
    "MEMORY_LEAK":        "#e74c3c",
    "CPU_SATURATION":     "#e67e22",
    "LATENCY_SPIKE":      "#f39c12",
    "DB_SLOWDOWN":        "#3498db",
    "CACHE_STAMPEDE":     "#9b59b6",
    "QUEUE_BACKUP":       "#1abc9c",
    "DEPENDENCY_TIMEOUT": "#2980b9",
    "BAD_DEPLOY":         "#c0392b",
    "BAD_DEPLOYMENT":     "#c0392b",
    "ERROR_STORM":        "#e91e63",
    "RETRY_STORM":        "#ff5722",
    "DISK_IO_SATURATION": "#795548",
    "CASCADING_FAILURE":  "#6c3483",
}

ALGO_COLORS = {
    "pmdarima":   "#27ae60",
    "exponential":"#e67e22",
    "linear":     "#3498db",
    "constant":   "#95a5a6",
}


def _mode_color(mode: str) -> str:
    return MODE_COLORS.get(mode, "#7f8c8d")


def _algo_color(algo: str) -> str:
    for key, color in ALGO_COLORS.items():
        if key in algo.lower():
            return color
    return "#7f8c8d"


def load_data() -> pd.DataFrame:
    if not FORECASTING_OUTPUT_CSV.exists():
        print(f"ERROR: {FORECASTING_OUTPUT_CSV} not found.")
        print("Run: python app_simulator/offline/run_forecasting.py first.")
        sys.exit(1)
    df = pd.read_csv(FORECASTING_OUTPUT_CSV)
    print(f"Loaded {len(df):,} rows from {FORECASTING_OUTPUT_CSV.name}")
    return df


# =============================================================================
# PLOT 1: Confidence Distribution per Mode
# =============================================================================
def plot_confidence_distribution(df: pd.DataFrame, save_dir: Path) -> None:
    modes = sorted(df["failure_mode"].unique())
    fig, axes = plt.subplots(3, 4, figsize=(20, 14))
    fig.suptitle("Forecast Confidence Distribution per Failure Mode", fontsize=16, fontweight="bold", y=1.01)
    axes_flat = axes.flatten()

    for i, mode in enumerate(modes[:12]):
        ax = axes_flat[i]
        data = df[df["failure_mode"] == mode]["forecast_confidence"].dropna()
        if len(data):
            ax.hist(data, bins=20, color=_mode_color(mode), alpha=0.85, edgecolor="white", linewidth=0.5)
            ax.axvline(data.mean(), color="white", linestyle="--", linewidth=1.5, label=f"mean={data.mean():.3f}")
            ax.axvline(data.median(), color="yellow", linestyle=":", linewidth=1.5, label=f"median={data.median():.3f}")
            ax.legend(fontsize=7, loc="upper left")
        ax.set_title(mode.replace("_", " "), fontsize=9, fontweight="bold", color=_mode_color(mode))
        ax.set_xlabel("Confidence Score", fontsize=8)
        ax.set_ylabel("Count", fontsize=8)
        ax.set_xlim(0, 1)
        ax.set_facecolor("#1a1a2e")
        ax.tick_params(labelsize=7, colors="white")
        ax.spines[:].set_color("#444")

    # Hide unused subplots
    for j in range(len(modes), 12):
        axes_flat[j].set_visible(False)

    fig.patch.set_facecolor("#0f0f1a")
    plt.tight_layout()
    path = save_dir / "1_confidence_distribution.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {path.name}")


# =============================================================================
# PLOT 2: Algorithm Distribution (Pie)
# =============================================================================
def plot_algorithm_distribution(df: pd.DataFrame, save_dir: Path) -> None:
    # Simplify algorithm names for display
    def simplify_algo(a):
        if "pmdarima" in str(a): return "pmdarima (Auto-ARIMA)"
        if "statsmodels" in str(a): return "statsmodels ARIMA"
        if "exponential" in str(a): return "Exponential Regression"
        if "linear" in str(a): return "Linear Regression"
        return "Constant / Unknown"

    algo_series = df["algorithm_used"].apply(simplify_algo)
    counts = algo_series.value_counts()

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor("#0f0f1a")
    ax.set_facecolor("#0f0f1a")

    colors = [
        "#27ae60" if "pmdarima" in l else
        "#3498db" if "statsmodels" in l else
        "#e67e22" if "Exponential" in l else
        "#9b59b6" if "Linear" in l else
        "#95a5a6"
        for l in counts.index
    ]

    wedges, texts, autotexts = ax.pie(
        counts.values,
        labels=counts.index,
        autopct="%1.1f%%",
        colors=colors,
        startangle=140,
        wedgeprops={"edgecolor": "#0f0f1a", "linewidth": 2},
        textprops={"color": "white", "fontsize": 11},
    )
    for at in autotexts:
        at.set_fontsize(10)
        at.set_fontweight("bold")

    ax.set_title("Algorithm Used Across All Episodes", fontsize=14, fontweight="bold",
                 color="white", pad=20)

    # Legend
    legend_patches = [mpatches.Patch(color=c, label=f"{l}  ({n:,})")
                      for l, n, c in zip(counts.index, counts.values, colors)]
    ax.legend(handles=legend_patches, loc="lower center", bbox_to_anchor=(0.5, -0.12),
              ncol=2, fontsize=9, framealpha=0, labelcolor="white")

    plt.tight_layout()
    path = save_dir / "2_algorithm_distribution.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {path.name}")


# =============================================================================
# PLOT 3: TTF Distribution per Mode (Boxplot)
# =============================================================================
def plot_ttf_distribution(df: pd.DataFrame, save_dir: Path) -> None:
    ttf_df = df[df["time_to_failure"].notna() & (df["time_to_failure"] >= 0)].copy()
    ttf_df["ttf_min"] = ttf_df["time_to_failure"] / 60.0

    modes = sorted(ttf_df["failure_mode"].unique())
    data_by_mode = [ttf_df[ttf_df["failure_mode"] == m]["ttf_min"].values for m in modes]
    colors_list  = [_mode_color(m) for m in modes]

    fig, ax = plt.subplots(figsize=(16, 7))
    fig.patch.set_facecolor("#0f0f1a")
    ax.set_facecolor("#1a1a2e")

    bp = ax.boxplot(data_by_mode, patch_artist=True, notch=False,
                    whiskerprops={"color": "white", "linewidth": 1.2},
                    capprops={"color": "white", "linewidth": 1.5},
                    medianprops={"color": "yellow", "linewidth": 2.5},
                    flierprops={"marker": "o", "markersize": 3, "alpha": 0.4})

    for patch, color in zip(bp["boxes"], colors_list):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
        patch.set_edgecolor("white")

    ax.set_xticks(range(1, len(modes) + 1))
    ax.set_xticklabels([m.replace("_", "\n") for m in modes], fontsize=8, color="white")
    ax.set_ylabel("Time to Failure (minutes)", fontsize=11, color="white")
    ax.set_title("Predicted Time to Failure Distribution per Failure Mode", fontsize=13,
                 fontweight="bold", color="white")
    ax.tick_params(colors="white", labelsize=8)
    ax.spines[:].set_color("#444")
    ax.yaxis.grid(True, alpha=0.3, color="#555")

    plt.tight_layout()
    path = save_dir / "3_ttf_distribution.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {path.name}")


# =============================================================================
# PLOT 4: Confidence vs TTF Scatter
# =============================================================================
def plot_confidence_vs_ttf(df: pd.DataFrame, save_dir: Path) -> None:
    plot_df = df[df["time_to_failure"].notna() & (df["time_to_failure"] >= 0)].copy()
    plot_df["ttf_min"] = plot_df["time_to_failure"] / 60.0

    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor("#0f0f1a")
    ax.set_facecolor("#1a1a2e")

    modes = plot_df["failure_mode"].unique()
    for mode in modes:
        sub = plot_df[plot_df["failure_mode"] == mode]
        ax.scatter(sub["forecast_confidence"], sub["ttf_min"],
                   color=_mode_color(mode), alpha=0.6, s=30, label=mode.replace("_", " "),
                   edgecolors="none")

    ax.set_xlabel("Forecast Confidence Score", fontsize=12, color="white")
    ax.set_ylabel("Time to Failure (minutes)", fontsize=12, color="white")
    ax.set_title("Forecast Confidence vs. Predicted Time to Failure", fontsize=13,
                 fontweight="bold", color="white")
    ax.set_xlim(0, 1)
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#444")
    ax.xaxis.grid(True, alpha=0.2, color="#555")
    ax.yaxis.grid(True, alpha=0.2, color="#555")

    legend = ax.legend(loc="upper right", fontsize=7, framealpha=0.2,
                       labelcolor="white", markerscale=1.5, ncol=2)
    legend.get_frame().set_edgecolor("#555")

    plt.tight_layout()
    path = save_dir / "4_confidence_vs_ttf.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {path.name}")


# =============================================================================
# PLOT 5: Feature Forecast Trajectory for one episode
# =============================================================================
def plot_feature_trajectory(df: pd.DataFrame, save_dir: Path, episode_id: str | None = None) -> None:
    """Show predicted feature trajectories + critical thresholds for one episode."""
    if episode_id:
        row = df[df["episode_id"] == episode_id]
    else:
        # Pick a CASCADING_FAILURE episode with threshold_crossed=True for visual interest
        for mode in ["CASCADING_FAILURE", "RETRY_STORM", "MEMORY_LEAK", "DB_SLOWDOWN"]:
            row = df[(df["failure_mode"] == mode) & (df["threshold_crossed"] == True)]
            if not row.empty:
                break
        if row.empty:
            row = df[df["threshold_crossed"] == True]
        if row.empty:
            row = df.head(1)

    row = row.iloc[0]
    failure_mode = row["failure_mode"]
    ep_id        = row["episode_id"]

    try:
        predictions  = json.loads(row["predictions_json"])
        thresholds   = json.loads(row["critical_thresholds_json"])
        curr_vals    = json.loads(row["current_values_json"])
        feature_ttfs = json.loads(row["feature_ttfs_json"])
    except Exception as e:
        print(f"  WARNING: Could not parse JSON for {ep_id}: {e}")
        return

    if not predictions:
        print("  No prediction data available for trajectory plot.")
        return

    # Only plot critical features (ones with thresholds)
    crit_features = list(thresholds.keys())
    n_feats = len(crit_features)
    if n_feats == 0:
        return

    cols = min(n_feats, 3)
    rows = (n_feats + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows + 1))
    if n_feats == 1:
        axes = [axes]
    elif rows == 1:
        axes = list(axes)
    else:
        axes = [ax for row_ in axes for ax in row_]

    fig.suptitle(
        f"Feature Forecast Trajectory\n{failure_mode.replace('_', ' ')}  |  {ep_id}",
        fontsize=13, fontweight="bold", color="white", y=1.01
    )
    fig.patch.set_facecolor("#0f0f1a")

    timestamps = [i * 2 for i in range(1, 11)]   # 2, 4, ..., 20 seconds ahead
    mode_color = _mode_color(failure_mode)

    for i, feat in enumerate(crit_features):
        ax = axes[i]
        ax.set_facecolor("#1a1a2e")

        pred_vals = predictions.get(feat, [])
        threshold = thresholds.get(feat)
        curr_val  = curr_vals.get(feat, 0.0)
        ttf_sec   = feature_ttfs.get(feat)

        if pred_vals:
            # Current value marker (t=0)
            ax.plot(0, curr_val, "o", color="white", markersize=8, zorder=5, label="Current")
            # Forecast trajectory
            ax.plot(timestamps, pred_vals, color=mode_color, linewidth=2.5,
                    marker="o", markersize=4, label="Forecast")
            # Shaded uncertainty band (±5%)
            pred_arr = np.array(pred_vals)
            ax.fill_between(timestamps,
                             pred_arr * 0.95, pred_arr * 1.05,
                             alpha=0.2, color=mode_color)

        # Critical threshold line
        if threshold is not None:
            ax.axhline(threshold, color="#e74c3c", linestyle="--", linewidth=1.8,
                       label=f"Threshold: {threshold:,.0f}")

        # TTF marker
        if ttf_sec is not None and 0 <= ttf_sec <= 25:
            ax.axvline(ttf_sec, color="yellow", linestyle=":", linewidth=1.5,
                       label=f"TTF: {ttf_sec:.1f}s")

        feat_label = feat.replace("_", " ").title()
        ax.set_title(feat_label, fontsize=9, color="white", fontweight="bold")
        ax.set_xlabel("Seconds ahead", fontsize=8, color="#aaa")
        ax.set_ylabel("Value", fontsize=8, color="#aaa")
        ax.tick_params(colors="white", labelsize=7)
        ax.spines[:].set_color("#444")
        ax.xaxis.grid(True, alpha=0.2, color="#555")
        ax.yaxis.grid(True, alpha=0.2, color="#555")
        ax.legend(fontsize=7, loc="upper left", framealpha=0.2, labelcolor="white")

    # Hide unused axes
    for j in range(n_feats, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    path = save_dir / "5_feature_trajectory.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {path.name}  (episode: {ep_id}, mode: {failure_mode})")


# =============================================================================
# PLOT 6: Threshold Crossing Rate per Mode
# =============================================================================
def plot_threshold_crossing_rate(df: pd.DataFrame, save_dir: Path) -> None:
    modes  = sorted(df["failure_mode"].unique())
    rates  = []
    counts = []
    for m in modes:
        sub = df[df["failure_mode"] == m]
        rate = sub["threshold_crossed"].mean() * 100
        rates.append(rate)
        counts.append(len(sub))

    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor("#0f0f1a")
    ax.set_facecolor("#1a1a2e")

    x = np.arange(len(modes))
    bars = ax.bar(x, rates, color=[_mode_color(m) for m in modes],
                  edgecolor="white", linewidth=0.5, width=0.65, alpha=0.9)

    # Annotate with episode count
    for bar_, rate, cnt in zip(bars, rates, counts):
        ax.text(bar_.get_x() + bar_.get_width() / 2,
                bar_.get_height() + 1.5,
                f"{rate:.0f}%\n(n={cnt})",
                ha="center", va="bottom", fontsize=8, color="white", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", "\n") for m in modes], fontsize=8, color="white")
    ax.set_ylabel("Episodes with TTF Prediction (%)", fontsize=11, color="white")
    ax.set_title("Threshold Crossing Rate — % of Episodes where TTF was Predicted",
                 fontsize=13, fontweight="bold", color="white")
    ax.set_ylim(0, 115)
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#444")
    ax.yaxis.grid(True, alpha=0.3, color="#555")

    plt.tight_layout()
    path = save_dir / "6_threshold_crossing_rate.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {path.name}")


# =============================================================================
# MAIN
# =============================================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize forecasting node output.")
    parser.add_argument("--episode", type=str, default=None,
                        help="Specific episode_id for trajectory plot (Plot 5)")
    args = parser.parse_args()

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data()

    print(f"\nGenerating plots -> {PLOT_DIR}\n")

    print("Plot 1: Confidence Distribution ...")
    plot_confidence_distribution(df, PLOT_DIR)

    print("Plot 2: Algorithm Distribution ...")
    plot_algorithm_distribution(df, PLOT_DIR)

    print("Plot 3: TTF Distribution (Boxplot) ...")
    plot_ttf_distribution(df, PLOT_DIR)

    print("Plot 4: Confidence vs TTF Scatter ...")
    plot_confidence_vs_ttf(df, PLOT_DIR)

    print("Plot 5: Feature Forecast Trajectory ...")
    plot_feature_trajectory(df, PLOT_DIR, episode_id=args.episode)

    print("Plot 6: Threshold Crossing Rate ...")
    plot_threshold_crossing_rate(df, PLOT_DIR)

    print(f"\n=== All 6 plots saved to: {PLOT_DIR} ===")
    for p in sorted(PLOT_DIR.glob("*.png")):
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
