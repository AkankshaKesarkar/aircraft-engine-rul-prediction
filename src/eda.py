"""
eda.py
Exploratory Data Analysis for aircraft engine sensor data.
Generates annotated charts for stakeholder storytelling.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import os

matplotlib.use("Agg")

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SENSOR_COLS = [f"sensor_{i}" for i in range(1, 22)]


def run_eda(df: pd.DataFrame):
    """Run full EDA pipeline and save all charts."""
    print("\n[EDA] Starting Exploratory Data Analysis...")
    print(f"[EDA] Dataset shape    : {df.shape}")
    print(f"[EDA] Engines          : {df['engine_id'].nunique()}")
    print(f"[EDA] Total cycles     : {len(df)}")
    print(f"[EDA] RUL range        : {df['RUL'].min()} → {df['RUL'].max()}")
    print(f"[EDA] Missing values   : {df.isnull().sum().sum()}")

    _plot_rul_distribution(df)
    _plot_sensor_trends(df)
    _plot_correlation_heatmap(df)
    _plot_engine_lifecycle(df)
    _print_statistical_summary(df)

    print("[EDA] All EDA charts saved to outputs/\n")


def _plot_rul_distribution(df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("RUL Distribution — Engine Fleet Overview", fontsize=14, fontweight="bold")

    # Histogram
    axes[0].hist(df["RUL"], bins=50, color="#2196F3", edgecolor="white", alpha=0.85)
    axes[0].set_title("Distribution of RUL Across All Cycles")
    axes[0].set_xlabel("Remaining Useful Life (cycles)")
    axes[0].set_ylabel("Frequency")
    axes[0].axvline(df["RUL"].mean(), color="red", linestyle="--",
                    label=f"Mean RUL = {df['RUL'].mean():.1f}")
    axes[0].legend()

    # Engine life distribution
    engine_life = df.groupby("engine_id")["cycle"].max()
    axes[1].hist(engine_life, bins=30, color="#4CAF50", edgecolor="white", alpha=0.85)
    axes[1].set_title("Engine Lifespan Distribution")
    axes[1].set_xlabel("Total Cycles to Failure")
    axes[1].set_ylabel("Number of Engines")
    axes[1].axvline(engine_life.mean(), color="red", linestyle="--",
                    label=f"Mean life = {engine_life.mean():.0f} cycles")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/rul_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[EDA] Saved → rul_distribution.png")


def _plot_sensor_trends(df):
    """Plot degrading sensor trends for a sample engine."""
    sample_engine = df[df["engine_id"] == 1].sort_values("cycle")
    degrading = ["sensor_2", "sensor_3", "sensor_4", "sensor_7",
                 "sensor_8", "sensor_11", "sensor_12", "sensor_15"]

    fig, axes = plt.subplots(4, 2, figsize=(16, 14))
    fig.suptitle("Sensor Degradation Trends — Engine #1 Lifecycle",
                 fontsize=14, fontweight="bold")
    axes = axes.flatten()

    colors = ["#E53935", "#1E88E5", "#43A047", "#FB8C00",
              "#8E24AA", "#00ACC1", "#F4511E", "#6D4C41"]

    for idx, (sensor, color) in enumerate(zip(degrading, colors)):
        ax = axes[idx]
        ax.plot(sample_engine["cycle"], sample_engine[sensor],
                color=color, linewidth=1.2, alpha=0.8)
        # Rolling mean
        rolling = sample_engine[sensor].rolling(window=10).mean()
        ax.plot(sample_engine["cycle"], rolling,
                color="black", linewidth=2, linestyle="--", label="Rolling avg (10)")
        ax.set_title(sensor.replace("_", " ").title(), fontsize=10)
        ax.set_xlabel("Flight Cycle")
        ax.set_ylabel("Sensor Reading")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/eda_sensor_trends.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[EDA] Saved → eda_sensor_trends.png")


def _plot_correlation_heatmap(df):
    fig, ax = plt.subplots(figsize=(18, 14))
    corr_data = df[SENSOR_COLS + ["RUL", "cycle"]]
    corr      = corr_data.corr()

    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, ax=ax,
        cmap="coolwarm", center=0,
        annot=True, fmt=".2f", annot_kws={"size": 7},
        linewidths=0.5, linecolor="white",
        cbar_kws={"shrink": 0.8}
    )
    ax.set_title("Sensor Correlation Heatmap — Feature Relationship Analysis",
                 fontsize=14, fontweight="bold", pad=20)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/correlation_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[EDA] Saved → correlation_heatmap.png")


def _plot_engine_lifecycle(df):
    """Show RUL decay curves for a sample of engines."""
    fig, ax = plt.subplots(figsize=(14, 6))
    sample_engines = df["engine_id"].unique()[:20]
    cmap = plt.cm.get_cmap("tab20", len(sample_engines))

    for i, eid in enumerate(sample_engines):
        engine_df = df[df["engine_id"] == eid].sort_values("cycle")
        ax.plot(engine_df["cycle"], engine_df["RUL"],
                color=cmap(i), linewidth=1.2, alpha=0.7)

    ax.set_title("RUL Decay Curves — 20 Engine Sample", fontsize=14, fontweight="bold")
    ax.set_xlabel("Flight Cycle")
    ax.set_ylabel("Remaining Useful Life (cycles)")
    ax.axhline(30, color="red", linestyle="--", linewidth=2,
               label="Critical threshold (RUL = 30)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/engine_lifecycle_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[EDA] Saved → engine_lifecycle_curves.png")


def _print_statistical_summary(df):
    print("\n[EDA] Statistical Summary (Key Sensors + RUL):")
    key_cols = ["sensor_2", "sensor_3", "sensor_11",
                "sensor_12", "sensor_15", "RUL"]
    summary = df[key_cols].describe().round(3)
    print(summary.to_string())
    print()
