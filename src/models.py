"""
models.py
Machine Learning models for RUL prediction:
  - Random Forest Regressor
  - Gradient Boosting Regressor
  - Linear Regression (baseline)
  - K-Means Clustering (engine health states)
"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import os

matplotlib.use("Agg")

from sklearn.ensemble       import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model   import LinearRegression
from sklearn.cluster        import KMeans
from sklearn.metrics        import mean_squared_error, mean_absolute_error, r2_score

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── Regression models ──────────────────────────────────────────

def train_all_models(X_train, y_train):
    """Train all regression models. Returns dict of fitted models."""
    print("\n[Models] Training ML models...")

    models = {
        "Linear Regression":   LinearRegression(),
        "Random Forest":       RandomForestRegressor(
                                   n_estimators=100,
                                   max_depth=12,
                                   random_state=42,
                                   n_jobs=-1
                               ),
        "Gradient Boosting":   GradientBoostingRegressor(
                                   n_estimators=150,
                                   learning_rate=0.08,
                                   max_depth=5,
                                   random_state=42
                               ),
    }

    fitted = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        fitted[name] = model
        print(f"  ✓ {name} trained")

    return fitted


def evaluate_models(fitted_models: dict, X_test, y_test) -> pd.DataFrame:
    """Evaluate all models and return metrics dataframe."""
    print("\n[Models] Evaluation Results")
    print("=" * 55)
    print(f"  {'Model':<25} {'RMSE':>8} {'MAE':>8} {'R²':>8}")
    print("-" * 55)

    results = []
    for name, model in fitted_models.items():
        preds = model.predict(X_test)
        preds = np.clip(preds, 0, None)   # RUL cannot be negative

        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae  = mean_absolute_error(y_test, preds)
        r2   = r2_score(y_test, preds)

        results.append({
            "Model":  name,
            "RMSE":   round(rmse, 3),
            "MAE":    round(mae,  3),
            "R2":     round(r2,   4),
            "Preds":  preds
        })
        print(f"  {name:<25} {rmse:>8.3f} {mae:>8.3f} {r2:>8.4f}")

    print("=" * 55)
    return pd.DataFrame(results)


def plot_model_comparison(results_df: pd.DataFrame):
    """Bar chart comparing RMSE, MAE, R² across models."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.suptitle("Model Performance Comparison — RUL Prediction",
                 fontsize=14, fontweight="bold")

    metrics  = ["RMSE", "MAE", "R2"]
    titles   = ["Root Mean Squared Error (lower = better)",
                "Mean Absolute Error (lower = better)",
                "R² Score (higher = better)"]
    colors   = ["#E53935", "#1E88E5", "#43A047"]

    for ax, metric, title, color in zip(axes, metrics, titles, colors):
        vals = results_df[metric]
        bars = ax.bar(results_df["Model"], vals, color=color, alpha=0.85,
                      edgecolor="white", linewidth=1.2)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel(metric)
        ax.tick_params(axis="x", rotation=15)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(vals) * 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[Models] Saved → model_comparison.png")


def plot_prediction_vs_actual(results_df: pd.DataFrame, y_test):
    """Scatter plot of predicted vs actual RUL for best model."""
    # Pick best model by R²
    best_row  = results_df.loc[results_df["R2"].idxmax()]
    best_name = best_row["Model"]
    preds     = best_row["Preds"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f"Prediction Analysis — {best_name} (Best Model)",
                 fontsize=14, fontweight="bold")

    # Scatter
    ax = axes[0]
    ax.scatter(y_test, preds, alpha=0.3, color="#1E88E5", s=8, label="Predictions")
    max_val = max(y_test.max(), preds.max())
    ax.plot([0, max_val], [0, max_val], "r--", linewidth=2, label="Perfect prediction")
    ax.set_xlabel("Actual RUL (cycles)")
    ax.set_ylabel("Predicted RUL (cycles)")
    ax.set_title("Predicted vs Actual RUL")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Residuals
    ax = axes[1]
    residuals = y_test - preds
    ax.hist(residuals, bins=50, color="#43A047", edgecolor="white", alpha=0.85)
    ax.axvline(0, color="red", linestyle="--", linewidth=2, label="Zero error line")
    ax.set_xlabel("Residual (Actual − Predicted)")
    ax.set_ylabel("Frequency")
    ax.set_title("Residual Distribution")
    ax.legend()

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/prediction_vs_actual.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[Models] Saved → prediction_vs_actual.png")


def plot_feature_importance(fitted_models: dict, feature_cols: list):
    """Feature importance from Random Forest."""
    rf = fitted_models.get("Random Forest")
    if rf is None:
        return

    importances = pd.Series(rf.feature_importances_, index=feature_cols)
    top20       = importances.nlargest(20)

    fig, ax = plt.subplots(figsize=(12, 8))
    top20.sort_values().plot(kind="barh", ax=ax, color="#FB8C00", alpha=0.85,
                             edgecolor="white")
    ax.set_title("Top 20 Feature Importances — Random Forest",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Importance Score")
    ax.grid(True, axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[Models] Saved → feature_importance.png")


# ── Clustering ─────────────────────────────────────────────────

def cluster_engine_health(df: pd.DataFrame, feature_cols: list,
                           X_scaled, n_clusters: int = 3):
    """
    K-Means clustering to group engines into health states:
    Healthy / Degrading / Critical
    """
    print(f"\n[Models] K-Means Clustering (k={n_clusters}) — Engine Health States...")

    kmeans  = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels  = kmeans.fit_predict(X_scaled)

    # Map cluster to health label by mean RUL
    df_plot        = df.copy().reset_index(drop=True)
    df_plot        = df_plot.iloc[:len(labels)].copy()
    df_plot["cluster"] = labels

    cluster_rul = df_plot.groupby("cluster")["RUL"].mean().sort_values(ascending=False)
    health_map  = {
        cluster_rul.index[0]: "Healthy",
        cluster_rul.index[1]: "Degrading",
        cluster_rul.index[2]: "Critical"
    }
    df_plot["health_state"] = df_plot["cluster"].map(health_map)

    print("  Cluster → Health State mapping:")
    for cluster, state in health_map.items():
        mean_rul = cluster_rul[cluster]
        count    = (df_plot["cluster"] == cluster).sum()
        print(f"    Cluster {cluster} → {state:10s} | Mean RUL: {mean_rul:.1f} "
              f"| Records: {count}")

    _plot_health_clusters(df_plot)
    return df_plot, kmeans


def _plot_health_clusters(df_plot: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Engine Health State Clustering — K-Means (k=3)",
                 fontsize=14, fontweight="bold")

    palette = {"Healthy": "#43A047", "Degrading": "#FB8C00", "Critical": "#E53935"}

    # RUL distribution by health state
    ax = axes[0]
    for state, color in palette.items():
        subset = df_plot[df_plot["health_state"] == state]["RUL"]
        ax.hist(subset, bins=30, color=color, alpha=0.7, label=state, edgecolor="white")
    ax.set_title("RUL Distribution by Health State")
    ax.set_xlabel("Remaining Useful Life (cycles)")
    ax.set_ylabel("Frequency")
    ax.legend()

    # Cycle vs RUL coloured by health state
    ax = axes[1]
    for state, color in palette.items():
        subset = df_plot[df_plot["health_state"] == state]
        ax.scatter(subset["cycle"], subset["RUL"],
                   c=color, alpha=0.3, s=5, label=state)
    ax.set_title("Cycle vs RUL — Health State Boundaries")
    ax.set_xlabel("Flight Cycle")
    ax.set_ylabel("Remaining Useful Life")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/health_state_clusters.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[Models] Saved → health_state_clusters.png")
