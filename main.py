"""
main.py
Aircraft Engine Predictive Maintenance — RUL Prediction
Full end-to-end pipeline runner.

Run:
    python main.py
"""

import os
import sys

# Allow imports from src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from data_generator import generate_engine_data
from eda            import run_eda
from preprocessing  import (
    assess_data_quality,
    clean_data,
    engineer_features,
    prepare_train_test,
)
from models import (
    train_all_models,
    evaluate_models,
    plot_model_comparison,
    plot_prediction_vs_actual,
    plot_feature_importance,
    cluster_engine_health,
)


def main():
    print("=" * 60)
    print("  AIRCRAFT ENGINE PREDICTIVE MAINTENANCE — RUL PREDICTION")
    print("  Author : Akanksha Ramchandra Kesarkar")
    print("  Domain : Aviation Digital Technology / Data Science")
    print("=" * 60)

    # ── Step 1: Generate / load data ──────────────────────────
    DATA_PATH = "data/engine_sensor_data.csv"
    if not os.path.exists(DATA_PATH):
        df_raw = generate_engine_data(save_path=DATA_PATH)
    else:
        import pandas as pd
        df_raw = pd.read_csv(DATA_PATH)
        print(f"[Main] Loaded existing dataset → {DATA_PATH} | Shape: {df_raw.shape}")

    # ── Step 2: EDA ────────────────────────────────────────────
    run_eda(df_raw)

    # ── Step 3: Data quality assessment & cleansing ───────────
    assess_data_quality(df_raw)
    df_clean = clean_data(df_raw.copy())

    # ── Step 4: Feature engineering ───────────────────────────
    df_feat = engineer_features(df_clean)

    # ── Step 5: Train/test split ──────────────────────────────
    X_train, X_test, y_train, y_test, feature_cols, scaler = \
        prepare_train_test(df_feat, test_ratio=0.2)

    # ── Step 6: Train models ──────────────────────────────────
    fitted_models = train_all_models(X_train, y_train)

    # ── Step 7: Evaluate ──────────────────────────────────────
    results_df = evaluate_models(fitted_models, X_test, y_test)

    # ── Step 8: Visualisations ────────────────────────────────
    plot_model_comparison(results_df)
    plot_prediction_vs_actual(results_df, y_test)
    plot_feature_importance(fitted_models, feature_cols)

    # ── Step 9: Clustering ────────────────────────────────────
    cluster_engine_health(df_feat, feature_cols, X_train, n_clusters=3)

    # ── Done ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print(f"  Best model : "
          f"{results_df.loc[results_df['R2'].idxmax(), 'Model']}")
    print(f"  Best R²    : "
          f"{results_df['R2'].max():.4f}")
    print(f"  Best RMSE  : "
          f"{results_df.loc[results_df['R2'].idxmax(), 'RMSE']:.3f} cycles")
    print(f"  All charts saved to → outputs/")
    print("=" * 60)


if __name__ == "__main__":
    main()
