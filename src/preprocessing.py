"""
preprocessing.py
Data quality assessment, cleansing, preprocessing,
and feature engineering for engine sensor data.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


SENSOR_COLS  = [f"sensor_{i}" for i in range(1, 22)]
STABLE_SENSORS = ["sensor_1", "sensor_5", "sensor_6",
                  "sensor_10", "sensor_16", "sensor_18", "sensor_19"]


def assess_data_quality(df: pd.DataFrame) -> dict:
    """
    Comprehensive data quality assessment report.
    Returns dict with quality metrics.
    """
    print("\n[Preprocessing] Data Quality Assessment")
    print("=" * 50)

    report = {}

    # Missing values
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    report["missing_values"]     = missing[missing > 0].to_dict()
    report["missing_pct"]        = missing_pct[missing_pct > 0].to_dict()
    print(f"  Missing values     : {missing.sum()} total")

    # Duplicate rows
    dupes = df.duplicated().sum()
    report["duplicate_rows"] = dupes
    print(f"  Duplicate rows     : {dupes}")

    # Outliers per sensor (IQR method)
    outlier_counts = {}
    for col in SENSOR_COLS:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr     = q3 - q1
        lo, hi  = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_out   = ((df[col] < lo) | (df[col] > hi)).sum()
        if n_out > 0:
            outlier_counts[col] = int(n_out)
    report["outliers"] = outlier_counts
    total_outliers = sum(outlier_counts.values())
    print(f"  Outliers detected  : {total_outliers} across "
          f"{len(outlier_counts)} sensors (IQR method)")

    # Variance — identify near-zero variance sensors
    low_var = [c for c in SENSOR_COLS if df[c].std() < 0.1]
    report["low_variance_sensors"] = low_var
    print(f"  Low-variance sens. : {len(low_var)} → will be dropped")

    print("=" * 50)
    return report


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply data cleansing steps:
    - Handle missing values
    - Remove duplicates
    - Clip outliers
    - Drop low-variance / stable sensors
    """
    print("\n[Preprocessing] Data Cleansing...")
    original_shape = df.shape

    # Drop duplicates
    df = df.drop_duplicates()

    # Fill any missing values with column median
    for col in SENSOR_COLS:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    # Clip outliers using IQR (cap not remove — preserve time series continuity)
    for col in SENSOR_COLS:
        q1, q3 = df[col].quantile(0.01), df[col].quantile(0.99)
        df[col] = df[col].clip(lower=q1, upper=q3)

    # Drop known stable/non-degrading sensors (low information gain for RUL)
    df = df.drop(columns=STABLE_SENSORS, errors="ignore")

    print(f"  Shape before clean : {original_shape}")
    print(f"  Shape after clean  : {df.shape}")
    print(f"  Dropped sensors    : {STABLE_SENSORS}")
    return df.reset_index(drop=True)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature engineering:
    - Rolling mean & std (window=5) for each sensor
    - Degradation slope per sensor
    - Cycle normalisation per engine
    """
    print("\n[Preprocessing] Feature Engineering...")

    remaining_sensors = [c for c in SENSOR_COLS if c not in STABLE_SENSORS
                         and c in df.columns]
    new_features = []

    for col in remaining_sensors:
        # Rolling statistics (captures local trend)
        df[f"{col}_roll_mean"] = (
            df.groupby("engine_id")[col]
            .transform(lambda x: x.rolling(5, min_periods=1).mean())
        )
        df[f"{col}_roll_std"] = (
            df.groupby("engine_id")[col]
            .transform(lambda x: x.rolling(5, min_periods=1).std().fillna(0))
        )
        new_features.extend([f"{col}_roll_mean", f"{col}_roll_std"])

    # Normalised cycle per engine (0 → 1 across engine lifetime)
    max_cycle = df.groupby("engine_id")["cycle"].transform("max")
    df["cycle_norm"] = df["cycle"] / max_cycle

    print(f"  New features added : {len(new_features)} rolling features + cycle_norm")
    print(f"  Final shape        : {df.shape}")
    return df


def scale_features(df: pd.DataFrame, feature_cols: list):
    """
    Apply MinMaxScaler to feature columns.
    Returns scaled array and fitted scaler.
    """
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(df[feature_cols])
    return X_scaled, scaler


def prepare_train_test(df: pd.DataFrame, test_ratio: float = 0.2):
    """
    Split by engine_id to avoid data leakage.
    Returns X_train, X_test, y_train, y_test, feature_cols.
    """
    print("\n[Preprocessing] Train/Test Split (by engine)...")

    all_engines  = df["engine_id"].unique()
    n_test       = int(len(all_engines) * test_ratio)
    np.random.seed(42)
    test_engines = np.random.choice(all_engines, size=n_test, replace=False)

    train_df = df[~df["engine_id"].isin(test_engines)]
    test_df  = df[df["engine_id"].isin(test_engines)]

    exclude = {"engine_id", "cycle", "RUL"}
    feature_cols = [c for c in df.columns if c not in exclude]

    X_train, scaler = scale_features(train_df, feature_cols)
    X_test, _       = scale_features(test_df,  feature_cols)
    X_test          = scaler.transform(test_df[feature_cols])

    y_train = train_df["RUL"].values
    y_test  = test_df["RUL"].values

    print(f"  Training engines   : {len(all_engines) - n_test}")
    print(f"  Test engines       : {n_test}")
    print(f"  X_train shape      : {X_train.shape}")
    print(f"  X_test shape       : {X_test.shape}")

    return X_train, X_test, y_train, y_test, feature_cols, scaler
