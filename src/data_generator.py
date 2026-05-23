"""
data_generator.py
Generates synthetic aircraft engine sensor data
modelled on the NASA CMAPSS turbofan engine dataset.
"""

import numpy as np
import pandas as pd
import os


def generate_engine_data(
    n_engines=100,
    max_cycles=300,
    random_seed=42,
    save_path="data/engine_sensor_data.csv"
):
    """
    Generate synthetic sensor readings for turbofan engines.

    Each engine starts healthy and degrades over time.
    21 sensor channels simulate real engine telemetry.

    Parameters
    ----------
    n_engines   : int  — number of engine units
    max_cycles  : int  — maximum flight cycles per engine
    random_seed : int  — reproducibility seed
    save_path   : str  — output CSV path

    Returns
    -------
    pd.DataFrame with columns:
        engine_id, cycle, sensor_1 ... sensor_21, RUL
    """
    np.random.seed(random_seed)
    records = []

    sensor_names = [f"sensor_{i}" for i in range(1, 22)]

    # Sensor baseline means (approximate CMAPSS values)
    baselines = [
        518.67, 641.82, 1589.7, 1400.6, 14.62,
        21.61, 554.36, 2388.1, 9046.2, 1.3,
        47.47, 521.66, 2388.1, 8138.6, 8.4195,
        0.03, 392.0, 2388.0, 100.0, 39.06, 23.42
    ]

    # Which sensors degrade vs remain stable
    degrading_sensors = [2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21]
    stable_sensors    = [1, 5, 6, 10, 16, 18, 19]

    for engine_id in range(1, n_engines + 1):
        # Each engine lives a random number of cycles
        engine_life = np.random.randint(150, max_cycles)
        noise_scale = np.random.uniform(0.8, 1.2)

        for cycle in range(1, engine_life + 1):
            rul = engine_life - cycle          # Remaining Useful Life label
            degradation = cycle / engine_life  # 0 → 1 as engine wears

            row = {
                "engine_id": engine_id,
                "cycle":     cycle,
                "RUL":       rul
            }

            for i, sensor in enumerate(sensor_names, start=1):
                base  = baselines[i - 1]
                noise = np.random.normal(0, base * 0.005 * noise_scale)

                if i in degrading_sensors:
                    # Gradual degradation trend + noise
                    trend = base * 0.08 * degradation
                    row[sensor] = round(base + trend + noise, 4)
                else:
                    # Stable sensors — noise only
                    row[sensor] = round(base + noise, 4)

            records.append(row)

    df = pd.DataFrame(records)

    # Reorder columns
    cols = ["engine_id", "cycle"] + sensor_names + ["RUL"]
    df   = df[cols]

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"[data_generator] Dataset saved → {save_path}")
    print(f"[data_generator] Shape: {df.shape} | Engines: {n_engines} | "
          f"Total cycles: {len(df)}")
    return df


if __name__ == "__main__":
    generate_engine_data()
