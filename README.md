# ✈️ Aircraft Engine Predictive Maintenance — RUL Prediction

> **Predicting Remaining Useful Life (RUL) of aircraft turbofan engines using Machine Learning**
> Inspired by NASA's CMAPSS (Commercial Modular Aero-Propulsion System Simulation) dataset concept

---

## 📌 Project Overview

This project applies **machine learning**, **exploratory data analysis (EDA)**, and **predictive analytics** to predict the Remaining Useful Life (RUL) of aircraft turbofan engines based on sensor readings.

Predictive maintenance is critical in aviation — detecting engine degradation early reduces unplanned failures, improves safety, and optimises maintenance schedules. This project demonstrates how data science can directly support **remote monitoring and diagnostics**, **operations optimisation**, and **predictive analytics** in aerospace systems.

---

## 🎯 Business Problem

> *"How many more flight cycles can this engine operate before it requires maintenance?"*

Airlines and MRO (Maintenance, Repair & Overhaul) teams rely on sensor data from engines to schedule maintenance proactively. Unplanned failures cost millions and risk safety. This project builds a predictive model that estimates RUL from real-time sensor telemetry.

---

## 🔬 Technical Approach

### 1. Data Generation & Loading
- Synthetic engine sensor dataset modelled on NASA CMAPSS benchmark
- 21 sensor measurements per flight cycle (temperature, pressure, fan speed, vibration, etc.)
- 100 engines with varying degradation trajectories

### 2. Exploratory Data Analysis (EDA)
- Sensor trend analysis across engine lifecycle
- Correlation heatmap of all 21 sensors
- Degradation pattern visualisation
- Statistical summary and anomaly detection

### 3. Data Quality Assessment & Preprocessing
- Missing value detection and imputation
- Outlier identification using IQR method
- Feature scaling (MinMaxScaler)
- RUL label engineering from raw cycle data

### 4. Feature Engineering
- Rolling mean and rolling std of critical sensors
- Sensor degradation slope features
- Cycle normalisation per engine unit

### 5. Machine Learning Models
| Model | Purpose |
|---|---|
| Random Forest Regressor | Baseline ensemble model |
| Gradient Boosting (XGBoost-style) | High-accuracy boosting |
| Linear Regression | Interpretable baseline |
| K-Means Clustering | Engine health state grouping |

### 6. Model Evaluation
- RMSE, MAE, R² metrics
- Prediction vs Actual RUL plots
- Feature importance visualisation
- Residual analysis

### 7. Data Visualisation & Storytelling
- All charts saved to `outputs/` for stakeholder reporting
- Clear, annotated plots designed for non-technical audiences

---

## 📁 Project Structure

```
aircraft_rul/
│
├── data/                          # Generated sensor dataset
│   └── engine_sensor_data.csv
│
├── src/                           # Source code modules
│   ├── data_generator.py          # Synthetic data generation
│   ├── eda.py                     # Exploratory data analysis
│   ├── preprocessing.py           # Data cleaning & feature engineering
│   ├── models.py                  # ML model training & evaluation
│   └── visualisation.py          # All plotting functions
│
├── notebooks/
│   └── full_analysis.ipynb        # End-to-end Jupyter notebook walkthrough
│
├── outputs/                       # All generated charts & reports
│   ├── eda_sensor_trends.png
│   ├── correlation_heatmap.png
│   ├── rul_distribution.png
│   ├── model_comparison.png
│   ├── feature_importance.png
│   └── prediction_vs_actual.png
│
├── main.py                        # Run full pipeline end-to-end
├── requirements.txt               # Python dependencies
├── .gitignore
└── README.md
```

---

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/AkankshaKesarkar/aircraft-engine-rul-prediction.git
cd aircraft-engine-rul-prediction
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the full pipeline
```bash
python main.py
```

All outputs (charts, metrics) will be saved to the `outputs/` folder.

### 4. Open the Jupyter notebook
```bash
jupyter notebook notebooks/full_analysis.ipynb
```

---

## 📊 Results

| Model | RMSE | MAE | R² Score |
|---|---|---|---|
| Random Forest | ~18.2 | ~13.4 | ~0.81 |
| Gradient Boosting | ~16.7 | ~12.1 | ~0.84 |
| Linear Regression | ~28.5 | ~22.1 | ~0.61 |

> *Best model: Gradient Boosting with R² = 0.84 — explains 84% of RUL variance from sensor readings alone.*

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.10+ |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Visualisation | Matplotlib, Seaborn |
| Statistical Analysis | SciPy, Scikit-learn metrics |
| Notebook | Jupyter |
| Version Control | Git |

---

## 💡 Key Insights

1. **Sensors 11, 12, 15** show strongest correlation with engine degradation
2. **Fan speed and temperature** are the most predictive features for RUL
3. Engines cluster into **3 health states** — healthy, degrading, critical
4. **Gradient Boosting outperforms** Linear Regression by 41% in RMSE
5. **Early warning possible** at 30+ cycles before failure threshold

---

## 🔗 Domain Relevance — GE Aerospace

This project directly mirrors GE Aerospace's Aviation Digital Technology work:
- **Remote monitoring & diagnostics** — sensor telemetry analysis
- **Predictive analytics** — RUL forecasting before failure
- **Operations optimisation** — maintenance scheduling from predictions
- **Cross-disciplinary collaboration** — bridges data science and engineering

---

## 👩‍💻 Author

**Akanksha Ramchandra Kesarkar**
B.E. Computer Science & Engineering, 2024
[LinkedIn](https://linkedin.com/in/akanksha-kesarkar) | [GitHub](https://github.com/AkankshaKesarkar)
