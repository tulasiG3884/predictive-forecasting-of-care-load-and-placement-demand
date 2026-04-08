# 🏠 Predictive Forecasting of Care Load & Placement Demand
### HHS Unaccompanied Alien Children (UAC) Program

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange?logo=xgboost)
![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red?logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green)
![Internship](https://img.shields.io/badge/Unified%20Mentor-Internship%20Project-blueviolet)

---

## 📌 Project Overview

This project builds a **complete machine learning forecasting system** for the U.S. Department of Health & Human Services (HHS) Unaccompanied Alien Children (UAC) Program.

The program houses migrant children who arrive at the U.S. border without a parent or guardian, and arranges sponsor placements. Daily care load can swing between **1,972 and 11,516 children** — a 5.8× range — making capacity planning extremely difficult without reliable short-term forecasts.

**This system answers three operational questions:**
- How many children will be in HHS care tomorrow?
- How many discharges should be planned for tomorrow?
- Is a surge condition approaching?

---

## 🏆 Key Results

| Metric | Value |
|---|---|
| **Best Model** | XGBoost (Recent Window) |
| **Care Load MAE** | 5.48 children/day |
| **Care Load MAPE** | 0.23% (99.77% accuracy) |
| **vs Naïve Baseline** | 9.6% improvement |
| **Discharge MAE** | 0.63 children/day |
| **Models Evaluated** | 9 total |
| **Training Records** | 720 real HHS records (Jan 2023 – Dec 2025) |

---

## 🔑 Central Finding

> **Training window selection is as important as model selection in the presence of structural breaks.**

A **January 2025 structural break** caused care load to drop 66% permanently (6,500 → 2,200 children). Every model trained on the full dataset — including ARIMA, SARIMA, Exponential Smoothing, and full-window XGBoost — failed catastrophically because they were anchored to the wrong regime.

The solution: **recent-window retraining** (June 2024 onwards). This reduced the XGBoost MAE from 40.66 to 5.48 — a 7.4× improvement — by simply using the right data, not a more complex model.

---

## 📊 Complete Model Leaderboard

| Model | MAE ↓ | RMSE ↓ | MAPE ↓ |
|---|---|---|---|
| 🏆 **XGBoost (Recent Window)** | **5.48** | **7.12** | **0.23%** |
| Naïve Persistence | 6.06 | 7.24 | 0.27% |
| Random Forest (Recent) | 6.54 | 8.44 | 0.28% |
| Linear Regression (Recent) | 7.48 | 8.86 | 0.31% |
| Moving Average (w=3) | 9.76 | 11.77 | 0.43% |
| Ridge Regression (Recent) | 17.80 | 22.80 | 0.74% |
| Exponential Smoothing | 86.69 | 97.40 | 3.74% |
| ARIMA(3,1,3) | 144.35 | 161.63 | 6.20% |
| SARIMA | 433.17 | 501.04 | 18.53% |

---

## 📁 Project Structure

```
uac-forecasting/
│
├── data/
│   ├── raw/
│   │   └── HHS_Unaccompanied_Alien_Children_Program.csv
│   └── processed/
│       ├── cleaned_data.csv
│       └── featured_data.csv
│
├── notebooks/
│   ├── 01_EDA.ipynb                  ← Exploratory data analysis, structural break detection
│   ├── 02_Preprocessing.ipynb        ← Date interpolation, outlier capping, cleaning
│   ├── 03_Feature_Engineering.ipynb  ← 30+ lag, rolling, flow, calendar features
│   ├── 04_Baseline_Models.ipynb      ← Naïve persistence, Moving average
│   ├── 05_Statistical_Models.ipynb   ← Exponential Smoothing, ARIMA, SARIMA
│   ├── 06_ML_Models.ipynb            ← Linear, Ridge, Random Forest, XGBoost + HPT
│   └── 07_Model_Evaluation.ipynb     ← Full comparison, early warning system, figures
│
├── models/
│   ├── best_model_recent.joblib           ← XGBoost care load model
│   ├── best_discharge_model_recent.joblib ← XGBoost discharge model
│   ├── scaler_recent.joblib
│   ├── scaler_discharge_recent.joblib
│   ├── feature_columns_recent.json
│   ├── discharge_feature_columns_recent.json
│   ├── thresholds.json                    ← Stress/Critical alert thresholds
│   ├── model_config.json
│   └── complete_results_final.csv
│
├── reports/
│   └── figures/                      ← All charts exported from notebooks
│
├── src/
│   └── app.py                        ← Streamlit dashboard (6 pages)
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 🗂️ Notebook Pipeline

| Notebook | Purpose | Key Output |
|---|---|---|
| `01_EDA.ipynb` | Trends, seasonality, correlation, structural break detection | lag-1 autocorrelation = 0.99, break confirmed Jan 2025 |
| `02_Preprocessing.ipynb` | Fill 355 missing dates via interpolation, IQR capping | `cleaned_data.csv` — 1,075 rows |
| `03_Feature_Engineering.ipynb` | Build 30+ predictive features | `featured_data.csv` |
| `04_Baseline_Models.ipynb` | Set performance floor | Naïve MAE = 6.06 |
| `05_Statistical_Models.ipynb` | Test classical time-series models | All failed — MAE 86–433 |
| `06_ML_Models.ipynb` | Full-window + recent-window ML with HPT | XGBoost (Recent) MAE = 5.48 |
| `07_Model_Evaluation.ipynb` | Full comparison, early warning system | Final leaderboard + alert thresholds |

---

## 🖥️ Streamlit Dashboard

A 6-page interactive dashboard that operationalises both forecasting models.

**Pages:**
1. **📊 Overview** — KPI cards, historical trend, intake vs discharge balance, leaderboard
2. **🔮 Care Load Forecast** — Enter last 14 days + today's pipeline → next-day prediction
3. **🏥 Discharge Forecast** — Same zero-CSV interface for discharge demand
4. **⚠️ Early Warning System** — Alert zones, 90-day history, surge detection KPIs
5. **📈 Model Performance** — Escalation story, feature importance, notebook figures
6. **📋 About & Dataset** — Problem statement, dataset details, tech stack

**Key design decision — Zero CSV Dependency:**
The forecast pages require no historical data file. Users enter only 17 numbers they naturally know from their daily operational report (last 14 days of care load + today's 3 pipeline figures). All 30+ model features are computed automatically. Works for any future date, any year.

### Live App
🚀 **[View Live Dashboard →](https://uac-forecasting.streamlit.app/)**

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/Sugnik27/uac-forecasting.git
cd uac-forecasting
```

### 2. Create and activate virtual environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit app
```bash
streamlit run src/app.py
```

---

## 📦 Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
statsmodels
streamlit
joblib
```

> Full pinned versions in `requirements.txt`

---

## 🔬 Methodology Summary

### The Escalation Strategy (Industry Standard Approach)

```
Phase 1 — Baselines      Naïve Persistence, Moving Average
      ↓ all too simple
Phase 2 — Statistical    Exponential Smoothing, ARIMA, SARIMA
      ↓ all failed due to structural break
Phase 3 — Full ML        Linear, Ridge, Random Forest, XGBoost (full window)
      ↓ same problem — wrong training distribution
Phase 4 — Recent ML ✅   All models retrained on Jun 2024+ data → XGBoost wins
```

### Feature Importance (XGBoost)

| Feature | Importance | Role |
|---|---|---|
| `hhs_care_roll_min_30` | 0.541 | Regime detection — captures post-break floor |
| `hhs_care_lag_2` | 0.159 | Short-term autoregressive signal |
| `hhs_care_lag_1` | 0.150 | Yesterday's value |
| `cbp_transferred` | 0.122 | Pipeline leading indicator |

Top 4 features = **97.2% of total importance**

### Early Warning Thresholds

| Level | Threshold | Basis |
|---|---|---|
| 🟢 Normal | < 8,040 | Below 75th percentile |
| 🟡 Stress | 8,040 – 9,831 | 75th to 90th percentile |
| 🔴 Critical | > 9,831 | Above 90th percentile |

---

## 🗃️ Dataset

| Property | Value |
|---|---|
| Source | HHS UAC Program (public records) |
| Raw records | 720 observations |
| Date range | January 2023 – December 2025 |
| After interpolation | 1,075 rows |
| Target 1 | `hhs_care` — children in HHS care (1,972 – 11,516) |
| Target 2 | `hhs_discharged` — daily discharges |
| lag-1 autocorrelation | 0.99 |
| Structural break | January 2025 (–66% permanent drop) |

---

## 🔗 Project Links

| Resource | Link |
|---|---|
| 📂 GitHub Repository | *This repo* |
| 📝 Research Paper | [Read on Hashnode →](https://your-hashnode-link-here) |
| 🚀 Live Dashboard | [Open Streamlit App →](https://uac-forecasting.streamlit.app/) |
| 📄 Executive Summary | [View on Google Drive →](https://drive.google.com/drive/folders/1di-SvV6YidjTOGIvU8sLPXdahH1qhgIa?usp=sharing) |

---

## 🛠️ Tech Stack

| Component | Tool |
|---|---|
| Language | Python 3.x |
| ML Model | XGBoost |
| Statistical Models | statsmodels (ARIMA, SARIMA, ES) |
| Data Processing | pandas, numpy |
| Visualisation | matplotlib, seaborn |
| Dashboard | Streamlit |
| Model Serialisation | joblib |
| Notebooks | Jupyter |
| Version Control | Git + GitHub |

---

