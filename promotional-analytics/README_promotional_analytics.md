# Promotional Analytics — Sell-Out Forecasting & Trade ROI Optimisation

**Markets:** Western Europe · Southeast Asia  
**Models:** XGBoost · R² 0.81 (Western Europe) · R² 0.70 (Southeast Asia)  
**Deployment:** GCP Vertex AI · Cloud Run · Streamlit  
**Data:** 640K+ promotional events (synthetic sample provided)

▶ **[Live Streamlit Forecaster App](https://unilever-promo-forecaster-128825737789.europe-west2.run.app)**

---

## 📓 Run the Notebooks — No Setup Needed

Click any badge to open directly in Google Colab and run in your browser:

| Notebook | Description | Open |
|---|---|---|
| 01 — EDA & Forecast Accuracy | Data exploration, planning bias, market comparison | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Panosemmanouilidis2/fmcg-ds-technical-portfolio/blob/main/promotional-analytics/notebooks/01_eda_planning_accuracy.ipynb) |
| 02 — Feature Engineering & Training | 6 model families, XGBoost tuning, overfitting check | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Panosemmanouilidis2/fmcg-ds-technical-portfolio/blob/main/promotional-analytics/notebooks/02_feature_engineering.ipynb) |
| 03 — Vertex AI Deployment | GCS, model registry, endpoint deployment *(GCP account required)* | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Panosemmanouilidis2/fmcg-ds-technical-portfolio/blob/main/promotional-analytics/notebooks/03_model_training.ipynb) |
| 04 — Evaluation & SHAP | Metrics, residuals, feature importance, ROI classification | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Panosemmanouilidis2/fmcg-ds-technical-portfolio/blob/main/promotional-analytics/notebooks/04_model_evaluation_shap.ipynb) |

> ⚠️ Notebook 03 requires a GCP account with Vertex AI enabled. Notebooks 01, 02 and 04 run entirely in Colab with no cloud setup needed.

---

## Problem Statement

A global FMCG manufacturer running thousands of trade promotions annually across Western Europe and Southeast Asia had no reliable, data-driven method to forecast promotional sell-out volume or evaluate return on investment before committing trade spend.

**Quantified consequences:**

| Issue | Western Europe | Southeast Asia |
|-------|---------------|----------------|
| Planning bias | -35% over-forecast | +103% under-forecast |
| Promos missing plan by >50% | 60% | 79% |
| Median planned ROI | -0.042 (value destructive) | +0.597 (value generative) |
| Worst mechanic ROI | -0.11 (Shopper Activation) | -10.00 (Discounted Retail Price) |
| Best mechanic ROI | +0.09 (Loyalty Mechanic) | +10.12 (Multi-Buy Free Gift) |

---

## What Was Built

### 1. Data Pipeline
- Cleaned and validated 640K+ promotional event records across two structurally different markets
- Resolved unit mismatches, decimal date formats (YYYY.WW stored as floats), zero-plan rows, negative actuals, and systematic data entry errors
- Engineered 15+ derived features from planned financial structure

### 2. Feature Engineering
Key features engineered (all pre-execution — no data leakage):

| Feature | Derivation |
|---------|-----------|
| `PlannedIncrementalVolume` | Planned promo volume − baseline volume |
| `PlannedVolumeUplift` | Planned promo volume ÷ baseline volume |
| `PlannedCostPerUnit` | Total trade spend ÷ planned volume |
| `PlannedROI` | (Gross profit − spend) ÷ spend |
| `PromoDurationWeeks` | End week − start week |
| `IsDefensivePromo` | 1 if incremental volume < 0 |
| `IsPipeFill` | 1 if promotion type = Stock Loading |

### 3. Modelling
6 model families evaluated per market:

| Model | Western Europe R² | Southeast Asia R² |
|-------|------------------|------------------|
| Linear Regression (baseline) | 0.15 | 0.27 |
| Ridge | 0.18 | 0.29 |
| Lasso | 0.17 | 0.28 |
| Random Forest | 0.71 | 0.58 |
| LightGBM | 0.74 | 0.62 |
| **XGBoost (tuned)** | **0.81** | **0.70** |

**SHAP insight:** 70% of Western Europe sell-out variance is driven by seasonality (timing), not promotion mechanics or spend level.

### 4. Deployment Architecture

```
Raw sell-out data
       │
       ▼
Data cleaning & feature engineering (15+ features, leakage-controlled)
       │
       ├──────────────────────────────────────┐
       ▼                                      ▼
Western Europe XGBoost              Southeast Asia XGBoost
    R² = 0.81                           R² = 0.70
       │                                      │
       └──────────────────┬───────────────────┘
                          ▼
              Vertex AI Model Registry
              Live REST prediction endpoints
                          │
                          ▼
              Cloud Run · Streamlit
              Planner-facing forecast interface
```

**Key deployment decisions:**
- `sync=False` on all deployments — prevents 15-minute blocking per market
- `endpoint.list_models()` guard check — always run before referencing deployed model IDs
- `objective_configs` as dict `{deployed_model_id: ObjectiveConfig}` — list raises `AttributeError`
- Application Default Credentials (ADC) — no hardcoded credential paths
- Model serialised as Booster JSON — required by Vertex AI pre-built XGBoost container

### 5. Planner Application
Streamlit interface on Cloud Run allows trade planners to:
- Input promotion parameters in plain English (no technical knowledge required)
- Compare up to 4 campaigns side by side
- Receive predicted sell-out volume, uplift %, and ROI estimate before committing spend
- Download results as CSV

---

## Repository Structure

```
promotional-analytics/
├── notebooks/
│   ├── sample_synthetic.csv                # 5,000 row synthetic sample (39 features)
│   ├── 01_eda_planning_accuracy.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_model_evaluation_shap.ipynb
├── deployment/
│   ├── app.py                              # Streamlit planner interface
│   ├── deploy.py                           # Vertex AI deployment script
│   ├── deploy_monitoring.py                # Drift monitoring + BigQuery logging
│   ├── predictor.py                        # Custom Vertex AI predictor class
│   ├── Dockerfile                          # Cloud Run container
│   └── requirements.txt                    # Pinned dependencies
└── results/
    ├── shap_beeswarm.png
    ├── shap_feature_importance.png
    ├── roi_analysis.png
    └── xgboost_tuned_evaluation.png
```

---

## Data & Confidentiality Notice

This project was developed against live commercial sell-out and sell-in data across two markets for a global FMCG manufacturer. Client identity, retail customer names, and raw datasets are not disclosed in line with professional confidentiality obligations. The file `notebooks/sample_synthetic.csv` is synthetically generated to preserve the statistical properties of the original — planning accuracy biases, ROI distributions, and mechanic performance rankings reflect real analytical findings. The methodology, deployment architecture, and business outcomes are genuine.

---

## Author

**Panos Emmanouilidis** · Data Scientist & Analytics Consultant  
[LinkedIn](https://www.linkedin.com/in/panosemmanouilidis)
