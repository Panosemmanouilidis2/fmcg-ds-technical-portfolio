# Promotional Analytics — Sell-Out Forecasting & Trade ROI Optimisation

**Markets:** Market A (Europe) · Market B (Asia)  
**Models:** XGBoost · R² 0.81 (Market A) · R² 0.70 (Market B)  
**Deployment:** GCP Vertex AI · Cloud Run · Streamlit  
**Data:** 700K+ promotional events (synthetic sample provided)

▶ **[Live Streamlit Forecaster App](https://unilever-promo-forecaster-128825737789.europe-west2.run.app)**

---

## 📓 Run the Notebooks — No Setup Needed

| Notebook | Description | Open |
|---|---|---|
| 01 — EDA & Forecast Accuracy | Data exploration, planning bias, market comparison | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Panosemmanouilidis2/fmcg-ds-technical-portfolio/blob/main/promotional-analytics/notebooks/01_eda_planning_accuracy.ipynb) |
| 02 — Feature Engineering & Training | 6 model families, XGBoost tuning, overfitting check | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Panosemmanouilidis2/fmcg-ds-technical-portfolio/blob/main/promotional-analytics/notebooks/02_feature_engineering.ipynb) |
| 03 — Vertex AI Deployment | GCS, model registry, endpoint deployment *(GCP account required)* | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Panosemmanouilidis2/fmcg-ds-technical-portfolio/blob/main/promotional-analytics/notebooks/03_model_training.ipynb) |
| 04 — Evaluation & SHAP | Metrics, residuals, feature importance, ROI classification | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Panosemmanouilidis2/fmcg-ds-technical-portfolio/blob/main/promotional-analytics/notebooks/04_model_evaluation_shap.ipynb) |

> ⚠️ Notebook 03 requires a GCP account. Notebooks 01, 02 and 04 run entirely in Colab with no cloud setup needed.

---

## Problem Statement

A global FMCG manufacturer running thousands of trade promotions annually across two markets had no reliable, data-driven method to forecast promotional sell-out volume or evaluate return on investment before committing trade spend.

**Quantified consequences:**

| Issue | Market A | Market B |
|-------|----------|----------|
| Planning bias | -35% over-forecast | +103% under-forecast |
| Promos missing plan by >50% | 60% | 79% |
| Median planned ROI | -0.042 (value destructive) | +0.597 (value generative) |
| Worst mechanic ROI | -0.11 | -10.00 |
| Best mechanic ROI | +0.09 | +10.12 |

---

## What Was Built

- **XGBoost sell-out forecasting model** — R² 0.81 (Market A), R² 0.70 (Market B), 5x improvement over linear baseline
- **Pre-execution ROI framework** — volume forecast and ROI estimate before any promotion runs
- **Live GCP Vertex AI deployment** — real-time forecasts via REST API endpoints
- **Streamlit planner application on Cloud Run** — business-facing interface requiring no technical skills
- **SHAP explainability layer** — identifies why a promotion is predicted to succeed or fail
- **Data quality resolution** — cleaned and validated 700K+ promotional records across two markets

---

## Model Comparison

| Model | Market A R² | Market B R² |
|-------|------------|------------|
| Linear Regression | 0.15 | 0.27 |
| Ridge | 0.18 | 0.29 |
| Lasso | 0.17 | 0.28 |
| Random Forest | 0.71 | 0.58 |
| LightGBM | 0.74 | 0.62 |
| **XGBoost (tuned)** | **0.81** | **0.70** |

**SHAP insight:** 70% of Market A sell-out variance is driven by seasonality (timing), not promotion mechanics or spend level.

---

## Deployment Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Raw promotional data                     │
│           700K+ records · sell-out & sell-in · two markets  │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│              Data cleaning & feature engineering             │
│                  15+ leakage-controlled features             │
│         GCP Vertex AI Workbench · Region: europe-west2      │
└──────────────┬──────────────────────────┬───────────────────┘
               │                          │
               ▼                          ▼
┌──────────────────────────┐  ┌──────────────────────────────┐
│   Market A · XGBoost     │  │   Market B · XGBoost         │
│   R² = 0.81 · 97 feats   │  │   R² = 0.70 · 78 feats       │
│   Tuned · log1p target   │  │   Conservative · log1p target │
│   Region: europe-west2   │  │   Region: europe-west2       │
└──────────────┬───────────┘  └─────────────┬────────────────┘
               │                            │
               └──────────────┬─────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Cloud Storage (GCS)                       │
│    gs://[project]-ml-data · model artefacts (.json)          │
│        Feature medians & scaling · Region: europe-west2     │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│               Vertex AI Model Registry                       │
│     Versioned model artefacts · XGBoost 1.7.6 container     │
│                    Region: europe-west2                      │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│              Vertex AI Prediction Endpoint                   │
│    n1-standard-2 · sync=False · 100% traffic · REST API     │
│                    Region: europe-west2                      │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│              Cloud Run · Streamlit planner app               │
│      Docker container · planner-facing forecast interface   │
│                    Region: europe-west2                      │
└─────────────────────────────────────────────────────────────┘
```

---

## Planner Application

Streamlit interface on Cloud Run allows trade planners to:

- Input promotion parameters in plain English — no technical knowledge required
- Compare campaigns across different retailers, mechanics and months
- Identify the highest-ROI promotion from a mixed portfolio before committing spend
- Receive predicted sell-out volume, uplift %, and ROI estimate per campaign
- Optimise trade investment allocation across retailer and mechanic combinations
- Download full results as CSV for further analysis

> **Demo note:** The live app uses a synthetic dataset and supports comparison of
> up to 4 campaigns simultaneously. In the original production deployment, predicted
> volumes were in the thousands to hundreds of thousands of units per promotion.
> The synthetic demo data has a compressed scale — the methodology, model
> architecture, and deployment are identical to the production system (R² 0.81, Market A).

---

## Data & Confidentiality Notice

Developed against live commercial sell-out and sell-in data for a global FMCG manufacturer across two markets. Client identity, retail customer names, and raw datasets are not disclosed. The file `notebooks/sample_synthetic.csv` is synthetically generated to preserve the statistical properties of the original. Planning accuracy biases, ROI distributions, and mechanic performance rankings reflect real analytical findings. Methodology, deployment architecture, and business outcomes are genuine.

---

## Author

**Panos Emmanouilidis** · Data Scientist & Analytics Consultant  
[LinkedIn](https://www.linkedin.com/in/panosemmanouilidis)
