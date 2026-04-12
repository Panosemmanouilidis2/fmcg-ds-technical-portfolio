# FMCG DS Technical Portfolio

**Panos Emmanouilidis** · Data Scientist & Analytics Consultant  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=flat&logo=linkedin)](https://www.linkedin.com/in/panosemmanouilidis)

---

## Project: Promotional Analytics — Sell-Out Forecasting & Trade ROI Optimisation

End-to-end ML pipeline for promotional sell-out forecasting and trade ROI optimisation across two retail markets. Built on 700K+ promotional event records. Models deployed as live endpoints on GCP Vertex AI.

**Experience and data are real. Select details anonymised for confidentiality. Data synthetically reproduced for portfolio purposes.**

▶ **[Live Streamlit Forecaster App](https://unilever-promo-forecaster-128825737789.europe-west2.run.app)**

---

## 📓 Run the Notebooks — No Setup Needed

Click any badge to open directly in Google Colab and run in your browser:

| # | Notebook | Description | Open |
|---|---|---|---|
| 01 | EDA & Forecast Accuracy | Data exploration, planning bias analysis, market comparison | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Panosemmanouilidis2/fmcg-ds-technical-portfolio/blob/main/promotional-analytics/notebooks/01_eda_planning_accuracy.ipynb) |
| 02 | Feature Engineering & Training | Feature encoding, 6 model families, XGBoost tuning, overfitting check | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Panosemmanouilidis2/fmcg-ds-technical-portfolio/blob/main/promotional-analytics/notebooks/02_feature_engineering.ipynb) |
| 03 | Vertex AI Deployment | GCS upload, model registration, endpoint deployment, smoke test | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Panosemmanouilidis2/fmcg-ds-technical-portfolio/blob/main/promotional-analytics/notebooks/03_model_training.ipynb) |
| 04 | Evaluation & SHAP | Model metrics, residuals, SHAP explainability, ROI classification | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Panosemmanouilidis2/fmcg-ds-technical-portfolio/blob/main/promotional-analytics/notebooks/04_model_evaluation_shap.ipynb) |

> ⚠️ Notebook 03 requires a GCP account with Vertex AI enabled. Notebooks 01, 02 and 04 run entirely in Colab with no cloud setup needed.

---

## Model Performance

| Metric | Market A | Market B |
|--------|----------|----------|
| Model | XGBoost (tuned) | XGBoost (conservative) |
| R² | **0.81** | **0.70** |
| RMSE | 1.46 | 1.49 |
| vs linear baseline | 5x improvement | 5x improvement |
| Planning bias found | -35% over-forecast | +103% under-forecast |
| Promos missing plan >50% | 60% | 79% |

---

## Technical Summary

### Data & Feature Engineering
- 700K+ promotional records cleaned and validated across two structurally different markets
- 15+ derived features from planned financial structure: incremental volume, uplift rate, cost per unit, ROI proxy, promo duration
- Strict leakage controls — all features use only pre-execution planned data
- Resolved: unit mismatches, decimal date formats, zero-plan rows, negative actuals, systematic data entry errors

### Modelling
- 6 model families evaluated: Linear Regression, Ridge, Lasso, Random Forest, LightGBM, XGBoost
- Two-stage hyperparameter tuning: random search on sample → retrain on full dataset
- SHAP analysis: 70% of Market A sell-out variance driven by seasonality, not promotion mechanics

### Deployment
- Models serialised as XGBoost Booster JSON, registered in Vertex AI Model Registry
- Live prediction endpoints on `n1-standard-2` via `sync=False` async pattern
- Streamlit planner app containerised with Docker, deployed on Cloud Run
- Feature drift monitoring with hourly schedule, BigQuery prediction logging, email alerting

### Key Design Decisions
- `sync=False` on all deployments — prevents 15-minute blocking per market
- `endpoint.list_models()` guard before any model reference — avoids stale ID errors
- All credentials via Application Default Credentials — no hardcoded paths

---

## Repository Structure

```
fmcg-ds-technical-portfolio/
│
└── promotional-analytics/
    ├── notebooks/
    │   ├── sample_synthetic.csv                # Synthetic dataset — 5,000 rows
    │   ├── 01_eda_planning_accuracy.ipynb
    │   ├── 02_feature_engineering.ipynb
    │   ├── 03_model_training.ipynb
    │   └── 04_model_evaluation_shap.ipynb
    ├── deployment/
    │   ├── app.py                              # Streamlit planner-facing forecast interface
    │   ├── deploy.py                           # Vertex AI model registry + endpoint deployment
    │   ├── deploy_monitoring.py                # Feature drift monitoring + BigQuery logging
    │   ├── predictor.py                        # Custom Vertex AI predictor class
    │   ├── Dockerfile                          # Cloud Run container build
    │   └── requirements.txt                    # Pinned dependencies
    └── results/
        ├── shap_beeswarm.png
        ├── shap_feature_importance.png
        ├── roi_analysis.png
        ├── xgboost_tuned_evaluation.png
        ├── planning_accuracy_by_customer_mechanic.png
        ├── southeast_asia_time_trend.png
        ├── streamlit_homepage.png
        ├── streamlit_retailer_dropdown.png
        ├── streamlit_forecast_results.png
        ├── gcp_vertex_endpoint_active.png
        ├── gcp_endpoint_detail.png
        ├── gcp_logs_api_calls.png
        ├── github_notebooks_folder.png
        ├── github_repo_structure.png
        └── colab_notebook_badge.png
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.10 |
| ML | XGBoost, LightGBM, Scikit-learn |
| Explainability | SHAP |
| Data | Pandas, NumPy |
| Cloud | GCP — Vertex AI, Cloud Run, GCS, BigQuery |
| App | Streamlit |
| Container | Docker |

---

## Data & Confidentiality Notice

Developed against live commercial data for a global FMCG manufacturer across two markets. Client identity, retail customer names, and raw datasets are not disclosed. All data in this repository is synthetically generated to preserve statistical properties of the original. Planning accuracy biases, ROI distributions, and mechanic performance rankings reflect real analytical findings. Methodology, deployment architecture, and business outcomes are genuine.

---

## About

10+ years FMCG and retail analytics experience. Specialising in demand forecasting, promotional analytics, on-shelf availability, and end-to-end ML deployment.

📧 [Connect on LinkedIn](https://www.linkedin.com/in/panosemmanouilidis)
