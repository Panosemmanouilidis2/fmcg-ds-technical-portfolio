# FMCG DS Technical Portfolio

**Panos Emmanouilidis** · Data Scientist & Analytics Consultant  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=flat&logo=linkedin)](https://www.linkedin.com/in/panosemmanouilidis)

> **Access note:** This is a private repository shared selectively with hiring managers and technical reviewers. If you have been given access, feel free to explore all notebooks, deployment scripts, and application code. To request access, connect via LinkedIn.

---

## Project: Promotional Analytics — Sell-Out Forecasting & Trade ROI Optimisation

Full technical implementation of an end-to-end ML pipeline for promotional sell-out forecasting and trade ROI optimisation across Western Europe and Southeast Asia retail markets.

**Experience and data are real. Select details have been anonymised for confidentiality.**

---

## Repository Structure

```
fmcg-ds-technical-portfolio/
│
└── promotional-analytics/
    ├── notebooks/
    │   ├── 01_eda_planning_accuracy.ipynb      # EDA, planning bias analysis, market comparison
    │   ├── 02_feature_engineering.ipynb        # Feature derivation, encoding, leakage controls
    │   ├── 03_model_training.ipynb             # XGBoost training, 6 model families, hyperparameter tuning
    │   └── 04_model_evaluation_shap.ipynb      # SHAP explainability, ROI derivation, business metrics
    ├── deployment/
    │   ├── app.py                              # Streamlit planner-facing forecast interface
    │   ├── deploy.py                           # Vertex AI model registry + endpoint deployment
    │   ├── deploy_monitoring.py                # Feature drift monitoring + BigQuery logging
    │   ├── predictor.py                        # Custom Vertex AI predictor class
    │   ├── Dockerfile                          # Cloud Run container build
    │   └── requirements.txt                    # Pinned dependencies
    └── results/
        ├── shap_beeswarm.png                   # SHAP beeswarm — feature impact distribution
        ├── shap_feature_importance.png         # SHAP feature importance ranking
        ├── roi_analysis.png                    # ROI distribution by promotion mechanic
        └── xgboost_tuned_evaluation.png        # Model evaluation — actual vs predicted
```

---

## Technical Summary

### Data & Feature Engineering
- 640K+ promotional event records cleaned and validated across two structurally different markets
- 15+ derived features engineered from planned financial structure: incremental volume, uplift rate, cost per unit, ROI proxy, promo duration
- Strict data leakage controls — all features use only pre-execution planned data, making the model operationally deployable before any promotion runs
- Resolved data quality issues: unit mismatches, decimal date formats, zero-plan rows, negative actuals, systematic data entry errors

### Modelling
- 6 model families evaluated per market: Linear Regression, Ridge, Lasso, Random Forest, Gradient Boosting, XGBoost
- Two-stage hyperparameter tuning using efficient sampling approach
- **Western Europe champion: XGBoost (tuned) — R² 0.81 | RMSE 1.46 | 5x improvement over linear baseline**
- **Southeast Asia champion: XGBoost (conservative tuned) — R² 0.70 | RMSE 1.49**
- SHAP analysis revealed 70% of Western Europe sell-out variance is driven by seasonality, not promotion mechanics

### Deployment Architecture
- Models serialised as XGBoost Booster format (`.bst`) — required by Vertex AI pre-built container
- Uploaded to GCS artefact URI, registered in Vertex AI Model Registry with versioning labels
- Deployed to live prediction endpoints (`n1-standard-2`, async `sync=False` pattern)
- Streamlit planner application containerised with Docker, deployed on Cloud Run
- Feature drift monitoring configured with hourly schedule, BigQuery prediction logging, email alerting

### Key Design Decisions
- `sync=False` on all deployments — prevents 15-minute blocking per market
- `endpoint.list_models()` guard check before any model reference — avoids stale ID errors
- `objective_configs` passed as dict `{deployed_model_id: ObjectiveConfig}` — list raises `AttributeError`
- All credentials via Application Default Credentials (ADC) — no hardcoded paths

---

## Model Performance

| Metric | Western Europe | Southeast Asia |
|--------|---------------|----------------|
| Model | XGBoost (tuned) | XGBoost (conservative) |
| R² | 0.81 | 0.70 |
| RMSE | 1.46 | 1.49 |
| Improvement over baseline | 5x | 5x |
| Planning bias identified | -35% over-forecast | +103% under-forecast |
| Promos missing plan by >50% | 60% | 79% |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.10 |
| ML Modelling | XGBoost, Scikit-learn |
| Explainability | SHAP |
| Data Engineering | Pandas, NumPy |
| Cloud Platform | Google Cloud Platform (GCP) |
| Model Deployment | Vertex AI Model Registry, REST endpoints |
| Containerisation | Docker |
| Application | Streamlit on Cloud Run |
| Monitoring | Vertex AI Model Monitoring, BigQuery |

---

## Data & Confidentiality Notice

This project was developed against live commercial sell-out and sell-in data across two markets for a global FMCG manufacturer. Client identity, retail customer names, and raw datasets are not disclosed in line with professional confidentiality obligations. All data presented in this repository is synthetically generated to preserve the statistical properties and structural characteristics of the original — planning accuracy biases, ROI distributions, and mechanic performance rankings reflect real analytical findings. The methodology, deployment architecture, and business outcomes are genuine.

---

## About

10+ years of FMCG and retail analytics experience spanning both manufacturer and retailer perspectives. Specialising in demand forecasting, promotional analytics, on-shelf availability, and end-to-end ML deployment.

📧 [Get in touch via LinkedIn](https://www.linkedin.com/in/panosemmanouilidis)
