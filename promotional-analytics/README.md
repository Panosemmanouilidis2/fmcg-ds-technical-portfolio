# Promotional Analytics — Sell-Out Forecasting & Trade ROI Optimisation

**Markets:** Western Europe · Southeast Asia  
**Models:** XGBoost · R² 0.81 (Western Europe) · R² 0.70 (Southeast Asia)  
**Deployment:** GCP Vertex AI · Cloud Run · Streamlit  
**Data:** 640K+ promotional events (synthetic sample provided)

---

## Problem Statement

A global FMCG manufacturer running thousands of trade promotions annually across Western Europe and Southeast Asia had no reliable, data-driven method to forecast promotional sell-out volume or evaluate return on investment before committing trade spend.

**Quantified consequences:**

| Issue | Western Europe | Southeast Asia |
|-------|---------------|----------------|
| Planning bias | -35% over-forecast | +103% under-forecast |
| Promos missing plan by >50% | 60% | 79% |
| Median planned ROI | -0.042 (value destructive) | +0.597 (value generative) |
| Worst mechanic ROI | -0.11 (Shopper Marketing) | -10.00 (Discount RP) |
| Best mechanic ROI | +0.09 (Loyalty) | +10.12 (Multi-Buy Free) |

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
| `PlannedUpliftRate` | Planned promo volume ÷ baseline volume |
| `PlannedCostPerUnit` | Total TTS spend ÷ planned volume |
| `PlannedROI_proxy` | (iGP − spend) ÷ spend |
| `PromoDurationWeeks` | End week − start week |
| `IsDefensivePromo` | 1 if incremental volume < 0 |
| `IsPipeFill` | 1 if mechanic = Pipe Fill |

### 3. Modelling
6 model families evaluated per market:

| Model | Western Europe R² | Southeast Asia R² |
|-------|------------------|------------------|
| Linear Regression (baseline) | 0.15 | 0.27 |
| Ridge | 0.18 | 0.29 |
| Lasso | 0.17 | 0.28 |
| Random Forest | 0.71 | 0.58 |
| Gradient Boosting | 0.74 | 0.62 |
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
- `sync=False` on all deployments — prevents 15-minute blocking per market, both markets deploy in parallel
- `endpoint.list_models()` guard check — always run before referencing deployed model IDs
- `objective_configs` as dict `{deployed_model_id: ObjectiveConfig}` — list raises `AttributeError`
- Application Default Credentials (ADC) — no hardcoded credential paths
- Model serialised as `.bst` (Booster format) — only format accepted by Vertex AI pre-built XGBoost container

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
│   ├── 01_eda_planning_accuracy.ipynb      # EDA, planning bias, market comparison
│   ├── 02_feature_engineering.ipynb        # Feature derivation, encoding, leakage controls
│   ├── 03_model_training.ipynb             # XGBoost training, tuning, model comparison
│   └── 04_model_evaluation_shap.ipynb      # SHAP explainability, ROI derivation
├── deployment/
│   ├── app.py                              # Streamlit planner interface
│   ├── deploy.py                           # Vertex AI deployment script
│   ├── deploy_monitoring.py                # Drift monitoring + BigQuery logging
│   ├── predictor.py                        # Custom Vertex AI predictor class
│   ├── Dockerfile                          # Cloud Run container
│   └── requirements.txt                    # Pinned dependencies
├── results/
│   ├── shap_beeswarm.png                           # Feature impact distribution
│   ├── shap_feature_importance.png                 # Feature importance ranking
│   ├── roi_analysis.png                            # ROI by promotion mechanic
│   ├── xgboost_tuned_evaluation.png                # Actual vs predicted
│   ├── southeast_asia_time_trend.png               # Actual vs predicted by week — Southeast Asia
│   └── planning_accuracy_by_customer_mechanic.png  # Planning bias by retailer and mechanic — Western Europe
└── data/
    └── sample_synthetic.csv                # 5,000 row synthetic sample (39 features)
```

---

## Running the Notebooks

```bash
pip install -r deployment/requirements.txt
pip install shap matplotlib seaborn jupyter
jupyter notebook notebooks/
```

The notebooks run end to end on `data/sample_synthetic.csv`. No GCP credentials required for the analysis notebooks. Deployment notebooks (`03`, `04`) require a configured GCP project with Vertex AI enabled.

---

## Data & Confidentiality Notice

This project was developed against live commercial sell-out and sell-in data across two markets for a global FMCG manufacturer. Client identity, retail customer names, and raw datasets are not disclosed in line with professional confidentiality obligations. The file `data/sample_synthetic.csv` is synthetically generated to preserve the statistical properties of the original — planning accuracy biases, ROI distributions, and mechanic performance rankings reflect real analytical findings. The methodology, deployment architecture, and business outcomes are genuine.

---

## Author

**Panos Emmanouilidis** · Data Scientist & Analytics Consultant  
[LinkedIn](https://www.linkedin.com/in/panosemmanouilidis)
