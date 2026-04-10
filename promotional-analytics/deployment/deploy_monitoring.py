"""
FMCG Promotional Analytics — Vertex AI Model Monitoring Setup

Configures feature drift monitoring for both market endpoints.
Run AFTER deploy.py and once both endpoints show ACTIVE in GCP Console.

Monitoring behaviour:
- Checks for feature distribution drift every hour
- Logs all predictions to BigQuery for audit and retraining
- Sends email alerts when drift thresholds are exceeded

Key fix applied during development:
- objective_configs must be a dict {deployed_model_id: ObjectiveConfig},
  NOT a list — passing a list causes 'list has no attribute keys' error.

Author: Panos Emmanouilidis
"""

import os, pickle, json
import google.auth
import google.auth.transport.requests
from google.cloud import aiplatform
from google.cloud.aiplatform import model_monitoring

# ── Config ────────────────────────────────────────────────────────────────────
# Replace these values with your own GCP project details before running
ALERT_EMAIL = "YOUR_ALERT_EMAIL"
PROJECT_ID  = "YOUR_GCP_PROJECT_ID"
REGION      = "YOUR_REGION"                              # e.g. europe-west2
BQ_DATASET  = "promo_monitoring"

print("=" * 60)
print("  FMCG Promotional Analytics — Cloud Monitoring Setup")
print("=" * 60)

# ── Auth ──────────────────────────────────────────────────────────────────────
# Uses Application Default Credentials — no hardcoded credentials path
credentials, _ = google.auth.default(
    scopes=["https://www.googleapis.com/auth/cloud-platform"]
)
credentials.refresh(google.auth.transport.requests.Request())
aiplatform.init(project=PROJECT_ID, location=REGION, credentials=credentials)
print("\n✅ Authenticated")

# ── Load session info from deploy.py output ───────────────────────────────────
session       = json.load(open("models/vertex_session.json"))
we_endpoint   = aiplatform.Endpoint(session["western_europe_endpoint"])
sea_endpoint  = aiplatform.Endpoint(session["southeast_asia_endpoint"])
print("✅ Endpoints loaded")

# ── Get deployed model IDs ────────────────────────────────────────────────────
# objective_configs requires deployed model IDs as dict keys.
# Use endpoint.list_models() to retrieve the correct ID after deployment.
# NOTE: endpoint.list_models() is the guard check — always run this before
# referencing deployed model IDs to avoid stale or incorrect IDs.
we_deployed_models   = we_endpoint.list_models()
sea_deployed_models  = sea_endpoint.list_models()

we_deployed_model_id  = we_deployed_models[0].id
sea_deployed_model_id = sea_deployed_models[0].id

print(f"✅ Western Europe deployed model ID : {we_deployed_model_id}")
print(f"✅ Southeast Asia deployed model ID : {sea_deployed_model_id}")

# ── Shared alert config ───────────────────────────────────────────────────────
alert_config = model_monitoring.EmailAlertConfig(
    user_emails    = [ALERT_EMAIL],
    enable_logging = True,
)

# ── Western Europe Monitoring ─────────────────────────────────────────────────
print("\n[1/2] Setting up Western Europe monitoring...")
feat_cols_we = pickle.load(open("models/feature_cols_western_europe.pkl", "rb"))

drift_config_we = model_monitoring.DriftDetectionConfig(
    # Monitor top 20 features for distribution drift.
    # Threshold of 0.3 triggers alert if Jensen-Shannon divergence exceeds this value.
    drift_thresholds = {col: 0.3 for col in feat_cols_we[:20]},
)
objective_we = model_monitoring.ObjectiveConfig(
    drift_detection_config = drift_config_we,
)

monitoring_we = aiplatform.ModelDeploymentMonitoringJob.create(
    display_name              = "western-europe-promo-monitoring",
    project                   = PROJECT_ID,
    location                  = REGION,
    endpoint                  = we_endpoint.resource_name,
    logging_sampling_strategy = model_monitoring.RandomSampleConfig(sample_rate=1.0),
    schedule_config           = model_monitoring.ScheduleConfig(monitor_interval=1),   # hourly
    objective_configs         = {we_deployed_model_id: objective_we},                  # dict, not list
    alert_config              = alert_config,
    predict_instance_schema_uri  = "",
    analysis_instance_schema_uri = "",
    bigquery_tables_log_ttl      = 3650,
)
print(f"  ✅ Western Europe monitoring job: {monitoring_we.resource_name}")
print(f"     Alerts → {ALERT_EMAIL}")
print(f"     Logs   → BigQuery: {PROJECT_ID}.{BQ_DATASET}")

# ── Southeast Asia Monitoring ─────────────────────────────────────────────────
print("\n[2/2] Setting up Southeast Asia monitoring...")
feat_cols_sea = pickle.load(open("models/feature_cols_southeast_asia.pkl", "rb"))

drift_config_sea = model_monitoring.DriftDetectionConfig(
    drift_thresholds = {col: 0.3 for col in feat_cols_sea[:20]},
)
objective_sea = model_monitoring.ObjectiveConfig(
    drift_detection_config = drift_config_sea,
)

monitoring_sea = aiplatform.ModelDeploymentMonitoringJob.create(
    display_name              = "southeast-asia-promo-monitoring",
    project                   = PROJECT_ID,
    location                  = REGION,
    endpoint                  = sea_endpoint.resource_name,
    logging_sampling_strategy = model_monitoring.RandomSampleConfig(sample_rate=1.0),
    schedule_config           = model_monitoring.ScheduleConfig(monitor_interval=1),   # hourly
    objective_configs         = {sea_deployed_model_id: objective_sea},                # dict, not list
    alert_config              = alert_config,
    predict_instance_schema_uri  = "",
    analysis_instance_schema_uri = "",
    bigquery_tables_log_ttl      = 3650,
)
print(f"  ✅ Southeast Asia monitoring job: {monitoring_sea.resource_name}")
print(f"     Alerts → {ALERT_EMAIL}")
print(f"     Logs   → BigQuery: {PROJECT_ID}.{BQ_DATASET}")

# ── Save updated session ──────────────────────────────────────────────────────
session["western_europe_monitoring"] = monitoring_we.resource_name
session["southeast_asia_monitoring"] = monitoring_sea.resource_name
with open("models/vertex_session.json", "w") as f:
    json.dump(session, f, indent=2)

print("\n" + "=" * 60)
print("  MONITORING ENABLED")
print("=" * 60)
print(f"\n  ✅ Both markets monitored hourly")
print(f"  ✅ Drift alerts → {ALERT_EMAIL}")
print(f"  ✅ All predictions logged to BigQuery")
print(f"\n  View monitoring dashboard:")
print(f"  https://console.cloud.google.com/vertex-ai/models?project={PROJECT_ID}")
print("=" * 60)
