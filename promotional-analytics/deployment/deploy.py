"""
FMCG Promotional Analytics — Vertex AI Deployment Script

Deploys two market-specific XGBoost promotional sell-out forecasting models
to Google Cloud Vertex AI as live prediction endpoints.

Markets: Western Europe (R²=0.81) and Southeast Asia (R²=0.70)
Infrastructure: Vertex AI Model Registry + n1-standard-2 endpoints
Deployment pattern: async (sync=False) — both markets deploy in parallel

Prerequisites:
- GCP project with Vertex AI API enabled
- GCS bucket containing model artefacts
- Application Default Credentials configured (gcloud auth application-default login)

Author: Panos Emmanouilidis
"""

import os, sys, pickle, shutil, json, time
import numpy as np
import xgboost as xgb

# ── Config ────────────────────────────────────────────────────────────────────
# Replace these values with your own GCP project details before running
PROJECT_ID        = "YOUR_GCP_PROJECT_ID"
REGION            = "YOUR_REGION"                        # e.g. europe-west2
BUCKET_NAME       = "YOUR_BUCKET_NAME"
BUCKET_URI        = f"gs://{BUCKET_NAME}"
XGBOOST_CONTAINER = "europe-docker.pkg.dev/vertex-ai/prediction/xgboost-cpu.1-7:latest"

# Application Default Credentials — set via gcloud CLI, not hardcoded path
# Run: gcloud auth application-default login
# No credentials path needed in production — ADC resolves automatically

print("=" * 60)
print("  FMCG Promotional Analytics — Vertex AI Deployment")
print("=" * 60)

# ── Step 1: Auth ──────────────────────────────────────────────────────────────
print("\n[1/7] Authenticating...")
import google.auth
import google.auth.transport.requests
from google.cloud import aiplatform, storage

credentials, project = google.auth.default(
    scopes=["https://www.googleapis.com/auth/cloud-platform"]
)
credentials.refresh(google.auth.transport.requests.Request())
aiplatform.init(project=PROJECT_ID, location=REGION, credentials=credentials)
print(f"  ✅ Authenticated | XGBoost {xgb.__version__} | Token valid: {credentials.valid}")

# ── Step 2: Prepare artefacts ─────────────────────────────────────────────────
print("\n[2/7] Preparing model artefacts...")
os.makedirs("staging/western_europe", exist_ok=True)
os.makedirs("staging/southeast_asia", exist_ok=True)

# Clean staging dirs before upload.
# GCS must contain EXACTLY ONE model file (model.bst) per prefix.
# Any leftover model.pkl / model.json from previous runs causes FailedPrecondition 400.
for f in os.listdir("staging/western_europe"): os.remove(f"staging/western_europe/{f}")
for f in os.listdir("staging/southeast_asia"): os.remove(f"staging/southeast_asia/{f}")

# Load as Booster — files are raw booster saves, NOT XGBRegressor saves.
# Save as model.bst — the only format the pre-built XGBoost container accepts.
booster_we = xgb.Booster()
booster_we.load_model("models/xgb_western_europe_tuned.json")
booster_we.save_model("staging/western_europe/model.bst")
print("  ✅ Western Europe artefacts ready")

booster_sea = xgb.Booster()
booster_sea.load_model("models/xgb_southeast_asia_tuned.json")
booster_sea.save_model("staging/southeast_asia/model.bst")
print("  ✅ Southeast Asia artefacts ready")

# ── Step 3: Upload to GCS ─────────────────────────────────────────────────────
print("\n[3/7] Uploading to GCS...")
client = storage.Client(project=PROJECT_ID)
bucket = client.bucket(BUCKET_NAME)

for market, local_dir, gcs_prefix in [
    ("Western Europe", "staging/western_europe", "artifacts/western_europe"),
    ("Southeast Asia", "staging/southeast_asia", "artifacts/southeast_asia"),
]:
    # Delete all existing blobs in this prefix first.
    # Vertex AI container requires exactly one model file per artefact URI.
    existing = list(bucket.list_blobs(prefix=gcs_prefix + "/"))
    for blob in existing:
        blob.delete()
        print(f"  🗑 Deleted old: {blob.name}")

    for fname in os.listdir(local_dir):
        local_path = os.path.join(local_dir, fname)
        bucket.blob(f"{gcs_prefix}/{fname}").upload_from_filename(local_path)
        print(f"  ↑ {market}: {gcs_prefix}/{fname}")
print("  ✅ All artefacts uploaded")

# ── Step 4: Register models ───────────────────────────────────────────────────
print("\n[4/7] Registering models in Vertex AI Model Registry...")
we_model = aiplatform.Model.upload(
    display_name                = "western-europe-promo-forecaster-v1",
    artifact_uri                = f"{BUCKET_URI}/artifacts/western_europe",
    serving_container_image_uri = XGBOOST_CONTAINER,
    description                 = "XGBoost 1.7 — Western Europe promo sell-out · R²=0.81 · 97 features",
    labels                      = {"market": "western_europe", "version": "v1"},
)
print(f"  ✅ Western Europe model: {we_model.resource_name}")

sea_model = aiplatform.Model.upload(
    display_name                = "southeast-asia-promo-forecaster-v1",
    artifact_uri                = f"{BUCKET_URI}/artifacts/southeast_asia",
    serving_container_image_uri = XGBOOST_CONTAINER,
    description                 = "XGBoost 1.7 — Southeast Asia promo sell-out · R²=0.70 · 78 features",
    labels                      = {"market": "southeast_asia", "version": "v1"},
)
print(f"  ✅ Southeast Asia model: {sea_model.resource_name}")

# ── Step 5: Create endpoints ──────────────────────────────────────────────────
print("\n[5/7] Creating endpoints...")
we_endpoint = aiplatform.Endpoint.create(
    display_name = "western-europe-promo-forecaster",
    labels       = {"market": "western_europe"},
)
print(f"  ✅ Western Europe endpoint: {we_endpoint.name}")

sea_endpoint = aiplatform.Endpoint.create(
    display_name = "southeast-asia-promo-forecaster",
    labels       = {"market": "southeast_asia"},
)
print(f"  ✅ Southeast Asia endpoint: {sea_endpoint.name}")

# ── Step 6: Deploy models ─────────────────────────────────────────────────────
print("\n[6/7] Deploying models (async — both markets deploy in parallel)...")

# sync=False is critical here — synchronous deployment blocks for ~15 min per model.
# Async allows both markets to deploy simultaneously and returns immediately.
we_endpoint.deploy(
    model                       = we_model,
    deployed_model_display_name = "western-europe-xgb-v1",
    machine_type                = "n1-standard-2",
    min_replica_count           = 1,
    max_replica_count           = 2,
    traffic_split               = {"0": 100},
    sync                        = False,         # async deployment — do not block
)
print("  ✅ Western Europe deployment started")

sea_endpoint.deploy(
    model                       = sea_model,
    deployed_model_display_name = "southeast-asia-xgb-v1",
    machine_type                = "n1-standard-2",
    min_replica_count           = 1,
    max_replica_count           = 2,
    traffic_split               = {"0": 100},
    sync                        = False,         # async deployment — do not block
)
print("  ✅ Southeast Asia deployment started")

# ── Step 7: Save session info ─────────────────────────────────────────────────
# Persists endpoint and model resource names for use by deploy_monitoring.py
print("\n[7/7] Saving session info...")
session = {
    "western_europe_endpoint": we_endpoint.resource_name,
    "southeast_asia_endpoint": sea_endpoint.resource_name,
    "western_europe_model":    we_model.resource_name,
    "southeast_asia_model":    sea_model.resource_name,
}
with open("models/vertex_session.json", "w") as f:
    json.dump(session, f, indent=2)

print("\n" + "=" * 60)
print("  DEPLOYMENT STARTED SUCCESSFULLY")
print("=" * 60)
print(f"\n  Western Europe endpoint : {we_endpoint.resource_name}")
print(f"  Southeast Asia endpoint : {sea_endpoint.resource_name}")
print(f"\n  ⏳ Both models deploying in background (~15 min each)")
print(f"  📊 Check status: https://console.cloud.google.com/vertex-ai/endpoints?project={PROJECT_ID}")
print(f"\n  Once both show ACTIVE, run deploy_monitoring.py")
print("=" * 60)
