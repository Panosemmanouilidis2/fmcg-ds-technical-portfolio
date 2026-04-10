# FMCG Promotional Analytics — Streamlit Application Container
#
# Builds the planner-facing forecast interface for deployment on Google Cloud Run.
# The app connects to a live Vertex AI prediction endpoint to serve
# real-time promotional sell-out forecasts and ROI estimates.
#
# Build:  docker build -t promo-forecaster .
# Run:    docker run -p 8080:8080 promo-forecaster
# Deploy: gcloud run deploy promo-forecaster --image gcr.io/YOUR_PROJECT/promo-forecaster

FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

EXPOSE 8080

CMD ["streamlit", "run", "app.py", \
     "--server.port=8080", \
     "--server.address=0.0.0.0", \
     "--server.enableCORS=false", \
     "--server.enableXsrfProtection=false"]
