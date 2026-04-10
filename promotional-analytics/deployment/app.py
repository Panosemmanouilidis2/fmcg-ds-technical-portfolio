"""
FMCG Promotion Sell-Out Forecaster
Streamlit application — Trade Planning Interface
Western Europe Market · XGBoost R²=0.81

Deployed on Google Cloud Run, connected to a live Vertex AI prediction endpoint.
Allows trade planners to forecast promotional sell-out volume and estimate ROI
before committing trade spend — no technical skills required.

Author: Panos Emmanouilidis
"""

import json
import os
import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="FMCG Promo Forecaster", page_icon="📦", layout="wide")

# ══════════════════════════════════════════════════════════════════════════════
# FEATURE NAMES
# 97 features used by the trained XGBoost model.
# Order is fixed — must match the feature order used during training exactly.
# ══════════════════════════════════════════════════════════════════════════════
MODEL_FEATURES = [
    'Level8Code','TUEAN','WeekSkID','InStoreStartWeek','InStoreEndWeek',
    'PromoShopperMechanic','SegmentName_VG','FormName_VG','EBFName_VG',
    'Brand_VG','SPF','SPFVName_VG','ListPrice',
    'PlannedPromoSalesVolumeSellIn','PlannedNetPromoGSVSellIn',
    'PlannedTTSOnSpend','PlannedNetPromoNIVSellIn','PlannedTTSOffSpend',
    'PlannedNetPromoTOSellIn','PlannedNetPromoGrossProfitsSellIn',
    'PlannedNetPromoCOGSSellIn','PlannedBaselineVolume','PlannedBaseGSVSellIn',
    'PlannedBaseTTSOnSpend','PlannedBaseNIVSellIn','PlannedBaseTTSOffSpend',
    'PlannedBaseTOSellIn','PlannedBaseGrossProfitsSellIn','PlannedBaseCOGSSellIn',
    'PlannedEventSpend','IsPipeFill','PlannedIncrementalVolume',
    'PlannedUpliftRate','PlannedTTSTotal','PlannedCostPerUnit',
    'PlannedROI_proxy','PromoDurationWeeks','IsDefensivePromo',
    'Customer_ASDA STORES LTD.','Customer_BOOTS UK LIMITED',
    'Customer_SAINSBURYS SUPERMARKETS LTD','Customer_T J MORRIS LTD',
    'Customer_TESCO STORES LTD','Customer_WAITROSE LTD',
    'Customer_WM MORRISON SUPERMARKETS LIMITED',
    'PromotionStatus_Executed','PromotionStatus_InFlight',
    'DivisionName_VG_FOODS CATEGORY','DivisionName_VG_HPC CATEGORY',
    'PromoMechanic_EDLP','PromoMechanic_Loyalty','PromoMechanic_Multi-Buy',
    'PromoMechanic_Other','PromoMechanic_Pipe Fill',
    'PromoMechanic_Shopper Marketing','PromoMechanic_Special Packs / Offer',
    'PromoMechanic_TPR',
    'PromoFeature_Check out end','PromoFeature_Event',
    'PromoFeature_Free Standing Unit','PromoFeature_Gondola End',
    'PromoFeature_Hot Spot','PromoFeature_In queue fixture',
    'PromoFeature_Ladder Rack','PromoFeature_Mid Gondola',
    'PromoFeature_None Specified','PromoFeature_Online',
    'PromoFeature_Pallet Drop','PromoFeature_Plinth','PromoFeature_Shelf',
    'PromoFeature_Shipper/OFD','PromoFeature_Side Stack',
    'PromoFeature_Store Entrance',
    'CategoryName_VG_BEVERAGE','CategoryName_VG_DEODORANT & FRAGRANCE',
    'CategoryName_VG_DRESSING','CategoryName_VG_FABRIC CLEANING',
    'CategoryName_VG_FABRIC ENHANCER','CategoryName_VG_HAIR CARE',
    'CategoryName_VG_HEALTHY SNACKING','CategoryName_VG_HOME & HYGIENE',
    'CategoryName_VG_ICE CREAM CATEGORY',
    'CategoryName_VG_NON CORPORATE PC CATEGORY','CategoryName_VG_ORAL CARE',
    'CategoryName_VG_OTH NUTRITION','CategoryName_VG_PLANT BASED MEAT',
    'CategoryName_VG_SCRATCH COOKING AID','CategoryName_VG_SKIN CARE',
    'CategoryName_VG_SKIN CLEANSING',
    'InstoreStartDate_Month','InstoreStartDate_Week',
    'InstoreEndDate_Month','InstoreEndDate_Week',
    'ShipmentStartDate_Month','ShipmentStartDate_Week',
    'ShipmentEndDate_Month','ShipmentEndDate_Week',
]

# Prefixes used to identify one-hot encoded columns during inference
ONE_HOT_PREFIXES = [
    'Customer_','PromotionStatus_','DivisionName_VG_',
    'PromoMechanic_','PromoFeature_','CategoryName_VG_'
]

# ══════════════════════════════════════════════════════════════════════════════
# PLAIN ENGLISH MAPPINGS
# Maps user-friendly UI labels to the internal feature values expected by the model.
# Allows non-technical planners to use the interface without knowing column names.
# ══════════════════════════════════════════════════════════════════════════════
MECHANIC_LABELS = {
    "Temporary Price Reduction":  "TPR",
    "Every Day Low Price":        "EDLP",
    "Multi-Buy Deal":             "Multi-Buy",
    "Loyalty Card Promotion":     "Loyalty",
    "Special Pack Offer":         "Special Packs / Offer",
    "Shopper Marketing Event":    "Shopper Marketing",
    "Pipeline Fill":              "Pipe Fill",
    "Other Promotion":            "Other",
}
FEATURE_LABELS = {
    "No Feature Display":       "None Specified",
    "Gondola End Display":      "Gondola End",
    "Checkout End Display":     "Check out end",
    "Pallet Display":           "Pallet Drop",
    "Shelf Display":            "Shelf",
    "Mid-Gondola Display":      "Mid Gondola",
    "Store Entrance Display":   "Store Entrance",
    "Shipper Display":          "Shipper/OFD",
    "Free Standing Unit":       "Free Standing Unit",
    "Hot Spot Display":         "Hot Spot",
    "Online Promotion":         "Online",
    "In-Store Event":           "Event",
    "Side Stack Display":       "Side Stack",
    "Plinth Display":           "Plinth",
    "Ladder Rack":              "Ladder Rack",
    "Queue Fixture":            "In queue fixture",
}
CUSTOMER_LABELS = {
    "Tesco":         "TESCO STORES LTD",
    "Sainsbury's":   "SAINSBURYS SUPERMARKETS LTD",
    "Asda":          "ASDA STORES LTD.",
    "Morrisons":     "WM MORRISON SUPERMARKETS LIMITED",
    "Waitrose":      "WAITROSE LTD",
    "Boots":         "BOOTS UK LIMITED",
    "Home Bargains": "T J MORRIS LTD",
}
DIVISION_LABELS = {
    "Home & Personal Care": "HPC CATEGORY",
    "Food & Beverage":      "FOODS CATEGORY",
}
CATEGORY_LABELS = {
    "Skin Care":             "SKIN CARE",
    "Skin Cleansing":        "SKIN CLEANSING",
    "Hair Care":             "HAIR CARE",
    "Deodorant & Fragrance": "DEODORANT & FRAGRANCE",
    "Fabric Cleaning":       "FABRIC CLEANING",
    "Fabric Enhancer":       "FABRIC ENHANCER",
    "Home & Hygiene":        "HOME & HYGIENE",
    "Oral Care":             "ORAL CARE",
    "Ice Cream":             "ICE CREAM CATEGORY",
    "Healthy Snacking":      "HEALTHY SNACKING",
    "Beverage":              "BEVERAGE",
    "Dressing":              "DRESSING",
    "Scratch Cooking Aid":   "SCRATCH COOKING AID",
    "Plant-Based Meat":      "PLANT BASED MEAT",
    "Other Nutrition":       "OTH NUTRITION",
    "Other":                 "NON CORPORATE PC CATEGORY",
}
MONTH_TO_WEEK = {
    "January":2,"February":6,"March":10,"April":15,
    "May":19,"June":24,"July":28,"August":32,
    "September":37,"October":41,"November":45,"December":50,
}

# ══════════════════════════════════════════════════════════════════════════════
# MODEL LOADER — Vertex AI endpoint
# Replace the endpoint path below with your own Vertex AI endpoint resource name.
# Format: projects/YOUR_PROJECT_ID/locations/YOUR_REGION/endpoints/YOUR_ENDPOINT_ID
# ══════════════════════════════════════════════════════════════════════════════
ENDPOINT_PATH = (
    "projects/YOUR_PROJECT_ID/locations/YOUR_REGION/endpoints/YOUR_ENDPOINT_ID"
)

@st.cache_resource
def load_resources():
    """
    Initialises the Vertex AI client and loads supporting artefacts:
    - Live prediction endpoint (XGBoost model served via Vertex AI)
    - Feature median values (used to fill unspecified inputs at inference time)
    - Scaling group definitions (used to proportionally adjust related financial features)
    """
    from google.cloud import aiplatform
    aiplatform.init(project="YOUR_GCP_PROJECT_ID", location="YOUR_REGION")
    endpoint = aiplatform.Endpoint(ENDPOINT_PATH)
    feature_names = MODEL_FEATURES
    with open("models/feature_medians.json") as f:
        medians = json.load(f)
    with open("models/scaling_groups.json") as f:
        scaling = json.load(f)
    return endpoint, feature_names, medians, scaling

# ══════════════════════════════════════════════════════════════════════════════
# PREDICTION
# Constructs the full 97-feature input vector from planner inputs,
# calls the Vertex AI endpoint, and derives business metrics from the prediction.
# ══════════════════════════════════════════════════════════════════════════════
def predict(c, endpoint, feature_names, medians, scaling):
    """
    Builds the feature vector, calls the live Vertex AI endpoint,
    and returns predicted volume plus derived business metrics:
    - Uplift vs baseline (units and %)
    - Estimated incremental gross profit (iGP)
    - Return on investment (ROI)
    - Cost per predicted unit
    """
    pv  = float(c["planned_volume"])
    bv  = float(c["baseline_volume"])
    sp  = float(c["planned_spend"])
    dur = float(c["duration_weeks"])
    mw  = MONTH_TO_WEEK[c["start_month"]]
    mn  = list(MONTH_TO_WEEK.keys()).index(c["start_month"]) + 1

    # Start from median feature values — ensures all 97 features are populated
    row = {col: float(medians.get(col, 0.0)) for col in feature_names}

    # Scale financial features proportionally to planned and baseline volumes
    pv_ratio = pv / max(float(scaling["median_planned_volume"]), 1)
    bv_ratio = bv / max(float(scaling["median_baseline_volume"]), 1)
    for col in scaling["promo_volume_cols"]:
        if col in row: row[col] = float(medians.get(col, 0.0)) * pv_ratio
    for col in scaling["baseline_volume_cols"]:
        if col in row: row[col] = float(medians.get(col, 0.0)) * bv_ratio

    # Set core planned financial features directly from planner inputs
    row["PlannedPromoSalesVolumeSellIn"] = pv
    row["PlannedBaselineVolume"]         = bv
    row["PlannedIncrementalVolume"]      = pv - bv
    row["PlannedUpliftRate"]             = pv / max(bv, 1)
    row["PlannedTTSTotal"]               = sp
    row["PlannedTTSOnSpend"]             = sp * 0.6
    row["PlannedTTSOffSpend"]            = sp * 0.4
    row["PlannedCostPerUnit"]            = sp / max(pv, 1)
    row["PromoDurationWeeks"]            = dur
    row["IsDefensivePromo"]              = 1.0 if (pv - bv) < 0 else 0.0
    row["IsPipeFill"]                    = 1.0 if c["mechanic_code"] == "Pipe Fill" else 0.0
    row["InstoreStartDate_Month"]        = mn
    row["InstoreStartDate_Week"]         = mw
    row["InstoreEndDate_Month"]          = mn
    row["InstoreEndDate_Week"]           = mw + dur

    # Reset all one-hot columns to zero before setting the selected values
    for col in feature_names:
        if any(col.startswith(p) for p in ONE_HOT_PREFIXES):
            row[col] = 0.0
    row["PromotionStatus_Executed"] = 1.0

    # Set the one-hot flag for each selected categorical dimension
    for col_key, prefix in [
        ("customer_code", "Customer_"),
        ("division_code", "DivisionName_VG_"),
        ("mechanic_code", "PromoMechanic_"),
        ("feature_code",  "PromoFeature_"),
        ("category_code", "CategoryName_VG_"),
    ]:
        col_name = prefix + c[col_key]
        if col_name in row: row[col_name] = 1.0

    # Send ordered feature list to Vertex AI endpoint and retrieve log-scale prediction
    payload  = [float(row[col]) for col in feature_names]
    response = endpoint.predict(instances=[payload])
    log_pred = float(response.predictions[0])

    # Model was trained on log1p(volume) — inverse transform to recover unit volume
    pred_vol = max(float(np.expm1(log_pred)), 0)

    # Derive business metrics from predicted volume
    uplift     = pred_vol - bv
    uplift_pct = (uplift / max(bv, 1)) * 100
    igp        = (pred_vol - bv) * 2.75 * 0.35   # incremental gross profit estimate
    roi        = (igp - sp) / max(sp, 1)

    return {
        "predicted_volume": pred_vol,
        "uplift_units":     uplift,
        "uplift_pct":       uplift_pct,
        "cost_per_unit":    sp / max(pred_vol, 1),
        "roi":              roi,
    }

# ══════════════════════════════════════════════════════════════════════════════
# CAMPAIGN FORM
# Renders a single campaign input form for the planner.
# All inputs use plain English labels mapped internally to model feature values.
# ══════════════════════════════════════════════════════════════════════════════
def campaign_form(idx):
    st.markdown(f"### Campaign {idx}")
    c1, c2 = st.columns(2)
    with c1:
        cust  = st.selectbox("Retailer",          list(CUSTOMER_LABELS.keys()), key=f"cust_{idx}")
        mech  = st.selectbox("Promotion Type",    list(MECHANIC_LABELS.keys()), key=f"mech_{idx}")
        feat  = st.selectbox("In-Store Display",  list(FEATURE_LABELS.keys()),  key=f"feat_{idx}")
        div   = st.selectbox("Business Division", list(DIVISION_LABELS.keys()), key=f"div_{idx}")
        cat   = st.selectbox("Product Category",  list(CATEGORY_LABELS.keys()), key=f"cat_{idx}")
    with c2:
        month = st.selectbox("Promotion Month",   list(MONTH_TO_WEEK.keys()),   key=f"month_{idx}")
        pv    = st.number_input("Planned Sales Volume (units)",          100, 5_000_000, 10_000, 500,    key=f"pv_{idx}")
        bv    = st.number_input("Normal Weekly Sales Without Promotion", 100, 5_000_000,  5_000, 500,    key=f"bv_{idx}")
        sp    = st.number_input("Promotion Budget (£)",                    0, 10_000_000, 50_000, 1_000, key=f"sp_{idx}")
        dur   = st.slider("Promotion Duration (weeks)", 1, 12, 2, key=f"dur_{idx}")
    return {
        "label":          f"Campaign {idx}",
        "customer_label": cust,  "mechanic_label": mech,
        "feature_label":  feat,  "division_label": div,
        "category_label": cat,   "start_month":    month,
        "customer_code":  CUSTOMER_LABELS[cust],
        "mechanic_code":  MECHANIC_LABELS[mech],
        "feature_code":   FEATURE_LABELS[feat],
        "division_code":  DIVISION_LABELS[div],
        "category_code":  CATEGORY_LABELS[cat],
        "planned_volume": pv, "baseline_volume": bv,
        "planned_spend":  sp, "duration_weeks":  dur,
    }

# ══════════════════════════════════════════════════════════════════════════════
# RESULT CARD
# Renders a forecast result card for a single campaign.
# ══════════════════════════════════════════════════════════════════════════════
def result_card(c, r, is_winner=False):
    sign = "+" if r["uplift_units"] >= 0 else ""
    if is_winner:
        st.success(f"🏆 **{c['label']}** — Best Campaign")
    else:
        st.info(f"**{c['label']}**")
    st.caption(f"{c['customer_label']}  ·  {c['mechanic_label']}  ·  {c['start_month']}")
    st.metric("📦 Predicted Sell-Out Volume",
              f"{r['predicted_volume']:,.0f} units",
              f"{sign}{r['uplift_units']:,.0f} vs normal trading")
    ca, cb = st.columns(2)
    with ca:
        st.metric("📈 Sales Uplift", f"{sign}{r['uplift_pct']:.1f}%")
    with cb:
        st.metric("💰 Est. Return on Investment",
                  f"{'+'if r['roi']>=0 else ''}{r['roi']:.2f}x")
    with st.expander("See full details"):
        st.write(f"**Retailer:** {c['customer_label']}")
        st.write(f"**Promotion Type:** {c['mechanic_label']}")
        st.write(f"**In-Store Display:** {c['feature_label']}")
        st.write(f"**Category:** {c['category_label']}")
        st.write(f"**Division:** {c['division_label']}")
        st.write(f"**Month:** {c['start_month']}")
        st.write(f"**Duration:** {c['duration_weeks']} weeks")
        st.write(f"**Budget:** £{c['planned_spend']:,.0f}")
        st.write(f"**Cost per Predicted Unit:** £{r['cost_per_unit']:.2f}")

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    st.markdown("# 📦 FMCG Promotion Forecaster")
    st.markdown("Predict sell-out volumes and compare promotions · **Western Europe Market** · XGBoost R²=0.81")
    st.divider()

    try:
        endpoint, feature_names, medians, scaling = load_resources()
    except FileNotFoundError as e:
        st.error(f"⚠️ Resource file not found: {e}")
        st.stop()

    st.markdown("### How many promotions do you want to compare?")
    n = st.radio("", [2, 3, 4], horizontal=True,
                 format_func=lambda x: f"{x} Campaigns",
                 label_visibility="collapsed")
    st.divider()

    campaigns = []
    for col, i in zip(st.columns(n), range(1, n + 1)):
        with col:
            campaigns.append(campaign_form(i))

    st.divider()
    _, btn, _ = st.columns([1, 2, 1])
    with btn:
        run = st.button("🔮  Run Forecast for All Campaigns",
                        type="primary", use_container_width=True)

    if run:
        st.divider()
        with st.spinner("Running forecast models..."):
            results = [predict({**c, "endpoint": endpoint}, endpoint, feature_names, medians, scaling) for c in campaigns]

        best = max(range(len(results)), key=lambda i: results[i]["predicted_volume"])
        bc, br = campaigns[best], results[best]
        st.success(
            f"🏆 **Best Campaign: {bc['label']}** — "
            f"{bc['customer_label']} · {bc['mechanic_label']} · {bc['start_month']} · "
            f"Predicted **{br['predicted_volume']:,.0f} units** "
            f"(+{br['uplift_pct']:.1f}% uplift)"
        )

        st.markdown("### Side-by-Side Results")
        for col, c, r, i in zip(st.columns(n), campaigns, results, range(n)):
            with col:
                result_card(c, r, is_winner=(i == best))

        st.divider()
        st.markdown("### Full Comparison Table")
        rows = [{
            "Campaign":         c["label"],
            "Retailer":         c["customer_label"],
            "Promotion Type":   c["mechanic_label"],
            "Month":            c["start_month"],
            "Category":         c["category_label"],
            "Budget (£)":       f"£{c['planned_spend']:,.0f}",
            "Duration":         f"{c['duration_weeks']} wks",
            "Predicted Volume": f"{r['predicted_volume']:,.0f}",
            "Uplift vs Normal": f"{'+'if r['uplift_pct']>=0 else ''}{r['uplift_pct']:.1f}%",
            "Est. ROI":         f"{'+'if r['roi']>=0 else ''}{r['roi']:.2f}x",
            "Cost/Unit (£)":    f"£{r['cost_per_unit']:.2f}",
        } for c, r in zip(campaigns, results)]

        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.download_button("📥  Download as CSV",
                           data=pd.DataFrame(rows).to_csv(index=False),
                           file_name="promo_forecast_comparison.csv",
                           mime="text/csv")

    st.divider()
    st.caption("FMCG Promotional Analytics · XGBoost · Western Europe Market · R²=0.81")

if __name__ == "__main__":
    main()
