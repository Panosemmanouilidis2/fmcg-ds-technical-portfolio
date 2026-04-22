"""
Unilever Promotion Sell-Out Forecaster
Streamlit app for Marketing Director demo
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
import os

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Unilever Promo Forecaster",
    page_icon="🧴",
    layout="wide"
)

# ══════════════════════════════════════════════════════════════════════════════
# PLAIN ENGLISH LABEL MAPPINGS
# ══════════════════════════════════════════════════════════════════════════════

MECHANIC_LABELS = {
    "Temporary Price Reduction":    "TPR",
    "Every Day Low Price":          "EDLP",
    "Multi-Buy Deal":               "Multi-Buy",
    "Loyalty Card Promotion":       "Loyalty",
    "Special Pack Offer":           "Special Packs / Offer",
    "Shopper Marketing Event":      "Shopper Marketing",
    "Pipeline Fill":                "Pipe Fill",
    "Other Promotion":              "Other",
}

FEATURE_LABELS = {
    "No Feature Display":           "None Specified",
    "Not Specified":                "Unknown",
    "Gondola End Display":          "Gondola End",
    "Checkout End Display":         "Check out end",
    "Pallet Display":               "Pallet Drop",
    "Shelf Display":                "Shelf",
    "Mid-Gondola Display":          "Mid Gondola",
    "Store Entrance Display":       "Store Entrance",
    "Shipper Display":              "Shipper/OFD",
    "Free Standing Unit":           "Free Standing Unit",
    "Hot Spot Display":             "Hot Spot",
    "Online Promotion":             "Online",
    "In-Store Event":               "Event",
    "Side Stack Display":           "Side Stack",
    "Plinth Display":               "Plinth",
    "Ladder Rack":                  "Ladder Rack",
    "Queue Fixture":                "In queue fixture",
}

CUSTOMER_LABELS = {
    "Tesco":            "TESCO STORES LTD",
    "Sainsbury's":      "SAINSBURYS SUPERMARKETS LTD",
    "Asda":             "ASDA STORES LTD.",
    "Morrisons":        "WM MORRISON SUPERMARKETS LIMITED",
    "Waitrose":         "WAITROSE LTD",
    "Boots":            "BOOTS UK LIMITED",
    "Home Bargains":    "T J MORRIS LTD",
}

DIVISION_LABELS = {
    "Home & Personal Care (HPC)":   "HPC CATEGORY",
    "Food & Beverage":              "FOODS CATEGORY",
}

CATEGORY_LABELS = {
    "Skin Care":                "SKIN CARE",
    "Skin Cleansing":           "SKIN CLEANSING",
    "Hair Care":                "HAIR CARE",
    "Deodorant & Fragrance":    "DEODORANT & FRAGRANCE",
    "Fabric Cleaning":          "FABRIC CLEANING",
    "Fabric Enhancer":          "FABRIC ENHANCER",
    "Home & Hygiene":           "HOME & HYGIENE",
    "Oral Care":                "ORAL CARE",
    "Ice Cream":                "ICE CREAM CATEGORY",
    "Healthy Snacking":         "HEALTHY SNACKING",
    "Beverage":                 "BEVERAGE",
    "Dressing":                 "DRESSING",
    "Scratch Cooking Aid":      "SCRATCH COOKING AID",
    "Plant-Based Meat":         "PLANT BASED MEAT",
    "Other Nutrition":          "OTH NUTRITION",
    "Other":                    "NON CORPORATE PC CATEGORY",
}

# ══════════════════════════════════════════════════════════════════════════════
# MODEL LOADER (cached so it only loads once)
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource
def load_model_and_encoders():
    booster = xgb.Booster()
    booster.load_model("models/xgb_uk_tuned.json")
    feat_cols = pickle.load(open("models/feature_cols_uk.pkl", "rb"))
    return booster, feat_cols

# ══════════════════════════════════════════════════════════════════════════════
# PREDICTION FUNCTION
# ══════════════════════════════════════════════════════════════════════════════
def predict(campaign: dict, booster, model_features: list) -> dict:
    """
    Build a feature row from campaign inputs and return predictions.
    campaign keys use internal codes (post-mapping from plain English).
    """

    # Start with all zeros
    row = {col: 0.0 for col in model_features}

    # ── Numeric features ──────────────────────────────────────────────────────
    planned_vol      = float(campaign.get("planned_volume", 5000))
    baseline_vol     = float(campaign.get("baseline_volume", 3000))
    planned_spend    = float(campaign.get("planned_spend", 50000))
    duration_weeks   = float(campaign.get("duration_weeks", 2))

    row["PlannedPromoSalesVolumeSellIn"] = planned_vol
    row["PlannedBaselineVolume"]         = baseline_vol
    row["PlannedTTSTotal"]               = planned_spend
    row["PlannedTTSOnSpend"]             = planned_spend * 0.6
    row["PlannedTTSOffSpend"]            = planned_spend * 0.4
    row["PlannedEventSpend"]             = 0.0
    row["PromoDurationWeeks"]            = duration_weeks
    row["PlannedIncrementalVolume"]      = planned_vol - baseline_vol
    row["PlannedUpliftRate"]             = planned_vol / max(baseline_vol, 1)
    row["PlannedCostPerUnit"]            = planned_spend / max(planned_vol, 1)
    row["PlannedROI_proxy"]              = 1.0
    row["IsDefensivePromo"]              = 1.0 if (planned_vol - baseline_vol) < 0 else 0.0
    row["IsPipeFill"]                    = 1.0 if campaign.get("mechanic_code") == "Pipe Fill" else 0.0

    # ── One-hot: Customer ─────────────────────────────────────────────────────
    cust_col = f"Customer_{campaign.get('customer_code', '')}"
    if cust_col in row:
        row[cust_col] = 1.0

    # ── One-hot: PromotionStatus (always Executed for forecasting) ────────────
    if "PromotionStatus_Executed" in row:
        row["PromotionStatus_Executed"] = 1.0

    # ── One-hot: Division ─────────────────────────────────────────────────────
    div_col = f"DivisionName_VG_{campaign.get('division_code', '')}"
    if div_col in row:
        row[div_col] = 1.0

    # ── One-hot: PromoMechanic ────────────────────────────────────────────────
    mech_col = f"PromoMechanic_{campaign.get('mechanic_code', '')}"
    if mech_col in row:
        row[mech_col] = 1.0

    # ── One-hot: PromoFeature ─────────────────────────────────────────────────
    feat_col = f"PromoFeature_{campaign.get('feature_code', 'None Specified')}"
    if feat_col in row:
        row[feat_col] = 1.0

    # ── One-hot: Category ─────────────────────────────────────────────────────
    cat_col = f"CategoryName_VG_{campaign.get('category_code', '')}"
    if cat_col in row:
        row[cat_col] = 1.0

    # ── Build DMatrix and predict ─────────────────────────────────────────────
    X = pd.DataFrame([row], columns=model_features).astype(float)
    dmat = xgb.DMatrix(X, feature_names=model_features)
    log_pred = booster.predict(dmat)[0]
    pred_vol = float(np.expm1(log_pred))

    # ── Derived business metrics ──────────────────────────────────────────────
    uplift      = pred_vol - baseline_vol
    uplift_pct  = (uplift / max(baseline_vol, 1)) * 100
    cost_per_u  = planned_spend / max(pred_vol, 1)
    roi         = (uplift * 2.5 - planned_spend) / max(planned_spend, 1)  # proxy iGP

    return {
        "predicted_volume": pred_vol,
        "uplift_units":     uplift,
        "uplift_pct":       uplift_pct,
        "cost_per_unit":    cost_per_u,
        "roi":              roi,
    }

# ══════════════════════════════════════════════════════════════════════════════
# CAMPAIGN INPUT FORM
# ══════════════════════════════════════════════════════════════════════════════
def campaign_form(idx: int):
    """Renders one campaign input form. Returns a dict of inputs."""
    st.markdown(f"### Campaign {idx}")

    col1, col2 = st.columns(2)

    with col1:
        customer_label  = st.selectbox(
            "Retailer",
            list(CUSTOMER_LABELS.keys()),
            key=f"customer_{idx}"
        )
        mechanic_label  = st.selectbox(
            "Promotion Type",
            list(MECHANIC_LABELS.keys()),
            key=f"mechanic_{idx}"
        )
        feature_label   = st.selectbox(
            "In-Store Display",
            list(FEATURE_LABELS.keys()),
            key=f"feature_{idx}"
        )
        division_label  = st.selectbox(
            "Business Division",
            list(DIVISION_LABELS.keys()),
            key=f"division_{idx}"
        )

    with col2:
        category_label  = st.selectbox(
            "Product Category",
            list(CATEGORY_LABELS.keys()),
            key=f"category_{idx}"
        )
        planned_vol     = st.number_input(
            "Planned Sales Volume (units)",
            min_value=100, max_value=5_000_000,
            value=10_000, step=500,
            key=f"planned_vol_{idx}"
        )
        baseline_vol    = st.number_input(
            "Normal Weekly Sales Without Promotion (units)",
            min_value=100, max_value=5_000_000,
            value=5_000, step=500,
            key=f"baseline_vol_{idx}"
        )
        planned_spend   = st.number_input(
            "Promotion Budget (£)",
            min_value=0, max_value=10_000_000,
            value=50_000, step=1_000,
            key=f"planned_spend_{idx}"
        )
        duration_weeks  = st.slider(
            "Promotion Duration (weeks)",
            min_value=1, max_value=12,
            value=2,
            key=f"duration_{idx}"
        )

    return {
        "label":          f"Campaign {idx}",
        "customer_label": customer_label,
        "mechanic_label": mechanic_label,
        "feature_label":  feature_label,
        "division_label": division_label,
        "category_label": category_label,
        "customer_code":  CUSTOMER_LABELS[customer_label],
        "mechanic_code":  MECHANIC_LABELS[mechanic_label],
        "feature_code":   FEATURE_LABELS[feature_label],
        "division_code":  DIVISION_LABELS[division_label],
        "category_code":  CATEGORY_LABELS[category_label],
        "planned_volume": planned_vol,
        "baseline_volume": baseline_vol,
        "planned_spend":  planned_spend,
        "duration_weeks": duration_weeks,
    }

# ══════════════════════════════════════════════════════════════════════════════
# RESULTS CARD
# ══════════════════════════════════════════════════════════════════════════════
def result_card(campaign: dict, result: dict):
    roi      = result["roi"]
    uplift   = result["uplift_units"]
    pred_vol = result["predicted_volume"]

    roi_colour   = "🟢" if roi > 0 else "🔴"
    uplift_sign  = "+" if uplift >= 0 else ""

    st.markdown(f"""
<div style="
    background: #1e1e2e;
    border: 1px solid #3a3a5c;
    border-radius: 12px;
    padding: 20px;
    height: 100%;
">
    <h4 style="color:#a0aec0; margin-bottom:4px;">{campaign['label']}</h4>
    <p style="color:#718096; font-size:13px; margin-bottom:16px;">
        {campaign['customer_label']} · {campaign['mechanic_label']}
    </p>

    <p style="color:#e2e8f0; font-size:13px; margin-bottom:4px;">📦 Predicted Sell-Out Volume</p>
    <h2 style="color:#63b3ed; margin:0 0 16px 0;">{pred_vol:,.0f} units</h2>

    <p style="color:#e2e8f0; font-size:13px; margin-bottom:4px;">📈 Sales Uplift vs Normal Trading</p>
    <h3 style="color:#68d391; margin:0 0 16px 0;">{uplift_sign}{uplift:,.0f} units
        ({uplift_sign}{result['uplift_pct']:.1f}%)</h3>

    <p style="color:#e2e8f0; font-size:13px; margin-bottom:4px;">{roi_colour} Estimated Return on Investment</p>
    <h3 style="color:{'#68d391' if roi > 0 else '#fc8181'}; margin:0 0 16px 0;">
        {'+' if roi > 0 else ''}{roi:.2f}x
    </h3>

    <hr style="border-color:#3a3a5c; margin: 16px 0;">

    <p style="color:#718096; font-size:12px;">
        💰 Budget: £{campaign['planned_spend']:,.0f} &nbsp;|&nbsp;
        📅 Duration: {campaign['duration_weeks']} week{'s' if campaign['duration_weeks'] > 1 else ''} &nbsp;|&nbsp;
        🏪 {campaign['category_label']}
    </p>
    <p style="color:#718096; font-size:12px;">
        🖥️ Display: {campaign['feature_label']} &nbsp;|&nbsp;
        💷 Cost per unit: £{result['cost_per_unit']:.2f}
    </p>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# WINNER BANNER
# ══════════════════════════════════════════════════════════════════════════════
def winner_banner(campaigns, results):
    best_idx = max(range(len(results)), key=lambda i: results[i]["predicted_volume"])
    best_c   = campaigns[best_idx]
    best_r   = results[best_idx]

    st.success(
        f"🏆 **Best Campaign: {best_c['label']}** — "
        f"{best_c['customer_label']} · {best_c['mechanic_label']} · "
        f"Predicted **{best_r['predicted_volume']:,.0f} units** "
        f"({'+' if best_r['uplift_pct'] >= 0 else ''}{best_r['uplift_pct']:.1f}% uplift)"
    )

# ══════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ══════════════════════════════════════════════════════════════════════════════
def main():
    # Header
    st.markdown("""
    <div style="padding: 16px 0 8px 0;">
        <h1 style="margin:0;">🧴 Unilever Promotion Forecaster</h1>
        <p style="color:#718096; margin:4px 0 0 0;">
            Predict sell-out volumes and compare promotions before committing budget · UK Market · Powered by Machine Learning
        </p>
    </div>
    <hr style="margin: 16px 0;">
    """, unsafe_allow_html=True)

    # Load model
    try:
        booster, model_features = load_model_and_encoders()
    except FileNotFoundError:
        st.error("⚠️ Model files not found. Make sure the `models/` folder is in the same directory as this app.")
        st.stop()

    # Number of campaigns
    st.markdown("### How many promotions do you want to compare?")
    n_campaigns = st.radio(
        "",
        [2, 3, 4],
        horizontal=True,
        format_func=lambda x: f"{x} Campaigns",
        label_visibility="collapsed"
    )

    st.markdown("---")

    # Render campaign forms
    campaigns = []
    cols = st.columns(n_campaigns)
    for i, col in enumerate(cols):
        with col:
            c = campaign_form(i + 1)
            campaigns.append(c)

    st.markdown("---")

    # Forecast button
    col_btn = st.columns([1, 2, 1])
    with col_btn[1]:
        run = st.button(
            "🔮  Run Forecast for All Campaigns",
            type="primary",
            use_container_width=True
        )

    if run:
        st.markdown("---")
        with st.spinner("Running forecast models..."):
            results = [predict(c, booster, model_features) for c in campaigns]

        # Winner banner
        winner_banner(campaigns, results)
        st.markdown("### Side-by-Side Results")

        # Result cards
        res_cols = st.columns(n_campaigns)
        for col, campaign, result in zip(res_cols, campaigns, results):
            with col:
                result_card(campaign, result)

        # Comparison table
        st.markdown("### Full Comparison Table")
        table_data = []
        for c, r in zip(campaigns, results):
            table_data.append({
                "Campaign":            c["label"],
                "Retailer":            c["customer_label"],
                "Promotion Type":      c["mechanic_label"],
                "In-Store Display":    c["feature_label"],
                "Category":            c["category_label"],
                "Budget (£)":          f"£{c['planned_spend']:,.0f}",
                "Duration":            f"{c['duration_weeks']} wks",
                "Predicted Volume":    f"{r['predicted_volume']:,.0f}",
                "Uplift vs Normal":    f"{'+' if r['uplift_pct'] >= 0 else ''}{r['uplift_pct']:.1f}%",
                "Est. ROI":            f"{'+' if r['roi'] >= 0 else ''}{r['roi']:.2f}x",
                "Cost per Unit (£)":   f"£{r['cost_per_unit']:.2f}",
            })

        df_table = pd.DataFrame(table_data)
        st.dataframe(df_table, use_container_width=True, hide_index=True)

        # Download button
        csv = df_table.to_csv(index=False)
        st.download_button(
            "📥  Download Comparison as CSV",
            data=csv,
            file_name="promo_forecast_comparison.csv",
            mime="text/csv"
        )

    # Footer
    st.markdown("---")
    st.markdown(
        "<p style='color:#4a5568; font-size:12px; text-align:center;'>"
        "Unilever DigiChallenge · XGBoost Model · UK Market · R² = 0.81 · "
        "Built with Vertex AI + Cloud Run</p>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
