"""
Flood Risk Early-Warning System — Streamlit Edition
-----------------------------------------------------
Redeployment of the Flask app, now powering BOTH the RA (Ridge Regression)
and DL (Keras MLP) models from a single dashboard, with an optional
LIVE WEATHER MODE that pulls real current Mumbai rainfall data (via the
free, keyless Open-Meteo API) and auto-scores risk — so the alert can
actually fire when it's genuinely raining heavily, not just on manual
demo input.

Run locally:  streamlit run streamlit_app.py
Deploy:       push to GitHub, then deploy via share.streamlit.io
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import requests
from datetime import datetime
import streamlit as st
import streamlit.components.v1 as components

# ============================================================
# CONFIG
# ============================================================
st.set_page_config(page_title="Flood Risk Early-Warning System", page_icon="🌊", layout="wide")

RA_MODEL_DIR = "model_artifacts"
DL_MODEL_DIR = "model_artifacts_dl"
DATA_FILE = "subscribers.json"
ALERT_LOG_FILE = "alert_log.json"
RISK_THRESHOLD = float(os.environ.get("FLOOD_RISK_THRESHOLD", 0.65))

# Mumbai coordinates (for live weather mode)
MUMBAI_LAT, MUMBAI_LON = 19.0760, 72.8777

FEATURE_INFO = {
    "MonsoonIntensity": "Monsoon rainfall intensity rating",
    "TopographyDrainage": "Natural terrain drainage quality",
    "RiverManagement": "River/waterway management quality",
    "Deforestation": "Deforestation level in region",
    "Urbanization": "Urbanization level",
    "ClimateChange": "Climate change impact severity",
    "DamsQuality": "Dam infrastructure quality",
    "Siltation": "Siltation / sediment build-up level",
    "AgriculturalPractices": "Agricultural practice impact",
    "Encroachments": "Waterway/floodplain encroachment level",
    "IneffectiveDisasterPreparedness": "Disaster preparedness gap",
    "DrainageSystems": "Urban drainage system quality",
    "CoastalVulnerability": "Coastal vulnerability level",
    "Landslides": "Landslide risk level",
    "Watersheds": "Watershed management quality",
    "DeterioratingInfrastructure": "Infrastructure deterioration level",
    "PopulationScore": "Population density pressure",
    "WetlandLoss": "Wetland loss severity",
    "InadequatePlanning": "Urban planning inadequacy",
    "PoliticalFactors": "Political/governance factor",
}

# A representative static profile for Mumbai (Vikhroli East / Western suburbs)
# — used as the baseline for all factors EXCEPT MonsoonIntensity, which gets
# overridden live in Live Weather Mode. Values are on the dataset's 0–16-ish
# scale, set to reflect Mumbai's known high-urbanization, high-encroachment,
# moderate-drainage profile. Fully editable by the user.
MUMBAI_BASELINE = {
    "MonsoonIntensity": 8, "TopographyDrainage": 5, "RiverManagement": 4,
    "Deforestation": 5, "Urbanization": 9, "ClimateChange": 6, "DamsQuality": 5,
    "Siltation": 6, "AgriculturalPractices": 3, "Encroachments": 8,
    "IneffectiveDisasterPreparedness": 5, "DrainageSystems": 4,
    "CoastalVulnerability": 8, "Landslides": 3, "Watersheds": 4,
    "DeterioratingInfrastructure": 6, "PopulationScore": 9, "WetlandLoss": 6,
    "InadequatePlanning": 6, "PoliticalFactors": 5,
}

FEATURE_COLUMNS = list(FEATURE_INFO.keys())

# ============================================================
# MODEL LOADING
# ============================================================
@st.cache_resource
def load_ra_model():
    try:
        model = joblib.load(os.path.join(RA_MODEL_DIR, "flood_model.pkl"))
        scaler = joblib.load(os.path.join(RA_MODEL_DIR, "scaler.pkl"))
        cols = joblib.load(os.path.join(RA_MODEL_DIR, "feature_columns.pkl"))
        uses_scaling = joblib.load(os.path.join(RA_MODEL_DIR, "uses_scaling.pkl"))
        return model, scaler, cols, uses_scaling
    except Exception:
        return None, None, None, None


@st.cache_resource
def load_dl_model():
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model(os.path.join(DL_MODEL_DIR, "flood_dl_model.keras"))
        scaler = joblib.load(os.path.join(DL_MODEL_DIR, "dl_scaler.pkl"))
        cols = joblib.load(os.path.join(DL_MODEL_DIR, "dl_feature_columns.pkl"))
        return model, scaler, cols
    except Exception:
        return None, None, None


ra_model, ra_scaler, ra_cols, ra_uses_scaling = load_ra_model()
dl_model, dl_scaler, dl_cols = load_dl_model()

RA_LOADED = ra_model is not None
DL_LOADED = dl_model is not None

TWILIO_SID = st.secrets.get("TWILIO_ACCOUNT_SID", os.environ.get("TWILIO_ACCOUNT_SID"))
TWILIO_AUTH = st.secrets.get("TWILIO_AUTH_TOKEN", os.environ.get("TWILIO_AUTH_TOKEN"))
TWILIO_FROM = st.secrets.get("TWILIO_FROM_NUMBER", os.environ.get("TWILIO_FROM_NUMBER"))
TWILIO_ENABLED = all([TWILIO_SID, TWILIO_AUTH, TWILIO_FROM])

if TWILIO_ENABLED:
    from twilio.rest import Client
    twilio_client = Client(TWILIO_SID, TWILIO_AUTH)


def send_sms(to_number: str, message: str):
    if TWILIO_ENABLED:
        try:
            twilio_client.messages.create(body=message, from_=TWILIO_FROM, to=to_number)
            return {"status": "sent", "mode": "twilio"}
        except Exception as e:
            return {"status": "failed", "mode": "twilio", "error": str(e)}
    return {"status": "simulated", "mode": "simulation"}


# ============================================================
# STORAGE HELPERS
# ============================================================
def load_json(path, default):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return default


def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# ============================================================
# LIVE WEATHER (Open-Meteo — free, no API key required)
# ============================================================
@st.cache_data(ttl=300)  # refresh every 5 minutes
def fetch_mumbai_weather():
    """Pull current precipitation data for Mumbai from Open-Meteo."""
    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={MUMBAI_LAT}&longitude={MUMBAI_LON}"
        f"&current=precipitation,rain,weather_code"
        f"&hourly=precipitation_probability,precipitation"
        f"&timezone=Asia%2FKolkata"
    )
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    return resp.json()


def rainfall_mm_to_monsoon_intensity(rain_mm_hr):
    """
    Maps real-world rainfall rate (mm/hr) to the dataset's MonsoonIntensity
    scale (observed range ~0-16 in training data). India Meteorological
    Department (IMD) rainfall categories are used as anchor points:
      0 mm/hr        -> 2   (no rain, baseline low intensity)
      0.1–7.5 mm/hr  -> 4-6 (light to moderate rain)
      7.5–35 mm/hr   -> 7-10 (rather heavy to heavy rain)
      35–125 mm/hr   -> 11-14 (very heavy rain)
      >125 mm/hr     -> 15-16 (extremely heavy / cloudburst)
    This mapping is a deliberate, documented modeling assumption — state it
    explicitly in your paper's methodology/limitations section.
    """
    if rain_mm_hr <= 0:
        return 2
    elif rain_mm_hr <= 2.5:
        return 4
    elif rain_mm_hr <= 7.5:
        return 6
    elif rain_mm_hr <= 35:
        return 9
    elif rain_mm_hr <= 125:
        return 13
    else:
        return 16


# ============================================================
# PREDICTION
# ============================================================
def predict_risk(factors: dict, model_choice: str):
    if model_choice == "RA (Ridge Regression)" and RA_LOADED:
        row = pd.DataFrame([{col: factors.get(col, 0) for col in ra_cols}])
        X_input = ra_scaler.transform(row) if ra_uses_scaling else row.values
        risk = float(ra_model.predict(X_input)[0])
    elif model_choice == "DL (Neural Network)" and DL_LOADED:
        row = pd.DataFrame([{col: factors.get(col, 0) for col in dl_cols}])
        X_input = dl_scaler.transform(row)
        risk = float(dl_model.predict(X_input, verbose=0)[0][0])
    else:
        return None
    return max(0.0, min(1.0, risk))


def trigger_alert_if_needed(area, risk_score, source_note=""):
    alert_triggered = risk_score >= RISK_THRESHOLD
    result = {
        "area": area, "risk_score": round(risk_score, 4), "threshold": RISK_THRESHOLD,
        "alert_triggered": alert_triggered, "timestamp": datetime.utcnow().isoformat(),
        "source": source_note,
    }
    if alert_triggered:
        subscribers = [s for s in load_json(DATA_FILE, []) if s["area"].lower() == area.lower()]
        message = (f"⚠ FLOOD ALERT — {area}: Predicted flood risk is HIGH "
                   f"({risk_score*100:.0f}%). Please move valuables to higher ground and "
                   f"stay updated with local authorities.")
        send_results = [{"to": s["phone"], "name": s["name"], **send_sms(s["phone"], message)}
                         for s in subscribers]
        result["alerts_sent"] = send_results
        result["message"] = message
        log = load_json(ALERT_LOG_FILE, [])
        log.append(result)
        save_json(ALERT_LOG_FILE, log)
    return result


def show_phone_notification(area, risk_score):
    """Animated phone-style SMS popup, matching the original Flask dashboard."""
    html = f"""
    <div id="phoneNotif" style="
        position: fixed; top: 20px; right: 24px; width: 340px;
        background: #ffffff; border-radius: 16px; padding: 14px 16px;
        box-shadow: 0 10px 40px rgba(0,0,0,0.25); z-index: 999;
        border: 1px solid #e5e7eb; font-family: 'Segoe UI', system-ui, sans-serif;
        animation: slideIn 0.45s cubic-bezier(.17,.89,.32,1.28);">
      <div style="display:flex; align-items:center; gap:8px; margin-bottom:4px;">
        <div style="width:26px;height:26px;border-radius:6px;background:#dc2626;
                    display:flex;align-items:center;justify-content:center;font-size:14px;">🌊</div>
        <div style="font-size:12px;color:#9ca3af;font-weight:600;flex:1;">MESSAGES</div>
        <div style="font-size:11px;color:#9ca3af;">now</div>
      </div>
      <div style="font-size:14px;font-weight:700;color:#111827;margin:2px 0;">⚠ Flood Alert — {area}</div>
      <div style="font-size:13px;color:#374151;line-height:1.4;">
        Predicted flood risk is HIGH ({risk_score*100:.0f}%). Move valuables to higher ground and stay updated.
      </div>
    </div>
    <style>
      @keyframes slideIn {{ from {{ top: -140px; }} to {{ top: 20px; }} }}
    </style>
    """
    components.html(html, height=160)


# ============================================================
# UI
# ============================================================
st.title("🌊 Flood Risk Early-Warning Broadcast System")
st.caption("Predicts regional flood risk and auto-alerts registered subscribers — "
           "no app-checking required. Now powering both the RA and DL course models.")

col_status1, col_status2, col_status3 = st.columns(3)
col_status1.metric("RA Model", "Loaded ✅" if RA_LOADED else "Not found ⚠️")
col_status2.metric("DL Model", "Loaded ✅" if DL_LOADED else "Not found ⚠️")
col_status3.metric("SMS Mode", "Live (Twilio)" if TWILIO_ENABLED else "Simulated")

st.divider()

tab1, tab2, tab3 = st.tabs(["📱 Subscribers", "🧪 Manual Check", "🌧️ Live Mumbai Weather Mode"])

# -------------------- TAB 1: Subscribers --------------------
with tab1:
    st.subheader("Register a Subscriber")
    c1, c2, c3 = st.columns(3)
    name = c1.text_input("Name", placeholder="Carmen")
    phone = c2.text_input("Phone (+countrycode)", placeholder="+919999999999")
    area = c3.text_input("Area", placeholder="Vikhroli East")
    if st.button("Register Subscriber"):
        if name and phone and area:
            subs = load_json(DATA_FILE, [])
            subs.append({"name": name, "phone": phone, "area": area,
                         "registered_at": datetime.utcnow().isoformat()})
            save_json(DATA_FILE, subs)
            st.success(f"Registered {name} for alerts in {area}.")
        else:
            st.error("Name, phone, and area are all required.")

    st.subheader("Registered Subscribers")
    subs = load_json(DATA_FILE, [])
    if subs:
        st.dataframe(pd.DataFrame(subs), use_container_width=True)
    else:
        st.info("No subscribers registered yet.")

# -------------------- TAB 2: Manual Check --------------------
with tab2:
    st.subheader("Simulate Regional Readings")
    model_choice = st.radio("Model to use", ["RA (Ridge Regression)", "DL (Neural Network)"],
                             horizontal=True, key="manual_model")
    manual_area = st.text_input("Area to score", value="Vikhroli East", key="manual_area")

    preset = st.selectbox("Quick preset", ["Custom", "Normal conditions", "Extreme values (demo)"])
    if preset == "Extreme values (demo)":
        defaults = {k: 14 for k in FEATURE_COLUMNS}
    elif preset == "Normal conditions":
        defaults = {k: 4 for k in FEATURE_COLUMNS}
    else:
        defaults = MUMBAI_BASELINE

    cols_per_row = 2
    factors = {}
    feature_items = list(FEATURE_INFO.items())
    for i in range(0, len(feature_items), cols_per_row):
        row_cols = st.columns(cols_per_row)
        for j, (feat, desc) in enumerate(feature_items[i:i+cols_per_row]):
            with row_cols[j]:
                factors[feat] = st.slider(f"{feat}", 0, 16, defaults.get(feat, 4), help=desc)

    if st.button("Score Region & Check Alert", type="primary"):
        risk = predict_risk(factors, model_choice)
        if risk is None:
            st.error(f"{model_choice} isn't loaded — check model_artifacts folder(s) are present.")
        else:
            result = trigger_alert_if_needed(manual_area, risk, source_note=f"manual/{model_choice}")
            if result["alert_triggered"]:
                st.error(f"🚨 ALERT TRIGGERED — Risk Score: {risk:.2%} (threshold {RISK_THRESHOLD:.0%})")
                show_phone_notification(manual_area, risk)
            else:
                st.success(f"✅ Safe — Risk Score: {risk:.2%} (threshold {RISK_THRESHOLD:.0%})")

# -------------------- TAB 3: Live Weather Mode --------------------
with tab3:
    st.subheader("🌧️ Live Mumbai Weather Mode")
    st.caption(
        "Pulls REAL current rainfall data for Mumbai from Open-Meteo (free, no API key) and "
        "auto-scores flood risk. MonsoonIntensity is derived live from actual rainfall; all "
        "other factors use a fixed Mumbai infrastructure baseline (editable below). This is "
        "what makes the alert fire based on genuine current weather, not manual demo input."
    )
    lw_model_choice = st.radio("Model to use", ["RA (Ridge Regression)", "DL (Neural Network)"],
                                horizontal=True, key="live_model")
    lw_area = st.text_input("Area to monitor", value="Vikhroli East", key="live_area")

    with st.expander("Edit static Mumbai baseline factors (optional)"):
        live_baseline = {}
        feature_items2 = [(k, v) for k, v in FEATURE_INFO.items() if k != "MonsoonIntensity"]
        for i in range(0, len(feature_items2), 2):
            row_cols = st.columns(2)
            for j, (feat, desc) in enumerate(feature_items2[i:i+2]):
                with row_cols[j]:
                    live_baseline[feat] = st.slider(f"{feat}", 0, 16, MUMBAI_BASELINE[feat],
                                                      help=desc, key=f"live_{feat}")

    if st.button("🔄 Fetch Live Weather & Check Risk Now", type="primary"):
        try:
            weather = fetch_mumbai_weather()
            current = weather.get("current", {})
            rain_now = current.get("rain", 0) or current.get("precipitation", 0) or 0
            monsoon_intensity = rainfall_mm_to_monsoon_intensity(rain_now)

            st.metric("Current Mumbai rainfall", f"{rain_now} mm/hr")
            st.metric("Mapped MonsoonIntensity", monsoon_intensity)

            live_factors = dict(live_baseline) if 'live_baseline' in dir() else dict(MUMBAI_BASELINE)
            live_factors["MonsoonIntensity"] = monsoon_intensity

            risk = predict_risk(live_factors, lw_model_choice)
            if risk is None:
                st.error(f"{lw_model_choice} isn't loaded.")
            else:
                result = trigger_alert_if_needed(lw_area, risk, source_note=f"live-weather/{lw_model_choice}")
                if result["alert_triggered"]:
                    st.error(f"🚨 REAL ALERT TRIGGERED from live weather — Risk: {risk:.2%}")
                    show_phone_notification(lw_area, risk)
                else:
                    st.success(f"✅ Currently safe based on live weather — Risk: {risk:.2%}")
        except Exception as e:
            st.error(f"Could not fetch live weather data: {e}")

    st.caption("Tip: during Mumbai's monsoon season (Jun–Sep), run this during actual rainy "
               "weather for a live, non-staged demo in your presentation.")

# -------------------- Alert Log --------------------
st.divider()
st.subheader("📋 Alert Log")
log = load_json(ALERT_LOG_FILE, [])
if log:
    st.dataframe(pd.DataFrame(log)[["timestamp", "area", "risk_score", "source"]].iloc[::-1],
                 use_container_width=True)
else:
    st.info("No alerts triggered yet.")
