"""
Phase 17-18 - Streamlit Dashboard.

Pages: Home, Live AQI, Forecast, EDA, Model Explainability, About.
Reads from the FastAPI service for live/predicted numbers, and from
the local CSV fallback for current AQI and historical charts + EDA.

Run:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(PROJECT_ROOT / "feature_pipeline"))
sys.path.insert(0, str(PROJECT_ROOT / "training_pipeline"))
API_BASE_URL = st.secrets.get("API_BASE_URL", os.environ.get("API_BASE_URL", "")).rstrip("/")

st.set_page_config(page_title="Pearls AQI Predictor", layout="wide")

st.markdown("""
<style>
    .block-container { padding-top: 2rem; padding-bottom: 3rem; }
    [data-testid="stMetric"] { background: #f4f7f2; border-left: 4px solid #2f855a; padding: 0.8rem; }
    h1, h2, h3 { color: #173b35; }
</style>
""", unsafe_allow_html=True)


def aqi_alert_level(aqi):
    if aqi is None:
        return "Unknown", "gray"
    if aqi > 300:
        return "Hazardous", "darkred"
    if aqi > 200:
        return "Very Unhealthy - Red Alert", "red"
    if aqi > 150:
        return "Unhealthy - Orange Alert", "orange"
    if aqi > 100:
        return "Unhealthy for Sensitive Groups", "gold"
    if aqi > 50:
        return "Moderate", "yellowgreen"
    return "Good", "green"


@st.cache_data(ttl=3600)
def load_features():
    for filename in ("features_all_cities.csv", "live_buffer.csv"):
        path = PROJECT_ROOT / "data" / filename
        if path.exists():
            return pd.read_csv(path, parse_dates=["time"])
    return pd.DataFrame()


def data_source():
    if (PROJECT_ROOT / "data" / "features_all_cities.csv").exists():
        return "Local historical features"
    if (PROJECT_ROOT / "data" / "live_buffer.csv").exists():
        return "Local tracked live buffer"
    return "Unavailable"


@st.cache_resource
def load_local_models():
    import joblib

    model_paths = {
        "24h": PROJECT_ROOT / "models" / "target_aqi_24h_delta_xgb.pkl",
        "48h": PROJECT_ROOT / "models" / "target_aqi_48h_delta_xgb.pkl",
        "72h": PROJECT_ROOT / "models" / "target_aqi_72h_delta_xgb.pkl",
    }
    if not all(path.exists() for path in model_paths.values()):
        return None
    return {horizon: joblib.load(path) for horizon, path in model_paths.items()}


@st.cache_data(ttl=3600)
def predict_locally(features):
    from inference_features import build_prediction_features

    current_rows = features.sort_values("time").groupby("city", as_index=False).tail(1)
    model_input, cities_order = build_prediction_features(features, current_rows)
    models = load_local_models()
    if models is None:
        return None

    predictions = {city: {} for city in cities_order}
    for horizon, model in models.items():
        values = model.predict(model_input)
        for city, value in zip(cities_order, values):
            predictions[city][horizon] = round(float(max(value, 0)), 1)
    return predictions

st.sidebar.title("Pearls AQI Predictor")
page = st.sidebar.radio("Navigate", ["Home", "Live AQI", "Forecast", "EDA", "Model Explainability", "About"])

df = load_features()
cities = sorted(df["city"].unique()) if not df.empty else []

if page == "Home":
    st.title("Pearls AQI Predictor")
    st.write("Serverless AQI forecasting for major Pakistani cities, with a feature store, model registry, and SHAP-based explainability.")
    if cities:
        latest = df.sort_values("time").iloc[-1]
        metric_columns = st.columns(4)
        metric_columns[0].metric("Cities tracked", len(cities))
        metric_columns[1].metric("Feature rows", f"{len(df):,}")
        metric_columns[2].metric("Latest AQI", f"{latest['aqi']:.0f}")
        metric_columns[3].metric("Temperature", f"{latest['temperature_2m']:.1f} C")
        st.caption(f"Data source: {data_source()} | Latest record: {latest['time']:%Y-%m-%d %H:%M}")

        overview = (df.sort_values("time").groupby("city", as_index=False).tail(1)
                    .sort_values("aqi", ascending=False))
        fig = px.bar(overview, x="city", y="aqi", color="aqi", color_continuous_scale="YlOrRd",
                     title="Latest AQI by city", labels={"aqi": "AQI", "city": ""})
        fig.update_layout(coloraxis_showscale=False, height=380)
        st.plotly_chart(fig, use_container_width=True)

elif page == "Live AQI":
    st.title("Live AQI")
    if not cities:
        st.info("No feature data found yet. Run the backfill and feature pipeline first.")
    else:
        selected_city = st.selectbox("City", cities)
        city_rows = df[df.city == selected_city].sort_values("time")
        try:
            if API_BASE_URL:
                resp = requests.get(f"{API_BASE_URL}/predict/{selected_city}", timeout=5)
                resp.raise_for_status()
            current_aqi = city_rows.iloc[-1]["aqi"]
        except Exception:
            current_aqi = df[df.city == selected_city].sort_values("time").iloc[-1]["aqi"] if not df.empty else None

        level, color = aqi_alert_level(current_aqi)
        current_temperature = city_rows.iloc[-1].get("temperature_2m")
        cards = st.columns(3)
        cards[0].metric("Current AQI", f"{current_aqi:.0f}" if pd.notna(current_aqi) else "N/A")
        cards[1].metric("Temperature", f"{current_temperature:.1f} C" if pd.notna(current_temperature) else "N/A")
        cards[2].metric("Status", level)
        st.markdown(f"<span style='color:{color}; font-weight:bold'>{level}</span>", unsafe_allow_html=True)

        chart_rows = city_rows.tail(168)
        fig = px.line(chart_rows, x="time", y=["aqi", "temperature_2m"],
                  title=f"AQI and temperature - {selected_city}",
                  labels={"value": "Reading", "variable": "Metric", "time": ""})
        fig.update_layout(height=380, legend_title_text="")
        st.plotly_chart(fig, use_container_width=True)

elif page == "Forecast":
    st.title("3-Day Forecast")
    if not cities:
        st.info("No feature data found yet.")
    else:
        selected_city = st.selectbox("City", cities, key="forecast_city")
        try:
            local_forecasts = predict_locally(df)
        except Exception as exc:
            local_forecasts = None
            st.warning(f"Local model prediction failed: {exc}")

        if local_forecasts and selected_city in local_forecasts:
            forecast = local_forecasts[selected_city]
            st.caption("Prediction source: trained XGBoost models")
            fc_df = pd.DataFrame({"Horizon": list(forecast.keys()), "Predicted AQI": list(forecast.values())})
            fig = px.bar(fc_df, x="Horizon", y="Predicted AQI", color="Predicted AQI",
                         color_continuous_scale="YlOrRd", title=f"Model forecast for {selected_city}")
            fig.update_layout(coloraxis_showscale=False, height=380)
            st.plotly_chart(fig, use_container_width=True)
        elif not API_BASE_URL:
            st.info("Trained model files are not available yet. Run the training pipeline and publish the model files.")
        else:
            try:
                resp = requests.get(f"{API_BASE_URL}/predict/{selected_city}", timeout=5)
                resp.raise_for_status()
                forecast = resp.json()["forecast"]
                fc_df = pd.DataFrame({"Horizon": list(forecast.keys()), "Predicted AQI": list(forecast.values())})
                fig = px.bar(fc_df, x="Horizon", y="Predicted AQI", title=f"Forecast for {selected_city}")
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not reach prediction API: {e}")

elif page == "EDA":
    st.title("Exploratory Data Analysis")
    if df.empty:
        st.info("No feature data found yet.")
    else:
        selected_city = st.selectbox("City", cities, key="eda_city")
        city_df = df[df.city == selected_city]
        fig = px.line(city_df, x="time", y="aqi", title=f"AQI over time - {selected_city}")
        st.plotly_chart(fig, use_container_width=True)

        fig2 = px.box(df, x="city", y="aqi", title="AQI distribution by city")
        st.plotly_chart(fig2, use_container_width=True)

        temp_fig = px.scatter(df.sample(min(len(df), 3000), random_state=42), x="temperature_2m", y="aqi",
                      color="city", opacity=0.65, title="Temperature and AQI relationship",
                      labels={"temperature_2m": "Temperature (C)", "aqi": "AQI"})
        st.plotly_chart(temp_fig, use_container_width=True)

elif page == "Model Explainability":
    st.title("Model Explainability (SHAP)")
    st.write("Feature contribution plots generated by the training pipeline.")
    for horizon in ["24h", "48h", "72h"]:
        img_path = PROJECT_ROOT / "models" / f"shap_target_aqi_{horizon}.png"
        try:
            st.image(img_path, caption=f"SHAP summary - {horizon} forecast")
        except Exception:
            st.info(f"No SHAP plot found yet for {horizon}. Train the model first.")

elif page == "About":
    st.title("About this project")
    st.write("""
    Pearls AQI Predictor is an end-to-end, serverless ML system forecasting AQI
    24h / 48h / 72h ahead for 10 major Pakistani cities.

    Pipeline: Open-Meteo (historical + backfill) -> feature engineering ->
    Hopsworks feature store -> LightGBM/XGBoost/RandomForest model comparison ->
    Hopsworks model registry -> FastAPI prediction service -> this Streamlit dashboard.

    Automation: GitHub Actions runs the hourly feature pipeline and a daily retrain.
    """)
