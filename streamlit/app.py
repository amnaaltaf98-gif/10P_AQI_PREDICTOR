"""
Phase 17-18 - Streamlit Dashboard.

Pages: Home, Live AQI, Forecast, EDA, Model Explainability, About.

Reads from the FastAPI service for live/predicted numbers, and from
data/features_all_cities.csv for historical charts + EDA.

Run locally:

    streamlit run app.py
"""

from pathlib import Path
import os

import streamlit as st
import pandas as pd
import requests
import plotly.express as px


# ---------------------------------------------------------
# Deployment-safe paths
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "features_all_cities.csv"
MODELS_DIR = BASE_DIR / "models"


# ---------------------------------------------------------
# API URL
#
# Local:
# http://localhost:8000
#
# Streamlit Cloud:
# Set API_BASE_URL in Streamlit Secrets
# ---------------------------------------------------------

try:
    API_BASE_URL = st.secrets["API_BASE_URL"]
except Exception:
    API_BASE_URL = os.getenv(
        "API_BASE_URL",
        "http://localhost:8000"
    )

API_BASE_URL = API_BASE_URL.rstrip("/")


st.set_page_config(
    page_title="Pearls AQI Predictor",
    layout="wide"
)


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

    try:
        df = pd.read_csv(
            DATA_PATH,
            parse_dates=["time"]
        )

        return df

    except FileNotFoundError:
        return pd.DataFrame()


st.sidebar.title("Pearls AQI Predictor")

page = st.sidebar.radio(
    "Navigate",
    [
        "Home",
        "Live AQI",
        "Forecast",
        "EDA",
        "Model Explainability",
        "About"
    ]
)


df = load_features()

cities = (
    sorted(df["city"].unique())
    if not df.empty
    else []
)


# =========================================================
# HOME
# =========================================================

if page == "Home":

    st.title("Pearls AQI Predictor")

    st.write(
        "Serverless AQI forecasting for major Pakistani cities, "
        "with a feature store, model registry, and SHAP-based "
        "explainability."
    )

    if cities:

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Cities tracked",
                len(cities)
            )

        with col2:
            st.metric(
                "Feature rows",
                len(df)
            )


# =========================================================
# LIVE AQI
# =========================================================

elif page == "Live AQI":

    st.title("Live AQI")

    if not cities:

        st.info(
            "No feature data found yet. "
            "Run the backfill and feature pipeline first."
        )

    else:

        selected_city = st.selectbox(
            "City",
            cities
        )

        try:

            resp = requests.get(
                f"{API_BASE_URL}/predict/{selected_city}",
                timeout=30
            )

            resp.raise_for_status()

            api_data = resp.json()

            current_aqi = api_data.get(
                "current_aqi"
            )

        except Exception:

            current_aqi = (
                df[df.city == selected_city]
                .sort_values("time")
                .iloc[-1]["aqi"]
                if not df.empty
                else None
            )

        level, color = aqi_alert_level(
            current_aqi
        )

        st.markdown(
            f"### Current AQI: **{current_aqi:.0f}**"
            if current_aqi is not None
            else "### Current AQI: N/A"
        )

        st.markdown(
            f"<span style='color:{color}; "
            f"font-weight:bold'>{level}</span>",
            unsafe_allow_html=True
        )


# =========================================================
# FORECAST
# =========================================================

elif page == "Forecast":

    st.title("3-Day Forecast")

    if not cities:

        st.info(
            "No feature data found yet."
        )

    else:

        selected_city = st.selectbox(
            "City",
            cities,
            key="forecast_city"
        )

        try:

            resp = requests.get(
                f"{API_BASE_URL}/predict/{selected_city}",
                timeout=60
            )

            resp.raise_for_status()

            forecast = resp.json()["forecast"]

            fc_df = pd.DataFrame({
                "Horizon": list(forecast.keys()),
                "Predicted AQI": list(forecast.values())
            })

            fig = px.bar(
                fc_df,
                x="Horizon",
                y="Predicted AQI",
                title=f"Forecast for {selected_city}"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        except Exception as e:

            st.warning(
                f"Could not reach prediction API: {e}"
            )


# =========================================================
# EDA
# =========================================================

elif page == "EDA":

    st.title("Exploratory Data Analysis")

    if df.empty:

        st.info(
            "No feature data found yet."
        )

    else:

        selected_city = st.selectbox(
            "City",
            cities,
            key="eda_city"
        )

        city_df = df[
            df.city == selected_city
        ]

        fig = px.line(
            city_df,
            x="time",
            y="aqi",
            title=f"AQI over time - {selected_city}"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        fig2 = px.box(
            df,
            x="city",
            y="aqi",
            title="AQI distribution by city"
        )

        st.plotly_chart(
            fig2,
            use_container_width=True
        )


# =========================================================
# MODEL EXPLAINABILITY
# =========================================================

elif page == "Model Explainability":

    st.title("Model Explainability (SHAP)")

    st.write(
        "Feature importance plots generated during training. "
        "Run training_pipeline/train.py to (re)generate these."
    )

    for horizon in ["24h", "48h", "72h"]:

        img_path = (
            MODELS_DIR /
            f"shap_target_aqi_{horizon}.png"
        )

        if img_path.exists():

            st.image(
                str(img_path),
                caption=(
                    f"SHAP summary - "
                    f"{horizon} forecast"
                )
            )

        else:

            st.info(
                f"No SHAP plot found yet for "
                f"{horizon}. Train the model first."
            )


# =========================================================
# ABOUT
# =========================================================

elif page == "About":

    st.title("About this project")

    st.write("""

    Pearls AQI Predictor is an end-to-end, serverless ML system forecasting AQI
    24h / 48h / 72h ahead for major Pakistani cities.

    Pipeline: Open-Meteo (historical + backfill) -> feature engineering ->
    Hopsworks feature store -> LightGBM/XGBoost/RandomForest model comparison ->
    Hopsworks model registry -> FastAPI prediction service -> this Streamlit dashboard.

    Automation: GitHub Actions runs the hourly feature pipeline and a daily retrain.

    """)