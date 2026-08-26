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
import pydeck as pdk
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
    :root {
        --ink: #eef6f4;
        --muted: #a6bab6;
        --glass: rgba(255, 255, 255, 0.075);
        --glass-strong: rgba(255, 255, 255, 0.12);
        --border: rgba(255, 255, 255, 0.14);
        --accent: #65e6b0;
        --accent-warm: #ffc875;
    }
    .stApp {
        color: var(--ink);
        background: #1e2029;
        background-image: radial-gradient(circle at 82% 8%, rgba(255, 200, 117, 0.13), transparent 28%),
                          radial-gradient(circle at 8% 84%, rgba(101, 230, 176, 0.10), transparent 30%);
        overflow-x: hidden;
    }
    .stApp::before, .stApp::after {
        content: "";
        position: fixed;
        inset: -35%;
        pointer-events: none;
        z-index: 0;
    }
    .stApp::before {
        background: radial-gradient(ellipse at 30% 20%, rgba(255, 203, 132, 0.16), transparent 23%),
                    radial-gradient(ellipse at 75% 70%, rgba(103, 232, 181, 0.10), transparent 24%);
        filter: blur(38px);
        animation: sunlight-drift 18s ease-in-out infinite alternate;
    }
    .stApp::after {
        background: repeating-linear-gradient(118deg, transparent 0 90px, rgba(255,255,255,0.025) 110px 155px, transparent 180px 290px);
        filter: blur(18px);
        opacity: 0.65;
        animation: shadow-pan 17s linear infinite;
    }
    @keyframes sunlight-drift {
        from { transform: translate3d(-4%, -2%, 0) rotate(-2deg); }
        to { transform: translate3d(5%, 3%, 0) rotate(3deg); }
    }
    @keyframes shadow-pan {
        from { transform: translate3d(-7%, -2%, 0); }
        to { transform: translate3d(7%, 2%, 0); }
    }
    .block-container { position: relative; z-index: 1; padding-top: 2.5rem; padding-bottom: 3.5rem; }
    h1, h2, h3, p, label, [data-testid="stCaptionContainer"] { color: var(--ink); }
    h1 { letter-spacing: 0.01em; }
    [data-testid="stSidebar"] {
        display: none;
    }
    [data-testid="stTabs"] [role="tablist"] {
        gap: 0.5rem;
        border-bottom: 1px solid rgba(255,255,255,0.12);
        padding: 0.35rem;
        background: rgba(255,255,255,0.05);
        border: 1px solid var(--border);
        border-radius: 999px;
        backdrop-filter: blur(12px);
    }
    [data-testid="stTabs"] button[role="tab"] {
        border: 1px solid transparent;
        border-radius: 999px;
        color: var(--muted);
        padding: 0.55rem 1rem;
        transition: all 0.3s ease;
    }
    [data-testid="stTabs"] button[role="tab"]:hover,
    [data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
        color: #1e2029;
        background: var(--accent-warm);
        border-color: rgba(252,191,134,0.8);
        box-shadow: 0 0 24px rgba(252,191,134,0.25);
    }
    [data-testid="stMetric"], [data-testid="stAlert"], [data-testid="stExpander"] {
        background: var(--glass);
        border: 1px solid var(--border);
        border-radius: 16px;
        box-shadow: 0 16px 40px rgba(0, 0, 0, 0.20), inset 0 1px 0 rgba(255,255,255,0.08);
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
    }
    [data-testid="stMetric"] { padding: 1rem; }
    [data-testid="stMetricLabel"] p { color: var(--muted) !important; font-weight: 700; }
    [data-testid="stMetricValue"] { color: var(--ink) !important; }
    [data-testid="stMetricDelta"] { color: var(--accent) !important; }
    [data-testid="stSelectbox"] > div > div {
        background: rgba(15, 23, 42, 0.72);
        border: 1px solid var(--border);
        border-radius: 12px;
        color: var(--ink);
        transition: all 0.3s ease;
    }
    [data-testid="stSelectbox"] > div > div:hover { border-color: var(--accent); box-shadow: 0 0 18px rgba(101, 230, 176, 0.12); }
    [data-testid="stPlotlyChart"] {
        background: var(--glass);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 0.4rem;
        box-shadow: 0 16px 40px rgba(0, 0, 0, 0.18);
    }
    .aqi-legend {
        margin-top: 1rem;
        padding: 1rem;
        border: 1px solid rgba(255, 255, 255, 0.15);
        border-radius: 12px;
        background: rgba(255, 255, 255, 0.08);
        box-shadow: 0 16px 40px rgba(0, 0, 0, 0.2), inset 0 1px 0 rgba(255,255,255,0.08);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
    }
    .legend-title { color: #eef6f4; font-size: 0.9rem; font-weight: 800; margin-bottom: 0.7rem; }
    .legend-bar {
        height: 11px;
        border-radius: 999px;
        background: linear-gradient(90deg, #8D99AE 0 20%, #BBB2C9 20% 40%, #EBDAEE 40% 60%, #FEE3CE 60% 80%, #FCBF86 80% 100%);
    }
    .legend-labels { display: flex; justify-content: space-between; gap: 0.25rem; margin-top: 0.45rem; color: #cbd8d5; font-size: 0.62rem; }
    .legend-note { color: #a6bab6; font-size: 0.72rem; line-height: 1.3; margin-top: 1rem; }
    [data-testid="stCaptionContainer"] { color: var(--muted) !important; }
</style>
""", unsafe_allow_html=True)


def style_plot(fig, height=380):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#eef6f4"},
        title_font={"color": "#eef6f4", "size": 18},
        height=height,
        margin={"l": 24, "r": 24, "t": 56, "b": 24},
        legend={"font": {"color": "#cbd8d5"}},
        xaxis={"gridcolor": "rgba(255,255,255,0.08)", "zerolinecolor": "rgba(255,255,255,0.12)"},
        yaxis={"gridcolor": "rgba(255,255,255,0.08)", "zerolinecolor": "rgba(255,255,255,0.12)"},
    )
    return fig


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


CITY_COORDINATES = {
    "Faisalabad": (31.4504, 73.1350),
    "Hyderabad": (25.3960, 68.3578),
    "Islamabad": (33.6844, 73.0479),
    "Karachi": (24.8607, 67.0011),
    "Lahore": (31.5204, 74.3587),
    "Multan": (30.1575, 71.5249),
    "Quetta": (30.1798, 66.9750),
    "Rawalpindi": (33.5651, 73.0169),
    "Sialkot": (32.4945, 74.5229),
}

AQI_COLORS = {
    "Good": [141, 153, 174, 180],
    "Moderate": [186, 178, 201, 180],
    "Unhealthy for Sensitive Groups": [235, 218, 238, 180],
    "Unhealthy": [254, 227, 206, 180],
    "Hazardous": [252, 191, 134, 180],
}


def aqi_map_color(aqi):
    if aqi <= 50:
        return AQI_COLORS["Good"]
    if aqi <= 100:
        return AQI_COLORS["Moderate"]
    if aqi <= 150:
        return AQI_COLORS["Unhealthy for Sensitive Groups"]
    if aqi <= 200:
        return AQI_COLORS["Unhealthy"]
    return AQI_COLORS["Hazardous"]


def render_city_map(latest_by_city, selected_city=None):
    map_data = latest_by_city.copy()
    map_data["latitude"] = map_data["city"].map(lambda city: CITY_COORDINATES[city][0])
    map_data["longitude"] = map_data["city"].map(lambda city: CITY_COORDINATES[city][1])
    map_data["color"] = map_data["aqi"].apply(aqi_map_color)
    map_data["radius"] = 5000
    halos = map_data.copy()
    halos["color"] = halos["city"].eq(selected_city).map(
        {True: [252, 191, 134, 90], False: [255, 255, 255, 35]}
    )
    halos["radius"] = halos["city"].eq(selected_city).map({True: 18000, False: 9000})

    layers = [
        pdk.Layer(
            "ScatterplotLayer", data=map_data, get_position="[longitude, latitude]",
            get_fill_color="color", get_radius="radius", pickable=True,
            stroked=True, get_line_color=[255, 255, 255, 110], line_width_min_pixels=1,
        ),
        pdk.Layer(
            "ScatterplotLayer", data=halos, get_position="[longitude, latitude]",
            get_fill_color="color", get_radius="radius", stroked=False,
        ),
    ]
    if selected_city in CITY_COORDINATES:
        latitude, longitude, zoom = (*CITY_COORDINATES[selected_city], 10.5)
    else:
        latitude, longitude, zoom = 30.3753, 69.3451, 5.5
    deck = pdk.Deck(
        layers=layers,
        initial_view_state=pdk.ViewState(latitude=latitude, longitude=longitude, zoom=zoom, pitch=0),
        map_style="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
        tooltip={"html": "<b>{city}</b><br/>AQI: {aqi}", "style": {"color": "white"}},
    )
    map_column, legend_column = st.columns([3.4, 1])
    with map_column:
        st.pydeck_chart(deck, use_container_width=True)
    with legend_column:
        st.markdown("""
        <div class="aqi-legend">
            <div class="legend-title">AQI scale</div>
            <div class="legend-bar"></div>
            <div class="legend-labels"><span>0-50</span><span>51-100</span><span>101-150</span><span>151-200</span><span>201+</span></div>
            <div class="legend-note">Selected city is highlighted</div>
        </div>
        """, unsafe_allow_html=True)


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

df = load_features()
cities = sorted(df["city"].unique()) if not df.empty else []

tabs = st.tabs(["Home", "Live AQI", "Forecast", "EDA", "Model Explainability", "About"])

with tabs[0]:
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
        fig.update_layout(coloraxis_showscale=False)
        style_plot(fig)
        st.plotly_chart(fig, use_container_width=True)
        st.subheader("Pakistan air-quality map")
        st.caption("Select a city on the Live AQI tab to focus the map.")
        render_city_map(overview, None)

with tabs[1]:
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

        latest_by_city = df.sort_values("time").groupby("city", as_index=False).tail(1)
        render_city_map(latest_by_city, selected_city)

        chart_rows = city_rows.tail(168)
        fig = px.line(chart_rows, x="time", y=["aqi", "temperature_2m"],
                  title=f"AQI and temperature - {selected_city}",
                  labels={"value": "Reading", "variable": "Metric", "time": ""})
        fig.update_layout(legend_title_text="")
        style_plot(fig)
        st.plotly_chart(fig, use_container_width=True)

with tabs[2]:
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
            fig.update_layout(coloraxis_showscale=False)
            style_plot(fig)
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
                style_plot(fig)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not reach prediction API: {e}")

with tabs[3]:
    st.title("Exploratory Data Analysis")
    if df.empty:
        st.info("No feature data found yet.")
    else:
        selected_city = st.selectbox("City", cities, key="eda_city")
        city_df = df[df.city == selected_city]
        fig = px.line(city_df, x="time", y="aqi", title=f"AQI over time - {selected_city}")
        style_plot(fig)
        st.plotly_chart(fig, use_container_width=True)

        fig2 = px.box(df, x="city", y="aqi", title="AQI distribution by city")
        style_plot(fig2)
        st.plotly_chart(fig2, use_container_width=True)

        temp_fig = px.scatter(df.sample(min(len(df), 3000), random_state=42), x="temperature_2m", y="aqi",
                      color="city", opacity=0.65, title="Temperature and AQI relationship",
                      labels={"temperature_2m": "Temperature (C)", "aqi": "AQI"})
        style_plot(temp_fig)
        st.plotly_chart(temp_fig, use_container_width=True)

with tabs[4]:
    st.title("Model Explainability (SHAP)")
    st.write("Feature contribution plots generated by the training pipeline.")
    for horizon in ["24h", "48h", "72h"]:
        img_path = PROJECT_ROOT / "models" / f"shap_target_aqi_{horizon}.png"
        try:
            st.image(img_path, caption=f"SHAP summary - {horizon} forecast")
        except Exception:
            st.info(f"No SHAP plot found yet for {horizon}. Train the model first.")

with tabs[5]:
    st.title("About this project")
    st.write("""
    Pearls AQI Predictor is an end-to-end, serverless ML system forecasting AQI
    24h / 48h / 72h ahead for 10 major Pakistani cities.

    Pipeline: Open-Meteo (historical + backfill) -> feature engineering ->
    Hopsworks feature store -> LightGBM/XGBoost/RandomForest model comparison ->
    Hopsworks model registry -> FastAPI prediction service -> this Streamlit dashboard.

    Automation: GitHub Actions runs the hourly feature pipeline and a daily retrain.
    """)
