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

st.set_page_config(page_title="Pearls AQI Predictor", layout="wide", initial_sidebar_state="collapsed")

# ---------------------------------------------------------------------------
# THEME
# Palette kept from the original build: deep navy base, mint accent, warm
# amber accent, frosted-glass panels. Values below just push translucency
# and contrast further so panels read as "glass over a map" rather than
# solid cards, and add a couple of missing states (buttons, tabs, sidebar
# kill-switch) that were leaking default Streamlit styling before.
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    :root {
        --ink: #eef6f4;
        --muted: #a6bab6;
        --glass: rgba(255, 255, 255, 0.055);
        --glass-strong: rgba(255, 255, 255, 0.10);
        --glass-hover: rgba(255, 255, 255, 0.14);
        --border: rgba(255, 255, 255, 0.14);
        --accent: #65e6b0;
        --accent-warm: #ffc875;
        --navy: #1e2029;
    }

    /* ---- kill Streamlit's default chrome so tabs are the only nav ---- */
    [data-testid="stSidebar"],
    [data-testid="stSidebarNav"],
    [data-testid="collapsedControl"],
    header[data-testid="stHeader"] { display: none !important; }

    .stApp {
        color: var(--ink);
        background: var(--navy);
        background-image:
            radial-gradient(circle at 82% 8%, rgba(255, 200, 117, 0.14), transparent 30%),
            radial-gradient(circle at 8% 84%, rgba(101, 230, 176, 0.11), transparent 32%),
            radial-gradient(circle at 50% 50%, rgba(101, 230, 176, 0.03), transparent 60%);
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

    .block-container { position: relative; z-index: 1; padding-top: 2rem; padding-bottom: 3.5rem; max-width: 1200px; }
    h1, h2, h3, p, label, [data-testid="stCaptionContainer"] { color: var(--ink); }
    h1 { letter-spacing: 0.01em; font-weight: 800; }

    /* ---- horizontal pill tabs (this IS the nav, no sidebar) ---- */
    [data-testid="stTabs"] [role="tablist"] {
        gap: 0.4rem;
        border-bottom: none;
        padding: 0.4rem;
        background: var(--glass);
        border: 1px solid var(--border);
        border-radius: 999px;
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        flex-wrap: wrap;
    }
    [data-testid="stTabs"] button[role="tab"] {
        border: 1px solid transparent;
        border-radius: 999px;
        color: var(--muted);
        padding: 0.55rem 1.1rem;
        font-weight: 600;
        transition: all 0.25s ease;
        background: transparent;
    }
    [data-testid="stTabs"] button[role="tab"]:hover {
        color: var(--ink);
        background: rgba(255,255,255,0.08);
    }
    [data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
        color: #1e2029;
        background: var(--accent-warm);
        border-color: rgba(252,191,134,0.8);
        box-shadow: 0 0 24px rgba(252,191,134,0.28);
    }

    /* ---- glass panels ---- */
    [data-testid="stMetric"], [data-testid="stAlert"], [data-testid="stExpander"] {
        background: var(--glass);
        border: 1px solid var(--border);
        border-radius: 16px;
        box-shadow: 0 16px 40px rgba(0, 0, 0, 0.20), inset 0 1px 0 rgba(255,255,255,0.06);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
    }
    [data-testid="stMetric"] { padding: 1rem; }
    [data-testid="stMetricLabel"] p { color: var(--muted) !important; font-weight: 700; }
    [data-testid="stMetricValue"] { color: var(--ink) !important; }
    [data-testid="stMetricDelta"] { color: var(--accent) !important; }

    /* ---- translucent inputs ---- */
    [data-testid="stSelectbox"] > div > div {
        background: rgba(255, 255, 255, 0.06) !important;
        border: 1px solid var(--border);
        border-radius: 12px;
        color: var(--ink);
        backdrop-filter: blur(10px);
        transition: all 0.25s ease;
    }
    [data-testid="stSelectbox"] > div > div:hover {
        border-color: var(--accent);
        background: rgba(255,255,255,0.09) !important;
        box-shadow: 0 0 18px rgba(101, 230, 176, 0.15);
    }

    /* ---- translucent buttons ---- */
    .stButton > button, .stDownloadButton > button {
        background: var(--glass-strong) !important;
        border: 1px solid var(--border) !important;
        color: var(--ink) !important;
        border-radius: 12px !important;
        backdrop-filter: blur(10px);
        transition: all 0.25s ease;
    }
    .stButton > button:hover, .stDownloadButton > button:hover {
        background: var(--glass-hover) !important;
        border-color: var(--accent) !important;
        box-shadow: 0 0 18px rgba(101, 230, 176, 0.18);
    }

    [data-testid="stPlotlyChart"] {
        background: var(--glass);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 0.4rem;
        box-shadow: 0 16px 40px rgba(0, 0, 0, 0.18);
        backdrop-filter: blur(10px);
    }

    /* ---- hero map wrapper: pulls the following row of glass cards up
       over its own bottom edge so they read as "floating on the map" ---- */
    .hero-map-wrap {
        border-radius: 20px;
        overflow: hidden;
        border: 1px solid var(--border);
        box-shadow: 0 20px 60px rgba(0,0,0,0.35);
        margin-bottom: -3.2rem;
        position: relative;
        z-index: 1;
    }
    .hero-card-row { position: relative; z-index: 2; margin-top: 1.2rem; }

    .aqi-legend {
        padding: 1rem;
        border: 1px solid var(--border);
        border-radius: 12px;
        background: var(--glass);
        box-shadow: 0 16px 40px rgba(0, 0, 0, 0.2), inset 0 1px 0 rgba(255,255,255,0.06);
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
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
    if aqi is None or pd.isna(aqi):
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


# 9 cities - keep this in sync with the copy on the About tab below.
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
    "Good": [141, 153, 174, 190],
    "Moderate": [186, 178, 201, 190],
    "Unhealthy for Sensitive Groups": [235, 218, 238, 190],
    "Unhealthy": [254, 227, 206, 190],
    "Hazardous": [252, 191, 134, 190],
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


def render_city_map(latest_by_city, selected_city=None, height=None):
    map_data = latest_by_city.copy()
    map_data["latitude"] = map_data["city"].map(lambda city: CITY_COORDINATES[city][0])
    map_data["longitude"] = map_data["city"].map(lambda city: CITY_COORDINATES[city][1])
    map_data["color"] = map_data["aqi"].apply(aqi_map_color)
    map_data["radius"] = 5000

    # Two concentric "glow" rings for the selected city, low opacity, drawn
    # BEHIND the markers (this is the fix: the original build drew these on
    # top of the markers, so the bigger halo just erased the dot underneath
    # instead of glowing around it).
    glow_outer = map_data.copy()
    glow_outer["color"] = glow_outer["city"].eq(selected_city).map(
        {True: [255, 200, 117, 55], False: [0, 0, 0, 0]}
    )
    glow_outer["radius"] = glow_outer["city"].eq(selected_city).map({True: 32000, False: 0})

    glow_inner = map_data.copy()
    glow_inner["color"] = glow_inner["city"].eq(selected_city).map(
        {True: [255, 200, 117, 110], False: [255, 255, 255, 30]}
    )
    glow_inner["radius"] = glow_inner["city"].eq(selected_city).map({True: 16000, False: 8000})

    layers = [
        # halos first -> render underneath
        pdk.Layer(
            "ScatterplotLayer", data=glow_outer, get_position="[longitude, latitude]",
            get_fill_color="color", get_radius="radius", stroked=False,
        ),
        pdk.Layer(
            "ScatterplotLayer", data=glow_inner, get_position="[longitude, latitude]",
            get_fill_color="color", get_radius="radius", stroked=False,
        ),
        # markers last -> render on top, always visible
        pdk.Layer(
            "ScatterplotLayer", data=map_data, get_position="[longitude, latitude]",
            get_fill_color="color", get_radius="radius", pickable=True,
            stroked=True, get_line_color=[255, 255, 255, 130], line_width_min_pixels=1.5,
        ),
    ]

    if selected_city in CITY_COORDINATES:
        latitude, longitude, zoom = (*CITY_COORDINATES[selected_city], 9.5)
    else:
        # full-country view
        latitude, longitude, zoom = 30.3753, 69.3451, 4.7

    deck = pdk.Deck(
        layers=layers,
        initial_view_state=pdk.ViewState(latitude=latitude, longitude=longitude, zoom=zoom, pitch=0),
        map_style="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
        tooltip={"html": "<b>{city}</b><br/>AQI: {aqi}", "style": {"color": "white"}},
    )

    map_column, legend_column = st.columns([3.4, 1])
    with map_column:
        st.markdown('<div class="hero-map-wrap">', unsafe_allow_html=True)
        st.pydeck_chart(deck, use_container_width=True, height=height or 460)
        st.markdown('</div>', unsafe_allow_html=True)
    with legend_column:
        st.markdown("""
        <div class="aqi-legend">
            <div class="legend-title">AQI scale</div>
            <div class="legend-bar"></div>
            <div class="legend-labels"><span>0-50</span><span>51-100</span><span>101-150</span><span>151-200</span><span>201+</span></div>
            <div class="legend-note">Selected city glows amber and the map zooms in on it.</div>
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


def fetch_live_aqi(city):
    """Try the FastAPI service for a fresh reading; fall back to the local
    CSV's latest row on any failure. Previously the API response was
    fetched and immediately discarded - this now actually uses it."""
    if API_BASE_URL:
        try:
            resp = requests.get(f"{API_BASE_URL}/predict/{city}", timeout=5)
            resp.raise_for_status()
            payload = resp.json()
            if "current_aqi" in payload:
                return payload["current_aqi"], "Live API"
        except Exception:
            pass
    return None, None


df = load_features()
cities = sorted(df["city"].unique()) if not df.empty else []

tabs = st.tabs(["Home", "Live AQI", "Forecast", "EDA", "Model Explainability", "About"])

with tabs[0]:
    st.title("Pearls AQI Predictor")
    st.write("Serverless AQI forecasting for major Pakistani cities, with a feature store, model registry, and SHAP-based explainability.")
    if cities:
        overview = (df.sort_values("time").groupby("city", as_index=False).tail(1)
                    .sort_values("aqi", ascending=False))

        # hero map first - country-wide view, no city selected yet
        render_city_map(overview, None, height=440)

        latest = df.sort_values("time").iloc[-1]
        st.markdown('<div class="hero-card-row">', unsafe_allow_html=True)
        metric_columns = st.columns(4)
        metric_columns[0].metric("Cities tracked", len(cities))
        metric_columns[1].metric("Feature rows", f"{len(df):,}")
        metric_columns[2].metric("Latest AQI", f"{latest['aqi']:.0f}")
        metric_columns[3].metric("Temperature", f"{latest['temperature_2m']:.1f} C")
        st.markdown('</div>', unsafe_allow_html=True)
        st.caption(f"Data source: {data_source()} | Latest record: {latest['time']:%Y-%m-%d %H:%M}")

        fig = px.bar(overview, x="city", y="aqi", color="aqi", color_continuous_scale="YlOrRd",
                     title="Latest AQI by city", labels={"aqi": "AQI", "city": ""})
        fig.update_layout(coloraxis_showscale=False)
        style_plot(fig)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No feature data found yet. Run the backfill and feature pipeline first.")

with tabs[1]:
    st.title("Live AQI")
    if not cities:
        st.info("No feature data found yet. Run the backfill and feature pipeline first.")
    else:
        selected_city = st.selectbox("City", cities)
        city_rows = df[df.city == selected_city].sort_values("time")

        live_aqi, source_label = fetch_live_aqi(selected_city)
        if live_aqi is not None:
            current_aqi = live_aqi
        else:
            current_aqi = city_rows.iloc[-1]["aqi"] if not city_rows.empty else None
            source_label = "Local feature store"

        level, color = aqi_alert_level(current_aqi)
        current_temperature = city_rows.iloc[-1].get("temperature_2m") if not city_rows.empty else None

        cards = st.columns(3)
        cards[0].metric("Current AQI", f"{current_aqi:.0f}" if pd.notna(current_aqi) else "N/A")
        cards[1].metric("Temperature", f"{current_temperature:.1f} C" if pd.notna(current_temperature) else "N/A")
        cards[2].metric("Status", level)
        st.markdown(f"<span style='color:{color}; font-weight:bold'>{level}</span> "
                     f"<span style='color:var(--muted); font-size:0.8rem;'>&middot; source: {source_label}</span>",
                     unsafe_allow_html=True)

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
    24h / 48h / 72h ahead for 9 major Pakistani cities.

    Pipeline: Open-Meteo (historical + backfill) -> feature engineering ->
    Hopsworks feature store -> LightGBM/XGBoost/RandomForest model comparison ->
    Hopsworks model registry -> FastAPI prediction service -> this Streamlit dashboard.

    Automation: GitHub Actions runs the hourly feature pipeline and a daily retrain.
    """)