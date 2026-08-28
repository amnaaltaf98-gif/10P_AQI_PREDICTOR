"""
Phase 17-18 - Streamlit Dashboard.

Pages: Home, Live AQI, Forecast, EDA, Model Explainability.
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

st.set_page_config(page_title="Pearls AQI Predictor", layout="wide", initial_sidebar_state="expanded")

# ---------------------------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------------------------
if "theme" not in st.session_state:
    st.session_state.theme = "dark"
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "home"
if "jump_city" not in st.session_state:
    st.session_state.jump_city = None

# ---------------------------------------------------------------------------
# THEMES
# Two full palettes, switched at runtime by the toggle button in the side
# rail. Everything that's theme-sensitive (CSS vars, the pydeck basemap,
# plotly font/grid colors, marker halo colors) is driven off this dict so
# nothing gets left half-switched.
# ---------------------------------------------------------------------------
THEMES = {
    "dark": {
        "ink": "#eef6f4",
        "muted": "#a6bab6",
        "glass": "rgba(255, 255, 255, 0.055)",
        "glass_strong": "rgba(255, 255, 255, 0.10)",
        "glass_hover": "rgba(255, 255, 255, 0.16)",
        "border": "rgba(255, 255, 255, 0.14)",
        "accent": "#65e6b0",
        "accent_warm": "#ffc875",
        "base": "#0d0d0f",
        "map_style": "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
        "plot_grid": "rgba(255,255,255,0.08)",
        "halo_faint": [255, 255, 255, 30],
        "toggle_icon": "\u2600\ufe0f",  # sun - shown when in dark mode (click to go light)
    },
    "light": {
        "ink": "#292b35",
        "muted": "#646672",
        "glass": "rgba(255, 255, 255, 0.42)",
        "glass_strong": "rgba(255, 255, 255, 0.58)",
        "glass_hover": "rgba(255, 255, 255, 0.78)",
        "border": "rgba(41, 43, 53, 0.14)",
        "accent": "#8d99ae",
        "accent_warm": "#fcbf86",
        "base": "#e8e1d8",
        "map_style": "https://basemaps.cartocdn.com/gl/voyager-gl-style/style.json",
        "plot_grid": "rgba(41,43,53,0.12)",
        "halo_faint": [41, 43, 53, 25],
        "toggle_icon": "\U0001F319",  # moon - shown when in light mode (click to go dark)
    },
}

theme = THEMES[st.session_state.theme]

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
st.markdown(f"""
<style>
    :root {{
        --ink: {theme['ink']};
        --muted: {theme['muted']};
        --glass: {theme['glass']};
        --glass-strong: {theme['glass_strong']};
        --glass-hover: {theme['glass_hover']};
        --border: {theme['border']};
        --accent: {theme['accent']};
        --accent-warm: {theme['accent_warm']};
        --base: {theme['base']};
    }}

    header[data-testid="stHeader"] {{ background: transparent !important; }}
    [data-testid="collapsedControl"] {{ display: none !important; }}

    .stApp {{
        color: var(--ink);
        background: var(--base);
        background-image:
            radial-gradient(circle at 82% 8%, color-mix(in srgb, var(--accent-warm) 22%, transparent), transparent 30%),
            radial-gradient(circle at 8% 84%, color-mix(in srgb, var(--accent) 18%, transparent), transparent 32%);
        overflow-x: hidden;
        transition: background 0.3s ease;
    }}

    .block-container {{ position: relative; z-index: 1; padding-top: 1.6rem; padding-bottom: 3.5rem; max-width: 1180px; }}
    h1, h2, h3, p, label, [data-testid="stCaptionContainer"] {{ color: var(--ink); }}
    h1 {{ letter-spacing: 0.01em; font-weight: 800; }}

    /* =====================================================================
       SIDE RAIL: collapsed to icon-width by default, expands on hover to
       reveal full nav labels. Active page uses Streamlit's built-in
       type="primary" button styling so we don't need custom JS state.
       ===================================================================== */
    [data-testid="stSidebar"] {{
        background: color-mix(in srgb, var(--base) 88%, black 4%) !important;
        border-right: 1px solid var(--border);
        width: 5rem !important;
        min-width: 5rem !important;
        max-width: 5rem !important;
        transition: width 0.28s cubic-bezier(0.4, 0, 0.2, 1), min-width 0.28s cubic-bezier(0.4, 0, 0.2, 1), max-width 0.28s cubic-bezier(0.4, 0, 0.2, 1);
        overflow: hidden;
    }}
    [data-testid="stSidebar"]:hover {{
        width: 15.5rem !important;
        min-width: 15.5rem !important;
        max-width: 15.5rem !important;
    }}
    [data-testid="stSidebar"] > div {{ padding-top: 1rem; }}
    [data-testid="stSidebarUserContent"] {{ padding: 0 0.6rem; }}

    .side-brand {{
        display: flex; align-items: center; gap: 0.6rem;
        white-space: nowrap; overflow: hidden;
        font-weight: 800; font-size: 1.05rem; color: var(--ink);
        padding: 0.4rem 0.35rem 1.1rem 0.35rem;
        border-bottom: 1px solid var(--border);
        margin-bottom: 0.8rem;
    }}

    [data-testid="stSidebar"] .stButton > button {{
        background: transparent !important;
        border: 1px solid transparent !important;
        color: var(--muted) !important;
        text-align: left !important;
        justify-content: flex-start !important;
        align-items: center !important;
        display: flex !important;
        min-height: 2.9rem !important;
        line-height: 1.8 !important;
        white-space: nowrap;
        overflow-x: hidden;
        overflow-y: visible;
        border-radius: 10px !important;
        padding: 0.5rem 0.75rem !important;
        margin-bottom: 0.3rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.2s ease;
    }}
    [data-testid="stSidebar"] .stButton > button p {{
        line-height: 1.8 !important;
        overflow: visible !important;
    }}
    [data-testid="stSidebar"] .stButton > button:hover {{
        background: var(--glass-hover) !important;
        color: var(--ink) !important;
    }}
    [data-testid="stSidebar"] .stButton > button[kind="primary"] {{
        background: var(--accent-warm) !important;
        color: #1e2029 !important;
        border-color: rgba(252,191,134,0.8) !important;
        box-shadow: 0 0 18px rgba(252,191,134,0.28);
    }}
    .nav-spacer {{ height: 0.6rem; }}
    .theme-row {{ display: flex; justify-content: flex-start; margin-bottom: 0.6rem; }}

    /* =====================================================================
       GLASS PANELS + INTERACTIVITY
       ===================================================================== */
    [data-testid="stMetric"], [data-testid="stAlert"], [data-testid="stExpander"] {{
        background: var(--glass);
        border: 1px solid var(--border);
        border-radius: 16px;
        box-shadow: 0 16px 40px rgba(0, 0, 0, 0.20), inset 0 1px 0 rgba(255,255,255,0.06);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
    }}
    [data-testid="stMetric"] {{ padding: 1rem; }}
    [data-testid="stMetric"]:hover {{
        transform: translateY(-3px);
        border-color: var(--accent);
        box-shadow: 0 20px 48px rgba(0,0,0,0.28);
    }}
    [data-testid="stMetricLabel"] p {{ color: var(--muted) !important; font-weight: 700; }}
    [data-testid="stMetricValue"] {{
        color: var(--ink) !important;
        white-space: normal !important;
        overflow: visible !important;
        text-overflow: unset !important;
        line-height: 1.2;
        font-size: clamp(1.05rem, 2.4vw, 1.6rem) !important;
    }}
    [data-testid="stMetricDelta"] {{ color: var(--accent) !important; }}

    [data-testid="stSelectbox"] > div > div {{
        background: var(--glass-strong) !important;
        border: 1px solid var(--border);
        border-radius: 12px;
        color: var(--ink);
        backdrop-filter: blur(10px);
        transition: all 0.25s ease;
    }}
    [data-testid="stSelectbox"] > div > div:hover {{
        border-color: var(--accent);
        background: var(--glass-hover) !important;
        box-shadow: 0 0 18px color-mix(in srgb, var(--accent) 25%, transparent);
    }}

    [data-testid="stMain"] .stButton > button, .stDownloadButton > button {{
        background: var(--glass-strong) !important;
        border: 1px solid var(--border) !important;
        color: var(--ink) !important;
        border-radius: 12px !important;
        backdrop-filter: blur(10px);
        transition: all 0.2s ease;
    }}
    [data-testid="stMain"] .stButton > button:hover, .stDownloadButton > button:hover {{
        background: var(--glass-hover) !important;
        border-color: var(--accent) !important;
        box-shadow: 0 0 18px color-mix(in srgb, var(--accent) 25%, transparent);
        transform: translateY(-2px);
    }}

    [data-testid="stPlotlyChart"], .hero-map-wrap {{
        background: var(--glass);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 0.4rem;
        box-shadow: 0 16px 40px rgba(0, 0, 0, 0.18);
        backdrop-filter: blur(10px);
        transition: box-shadow 0.25s ease, border-color 0.25s ease;
    }}
    [data-testid="stPlotlyChart"]:hover {{ border-color: var(--accent); }}
    .hero-map-wrap {{ overflow: hidden; padding: 0; }}

    .aqi-legend {{
        padding: 1rem;
        border: 1px solid var(--border);
        border-radius: 12px;
        background: var(--glass);
        box-shadow: 0 16px 40px rgba(0, 0, 0, 0.2), inset 0 1px 0 rgba(255,255,255,0.06);
        backdrop-filter: blur(14px);
    }}
    .legend-title {{ color: var(--ink); font-size: 0.9rem; font-weight: 800; margin-bottom: 0.7rem; }}
    .legend-bar {{
        height: 11px;
        border-radius: 999px;
        background: linear-gradient(90deg, #8D99AE 0 20%, #BBB2C9 20% 40%, #EBDAEE 40% 60%, #FEE3CE 60% 80%, #FCBF86 80% 100%);
    }}
    .legend-labels {{ display: flex; justify-content: space-between; gap: 0.25rem; margin-top: 0.45rem; color: var(--muted); font-size: 0.62rem; }}
    .legend-note {{ color: var(--muted); font-size: 0.72rem; line-height: 1.3; margin-top: 1rem; }}
    [data-testid="stCaptionContainer"] {{ color: var(--muted) !important; }}

    /* clickable "jump to city" chips on Home */
    .city-chip-row .stButton > button {{
        border-radius: 999px !important;
        padding: 0.4rem 1rem !important;
        font-size: 0.85rem;
    }}
</style>
""", unsafe_allow_html=True)


def style_plot(fig, height=380):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": theme["ink"]},
        title_font={"color": theme["ink"], "size": 18},
        height=height,
        margin={"l": 24, "r": 24, "t": 56, "b": 24},
        legend={"font": {"color": theme["muted"]}},
        xaxis={
            "gridcolor": theme["plot_grid"], "zerolinecolor": theme["plot_grid"],
            "color": theme["muted"], "tickfont": {"color": theme["muted"]},
            "title_font": {"color": theme["muted"]},
        },
        yaxis={
            "gridcolor": theme["plot_grid"], "zerolinecolor": theme["plot_grid"],
            "color": theme["muted"], "tickfont": {"color": theme["muted"]},
            "title_font": {"color": theme["muted"]},
        },
        coloraxis_colorbar={"tickfont": {"color": theme["muted"]}, "title_font": {"color": theme["muted"]}},
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


# 9 cities.
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

    glow_outer = map_data.copy()
    glow_outer["color"] = glow_outer["city"].eq(selected_city).map(
        {True: [255, 200, 117, 55], False: [0, 0, 0, 0]}
    )
    glow_outer["radius"] = glow_outer["city"].eq(selected_city).map({True: 32000, False: 0})

    glow_inner = map_data.copy()
    faint = theme["halo_faint"]
    glow_inner["color"] = glow_inner["city"].eq(selected_city).map(
        {True: [255, 200, 117, 110], False: faint}
    )
    glow_inner["radius"] = glow_inner["city"].eq(selected_city).map({True: 16000, False: 8000})

    layers = [
        pdk.Layer(
            "ScatterplotLayer", data=glow_outer, get_position="[longitude, latitude]",
            get_fill_color="color", get_radius="radius", stroked=False,
        ),
        pdk.Layer(
            "ScatterplotLayer", data=glow_inner, get_position="[longitude, latitude]",
            get_fill_color="color", get_radius="radius", stroked=False,
        ),
        pdk.Layer(
            "ScatterplotLayer", data=map_data, get_position="[longitude, latitude]",
            get_fill_color="color", get_radius="radius", pickable=True,
            stroked=True, get_line_color=[255, 255, 255, 130], line_width_min_pixels=1.5,
        ),
    ]

    if selected_city in CITY_COORDINATES:
        latitude, longitude, zoom = (*CITY_COORDINATES[selected_city], 9.5)
    else:
        latitude, longitude, zoom = 30.3753, 69.3451, 4.7

    deck = pdk.Deck(
        layers=layers,
        initial_view_state=pdk.ViewState(latitude=latitude, longitude=longitude, zoom=zoom, pitch=0),
        map_style=theme["map_style"],
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


@st.cache_data(ttl=3600)
def load_model_metrics():
    """Expects models/metrics.json written by the training pipeline, shaped:
    {"24h": {"r2": 0.87, "mae": 6.1, "rmse": 9.4}, "48h": {...}, "72h": {...}}
    """
    import json

    metrics_path = PROJECT_ROOT / "models" / "metrics.json"
    if not metrics_path.exists():
        return None
    try:
        with open(metrics_path) as f:
            return json.load(f)
    except Exception:
        return None


df = load_features()
cities = sorted(df["city"].unique()) if not df.empty else []

# ---------------------------------------------------------------------------
# SIDE RAIL NAV
# ---------------------------------------------------------------------------
NAV_ITEMS = [
    ("home", "\U0001F3E0", "Home"),
    ("live", "\U0001F4E1", "Live AQI"),
    ("forecast", "\U0001F4C8", "Forecast"),
    ("eda", "\U0001F50D", "EDA"),
    ("model", "\U0001F9E0", "Model Explainability"),
]

with st.sidebar:
    st.markdown('<div class="theme-row">', unsafe_allow_html=True)
    if st.button(theme["toggle_icon"] + "  Theme", key="theme_toggle", use_container_width=True):
        st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="side-brand">\U0001F32B\ufe0f Pearls AQI</div>', unsafe_allow_html=True)

    for key, icon, label in NAV_ITEMS:
        is_active = st.session_state.active_tab == key
        if st.button(f"{icon}  {label}", key=f"nav_{key}",
                     type="primary" if is_active else "secondary",
                     use_container_width=True):
            st.session_state.active_tab = key
            st.rerun()

active = st.session_state.active_tab

# ---------------------------------------------------------------------------
# HOME
# ---------------------------------------------------------------------------
if active == "home":
    st.title("Pearls AQI Predictor")
    if cities:
        overview = (df.sort_values("time").groupby("city", as_index=False).tail(1)
                    .sort_values("aqi", ascending=False))

        top_row = st.columns([1, 1, 1, 1, 0.6])
        latest = df.sort_values("time").iloc[-1]
        top_row[0].metric("Cities tracked", len(cities))
        top_row[1].metric("Feature rows", f"{len(df):,}")
        top_row[2].metric("Latest AQI", f"{latest['aqi']:.0f}")
        top_row[3].metric("Temperature", f"{latest['temperature_2m']:.1f} C")
        with top_row[4]:
            st.write("")
            if st.button("\U0001F504 Refresh", help="Clear cache and reload the latest feature data"):
                load_features.clear()
                st.rerun()
        st.caption(f"Data source: {data_source()} | Latest record: {latest['time']:%Y-%m-%d %H:%M}")

        st.subheader("Worst air quality right now")
        st.caption("Tap a city to jump straight to its live reading.")
        worst = overview.head(3)
        st.markdown('<div class="city-chip-row">', unsafe_allow_html=True)
        chip_cols = st.columns(len(worst) if len(worst) else 1)
        for col, (_, row) in zip(chip_cols, worst.iterrows()):
            level, _ = aqi_alert_level(row["aqi"])
            with col:
                if st.button(f"{row['city']} \u00b7 AQI {row['aqi']:.0f}", key=f"chip_{row['city']}",
                             use_container_width=True, help=level):
                    st.session_state.jump_city = row["city"]
                    st.session_state.active_tab = "live"
                    st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

        fig = px.bar(overview, x="city", y="aqi", color="aqi", color_continuous_scale="YlOrRd",
                     title="Latest AQI by city", labels={"aqi": "AQI", "city": ""})
        fig.update_layout(coloraxis_showscale=False)
        style_plot(fig)
        st.plotly_chart(fig, use_container_width=True, theme=None)
    else:
        st.info("No feature data found yet. Run the backfill and feature pipeline first.")

# ---------------------------------------------------------------------------
# LIVE AQI
# ---------------------------------------------------------------------------
elif active == "live":
    st.title("Live AQI")
    if not cities:
        st.info("No feature data found yet. Run the backfill and feature pipeline first.")
    else:
        default_index = cities.index(st.session_state.jump_city) if st.session_state.jump_city in cities else 0
        selected_city = st.selectbox("City", cities, index=default_index, key="live_city_select")
        st.session_state.jump_city = None  # only pre-select once
        city_rows = df[df.city == selected_city].sort_values("time")

        with st.spinner("Checking live feed..."):
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
        st.plotly_chart(fig, use_container_width=True, theme=None)

# ---------------------------------------------------------------------------
# FORECAST
# ---------------------------------------------------------------------------
elif active == "forecast":
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
            st.plotly_chart(fig, use_container_width=True, theme=None)
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
                st.plotly_chart(fig, use_container_width=True, theme=None)
            except Exception as e:
                st.warning(f"Could not reach prediction API: {e}")

# ---------------------------------------------------------------------------
# EDA
# ---------------------------------------------------------------------------
elif active == "eda":
    st.title("Exploratory Data Analysis")
    if df.empty:
        st.info("No feature data found yet.")
    else:
        selected_city = st.selectbox("City", cities, key="eda_city")
        city_df = df[df.city == selected_city]
        fig = px.line(city_df, x="time", y="aqi", title=f"AQI over time - {selected_city}")
        style_plot(fig)
        st.plotly_chart(fig, use_container_width=True, theme=None)

        fig2 = px.box(df, x="city", y="aqi", title="AQI distribution by city")
        style_plot(fig2)
        st.plotly_chart(fig2, use_container_width=True, theme=None)

        temp_fig = px.scatter(df.sample(min(len(df), 3000), random_state=42), x="temperature_2m", y="aqi",
                      color="city", opacity=0.65, title="Temperature and AQI relationship",
                      labels={"temperature_2m": "Temperature (C)", "aqi": "AQI"})
        style_plot(temp_fig)
        st.plotly_chart(temp_fig, use_container_width=True, theme=None)

# ---------------------------------------------------------------------------
# MODEL EXPLAINABILITY
# ---------------------------------------------------------------------------
elif active == "model":
    st.title("Model Explainability (SHAP)")
    st.write("Feature contribution plots generated by the training pipeline.")
    for horizon in ["24h", "48h", "72h"]:
        img_path = PROJECT_ROOT / "models" / f"shap_target_aqi_{horizon}.png"
        try:
            st.image(img_path, caption=f"SHAP summary - {horizon} forecast")
        except Exception:
            st.info(f"No SHAP plot found yet for {horizon}. Train the model first.")

    st.subheader("Model performance")
    metrics = load_model_metrics()
    if metrics:
        for horizon in ["24h", "48h", "72h"]:
            if horizon not in metrics:
                continue
            m = metrics[horizon]
            st.markdown(f"**{horizon} forecast**")
            score_cols = st.columns(3)
            score_cols[0].metric("R\u00b2", f"{m.get('r2', float('nan')):.3f}")
            score_cols[1].metric("MAE", f"{m.get('mae', float('nan')):.2f}")
            score_cols[2].metric("RMSE", f"{m.get('rmse', float('nan')):.2f}")
    else:
        st.info("No metrics.json found yet in /models. Have the training pipeline write "
                "R\u00b2, MAE, and RMSE per horizon there and this section will populate automatically.")