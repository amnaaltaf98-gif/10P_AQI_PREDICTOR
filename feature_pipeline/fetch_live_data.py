"""
Hourly live data pipeline (Phase 4 / Phase 6).

For each city:
    1. Fetch current pollutants from OpenWeather Air Pollution API
    2. Fetch current weather from Open-Meteo Forecast API (same variable names as the
       historical Archive API, so features line up exactly)
    3. Compute AQI the same way backfill_history.py does
    4. Append the new raw row to data/raw_all_cities.csv (keeps growing, becomes the
       running history used both for future retraining and for computing lag/rolling
       features on the next hourly run)
    5. Recompute lag/rolling/time/derived features using the last 48h of history + the
       new row, keep only the new row's fully-featured version
    6. Push that single feature row to Hopsworks

This is the script GitHub Actions runs every hour (see .github/workflows/hourly_feature_pipeline.yml).

Usage:
    python fetch_live_data.py
"""

import os
import time
import requests
import pandas as pd

from config import (
    CITIES,
    OPENWEATHER_API_KEY,
    OPENWEATHER_AIR_POLLUTION_URL,
    HOPSWORKS_API_KEY,
    HOPSWORKS_PROJECT_NAME,
    FEATURE_GROUP_NAME,
    FEATURE_GROUP_VERSION,
)
from aqi_utils import compute_aqi
from feature_engineering import build_features

RAW_HISTORY_PATH = "../data/live_buffer.csv"
FULL_HISTORY_PATH = "../data/raw_all_cities.csv"  # only used once, to seed the buffer on first run

# how much history (in hours) to pull alongside the new reading so lag_24h / rolling_24h can be computed
HISTORY_WINDOW_HOURS = 48

# how much history to KEEP in the rolling buffer file (a bit more than HISTORY_WINDOW_HOURS as
# safety margin). Keeping this small means it's cheap to commit back to git after every run.
BUFFER_RETENTION_HOURS = 24 * 7  # 1 week per city

OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 3

CURRENT_WEATHER_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "surface_pressure",
    "wind_speed_10m",
    "wind_direction_10m",
    "cloud_cover",
    "precipitation",
    "shortwave_radiation",
]


def _get_with_retry(url, params, timeout=30):
    """Small retry wrapper - handles transient connection resets / timeouts, common on flaky wifi."""
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, params=params, timeout=timeout)
            resp.raise_for_status()
            return resp
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            last_error = e
            if attempt < MAX_RETRIES:
                print(f"    retry {attempt}/{MAX_RETRIES} after error: {e}")
                time.sleep(RETRY_BACKOFF_SECONDS * attempt)
    raise last_error


def fetch_current_weather(lat, lon):
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": ",".join(CURRENT_WEATHER_VARS),
        "timezone": "Asia/Karachi",
    }
    resp = _get_with_retry(OPEN_METEO_FORECAST_URL, params)
    return resp.json()["current"]


def fetch_current_pollution(lat, lon):
    params = {"lat": lat, "lon": lon, "appid": OPENWEATHER_API_KEY}
    resp = _get_with_retry(OPENWEATHER_AIR_POLLUTION_URL, params)
    return resp.json()["list"][0]["components"]


def fetch_all_cities_raw():
    """One raw row per city, matching the exact column schema backfill_history.py produces."""
    rows = []
    now = pd.Timestamp.now(tz="Asia/Karachi").tz_localize(None).floor("h")

    for city_name, info in CITIES.items():
        try:
            weather = fetch_current_weather(info["lat"], info["lon"])
            pollution = fetch_current_pollution(info["lat"], info["lon"])

            row = {
                "time": now,
                "city": city_name,
                "temperature_2m": weather.get("temperature_2m"),
                "relative_humidity_2m": weather.get("relative_humidity_2m"),
                "surface_pressure": weather.get("surface_pressure"),
                "wind_speed_10m": weather.get("wind_speed_10m"),
                "wind_direction_10m": weather.get("wind_direction_10m"),
                "cloud_cover": weather.get("cloud_cover"),
                "precipitation": weather.get("precipitation"),
                "shortwave_radiation": weather.get("shortwave_radiation"),
                "pm2_5": pollution.get("pm2_5"),
                "pm10": pollution.get("pm10"),
                "nitrogen_dioxide": pollution.get("no2"),
                "sulphur_dioxide": pollution.get("so2"),
                "carbon_monoxide": pollution.get("co"),
                "ozone": pollution.get("o3"),
                "dust": None,                   # OpenWeather doesn't provide these two -
                "aerosol_optical_depth": None,  # fine, historical rows have them, live ones just carry NaN
            }
            row["aqi"] = compute_aqi(pm25=row["pm2_5"], pm10=row["pm10"])
            rows.append(row)
            print(f"  fetched {city_name}: AQI={row['aqi']}")
        except Exception as e:
            print(f"  FAILED for {city_name}: {e}")

    return pd.DataFrame(rows)


def append_to_raw_history(new_rows_df):
    if os.path.exists(RAW_HISTORY_PATH):
        history_df = pd.read_csv(RAW_HISTORY_PATH)
        history_df["time"] = pd.to_datetime(history_df["time"], format="mixed")
    elif os.path.exists(FULL_HISTORY_PATH):
        # first run: seed the small rolling buffer from the full backfilled history
        # (only need the tail - the buffer never needs the full multi-year archive)
        print(f"  {RAW_HISTORY_PATH} not found, seeding from {FULL_HISTORY_PATH}...")
        full_df = pd.read_csv(FULL_HISTORY_PATH)
        full_df["time"] = pd.to_datetime(full_df["time"], format="mixed")
        cutoff = full_df["time"].max() - pd.Timedelta(hours=BUFFER_RETENTION_HOURS)
        history_df = full_df[full_df["time"] >= cutoff].copy()
    else:
        history_df = pd.DataFrame(columns=new_rows_df.columns)

    combined = pd.concat([history_df, new_rows_df], ignore_index=True)
    combined = combined.drop_duplicates(subset=["city", "time"], keep="last")
    combined = combined.sort_values(["city", "time"]).reset_index(drop=True)

    # trim each city down to the retention window so this file stays small enough to commit to git
    cutoff = combined["time"].max() - pd.Timedelta(hours=BUFFER_RETENTION_HOURS)
    combined = combined[combined["time"] >= cutoff].reset_index(drop=True)

    combined.to_csv(RAW_HISTORY_PATH, index=False)

    return combined


def compute_live_features(combined_history_df, new_rows_df):
    """
    Recomputes features using recent history + the new rows, then returns ONLY the
    new rows (now with lag/rolling/derived features filled in from that history).
    """
    new_timestamps = set(zip(new_rows_df["city"], new_rows_df["time"]))

    # only need the recent window per city to compute lags/rolling stats - no need to
    # recompute features over the entire multi-year history every single hour
    recent_frames = []
    for city in combined_history_df["city"].unique():
        city_df = combined_history_df[combined_history_df["city"] == city].sort_values("time")
        recent_frames.append(city_df.tail(HISTORY_WINDOW_HOURS + 1))
    recent_df = pd.concat(recent_frames, ignore_index=True)

    featured = build_features(recent_df, drop_incomplete=False)

    featured["time"] = pd.to_datetime(featured["time"])
    is_new_row = featured.apply(lambda r: (r["city"], r["time"]) in new_timestamps, axis=1)
    return featured[is_new_row].reset_index(drop=True)


def push_to_hopsworks(df):
    import hopsworks

    project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY, project=HOPSWORKS_PROJECT_NAME)
    fs = project.get_feature_store()
    fg = fs.get_feature_group(name=FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)
    fg.insert(df, write_options={"wait_for_job": True})
    print(f"Pushed {len(df)} live rows to Hopsworks feature group '{FEATURE_GROUP_NAME}'.")


def main():
    print("Fetching current weather + pollution for all cities...")
    new_rows_df = fetch_all_cities_raw()

    if new_rows_df.empty:
        print("No data fetched for any city, aborting.")
        return

    print("Appending to local raw history...")
    combined_history_df = append_to_raw_history(new_rows_df)

    print("Recomputing lag/rolling/derived features for the new rows...")
    live_features_df = compute_live_features(combined_history_df, new_rows_df)
    print(live_features_df[["time", "city", "aqi", "aqi_lag_1h", "aqi_roll_mean_24h"]])

    print("Pushing to Hopsworks...")
    push_to_hopsworks(live_features_df)


if __name__ == "__main__":
    main()