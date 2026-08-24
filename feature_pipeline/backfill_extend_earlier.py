"""
Extends your historical backfill further back in time, then pushes ONLY the
newly-added (earlier) rows to Hopsworks - not the whole dataset again.

Why not just re-run backfill_history.py with an earlier start date? Because
your lag/rolling features need real history to compute correctly. If we only
fetched and feature-engineered the new earlier window in isolation, the
BOUNDARY rows (where old data meets new data) would still be wrong - the
first ~24 hours of your current earliest data currently have no valid
aqi_lag_24h and got dropped during feature engineering. Extending backward
should actually FIX that boundary, not just add a disconnected new chunk.

So this script:
    1. Fetches raw weather+pollution for [NEW_START_DATE, current_earliest - 1h]
    2. Merges it with your existing raw_all_cities.csv
    3. Re-runs feature engineering on the FULL combined raw data (so lag/rolling
       features are correct across the old/new boundary)
    4. Filters the result down to just the NEWLY covered rows
    5. Pushes only those new rows to Hopsworks (keeps the materialization job
       small and cheap, given your budget history with this)

IMPORTANT: Open-Meteo's Air Quality API (CAMS model) has a more limited
historical range than its weather archive (which goes back to 1940). If you
set NEW_START_DATE too far back, you may get empty/partial responses for the
air quality portion specifically. This script will print a warning if that
happens rather than silently producing broken data - check the output.

Usage:
    python backfill_extend_earlier.py 2022-01-01
"""

import sys
import time
import requests
import pandas as pd

from config import (
    CITIES,
    OPEN_METEO_ARCHIVE_URL,
    OPEN_METEO_AIR_QUALITY_URL,
    HOPSWORKS_API_KEY,
    HOPSWORKS_PROJECT_NAME,
    FEATURE_GROUP_NAME,
    FEATURE_GROUP_VERSION,
)
from aqi_utils import compute_aqi
from feature_engineering import build_features

WEATHER_HOURLY_VARS = [
    "temperature_2m", "relative_humidity_2m", "surface_pressure",
    "wind_speed_10m", "wind_direction_10m", "cloud_cover",
    "precipitation", "shortwave_radiation",
]

AIR_QUALITY_HOURLY_VARS = [
    "pm2_5", "pm10", "nitrogen_dioxide", "sulphur_dioxide",
    "carbon_monoxide", "ozone", "dust", "aerosol_optical_depth",
]

RAW_HISTORY_PATH = "../data/raw_all_cities.csv"
FEATURES_OUTPUT_PATH = "../data/features_all_cities.csv"


def fetch_weather(lat, lon, start_date, end_date):
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": start_date, "end_date": end_date,
        "hourly": ",".join(WEATHER_HOURLY_VARS),
        "timezone": "Asia/Karachi",
    }
    resp = requests.get(OPEN_METEO_ARCHIVE_URL, params=params, timeout=60)
    resp.raise_for_status()
    return pd.DataFrame(resp.json()["hourly"])


def fetch_air_quality(lat, lon, start_date, end_date):
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": start_date, "end_date": end_date,
        "hourly": ",".join(AIR_QUALITY_HOURLY_VARS),
        "timezone": "Asia/Karachi",
    }
    resp = requests.get(OPEN_METEO_AIR_QUALITY_URL, params=params, timeout=60)
    resp.raise_for_status()
    return pd.DataFrame(resp.json()["hourly"])


def fetch_extension(new_start_date, new_end_date):
    all_rows = []
    for city_name, info in CITIES.items():
        print(f"Fetching {city_name} ({new_start_date} to {new_end_date})...")
        try:
            weather_df = fetch_weather(info["lat"], info["lon"], new_start_date, new_end_date)
            air_df = fetch_air_quality(info["lat"], info["lon"], new_start_date, new_end_date)

            if weather_df.empty or air_df.empty:
                print(f"  WARNING: empty response for {city_name} - Open-Meteo may not have "
                      f"data this far back for one of these endpoints. Skipping.")
                continue

            df = pd.merge(weather_df, air_df, on="time", how="inner")
            df["time"] = pd.to_datetime(df["time"])
            df["city"] = city_name
            df["aqi"] = df.apply(lambda r: compute_aqi(pm25=r.get("pm2_5"), pm10=r.get("pm10")), axis=1)

            all_rows.append(df)
            print(f"  got {len(df)} rows")
        except Exception as e:
            print(f"  FAILED for {city_name}: {e}")

        time.sleep(2)  # be polite to the free API

    if not all_rows:
        return pd.DataFrame()
    return pd.concat(all_rows, ignore_index=True)


def push_new_rows_to_hopsworks(new_features_df):
    import hopsworks

    project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY, project=HOPSWORKS_PROJECT_NAME)
    fs = project.get_feature_store()
    fg = fs.get_feature_group(name=FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)

    print(f"Pushing {len(new_features_df)} newly-backfilled rows to Hopsworks...")
    fg.insert(new_features_df, write_options={"wait_for_job": False})
    print("Insert submitted.")


def main():
    if len(sys.argv) < 2:
        print("Usage: python backfill_extend_earlier.py YYYY-MM-DD")
        print("Example: python backfill_extend_earlier.py 2022-01-01")
        sys.exit(1)

    new_start_date = sys.argv[1]

    print(f"Loading existing raw history from {RAW_HISTORY_PATH}...")
    existing_raw = pd.read_csv(RAW_HISTORY_PATH)
    existing_raw["time"] = pd.to_datetime(existing_raw["time"], format="mixed")
    current_earliest = existing_raw["time"].min()
    new_end_date = (current_earliest - pd.Timedelta(hours=1)).strftime("%Y-%m-%d")

    print(f"Current earliest data: {current_earliest}")
    print(f"Will fetch new data from {new_start_date} to {new_end_date}")

    if pd.Timestamp(new_start_date) >= current_earliest:
        print("New start date is not actually earlier than your existing data. Nothing to do.")
        sys.exit(0)

    extension_raw = fetch_extension(new_start_date, new_end_date)
    if extension_raw.empty:
        print("No new data was fetched (all cities failed or returned empty). Stopping.")
        sys.exit(1)

    print(f"\nFetched {len(extension_raw)} new raw rows across all cities.")

    print("Merging with existing raw history...")
    combined_raw = pd.concat([extension_raw, existing_raw], ignore_index=True)
    combined_raw = combined_raw.drop_duplicates(subset=["city", "time"], keep="last")
    combined_raw = combined_raw.sort_values(["city", "time"]).reset_index(drop=True)
    combined_raw.to_csv(RAW_HISTORY_PATH, index=False)
    print(f"Saved combined raw history: {len(combined_raw)} total rows -> {RAW_HISTORY_PATH}")

    print("\nRe-running feature engineering on the FULL combined dataset "
          "(needed so lag/rolling features are correct across the old/new boundary)...")
    combined_features = build_features(combined_raw, drop_incomplete=True)
    combined_features.to_csv(FEATURES_OUTPUT_PATH, index=False)
    print(f"Saved combined features: {len(combined_features)} total rows -> {FEATURES_OUTPUT_PATH}")

    # only push the rows that are actually new (before what Hopsworks already has)
    new_rows_only = combined_features[combined_features["time"] < current_earliest].copy()
    print(f"\n{len(new_rows_only)} of those rows are newly-covered dates not yet in Hopsworks.")

    if len(new_rows_only) == 0:
        print("Nothing new to push (all new dates got dropped for lacking full lag history - "
              "try an even earlier start date, or this is expected if the extension was very short).")
        return

    push_new_rows_to_hopsworks(new_rows_only)


if __name__ == "__main__":
    main()