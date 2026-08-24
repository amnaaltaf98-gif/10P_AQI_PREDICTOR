"""
Rebuilds data/live_buffer.csv from Hopsworks instead of the local raw CSV.

Why this is needed: the local raw_all_cities.csv only updates when you
manually run a backfill/extend script, so it can go stale for weeks. Your
Hopsworks feature store, on the other hand, has been continuously fed by the
hourly GitHub Action this whole time - it's the more current source right now.

Hopsworks stores the RAW columns (temperature_2m, pm2_5, aqi, etc.) alongside
all the derived ones - this script just pulls the last N days and selects
back down to the raw columns live_buffer.csv actually needs, matching the
exact schema fetch_live_data.py expects.

Usage:
    python reseed_buffer_from_hopsworks.py
    python reseed_buffer_from_hopsworks.py 14   # custom number of days
"""

import sys
import pandas as pd

from config import HOPSWORKS_API_KEY, HOPSWORKS_PROJECT_NAME, FEATURE_GROUP_NAME, FEATURE_GROUP_VERSION

LIVE_BUFFER_PATH = "../data/live_buffer.csv"

RAW_COLUMNS = [
    "time", "city",
    "temperature_2m", "relative_humidity_2m", "surface_pressure",
    "wind_speed_10m", "wind_direction_10m", "cloud_cover",
    "precipitation", "shortwave_radiation",
    "pm2_5", "pm10", "nitrogen_dioxide", "sulphur_dioxide",
    "carbon_monoxide", "ozone", "dust", "aerosol_optical_depth",
    "aqi",
]

DEFAULT_DAYS = 12  # a bit more than 7 (BUFFER_RETENTION_HOURS) as safety margin


def main():
    days = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DAYS

    import hopsworks
    print("Logging into Hopsworks...")
    project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY, project=HOPSWORKS_PROJECT_NAME)
    fs = project.get_feature_store()
    fg = fs.get_feature_group(name=FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)

    print(f"Reading feature group (this can take a couple minutes)...")
    df = fg.read()
    print(f"Loaded {len(df)} total rows from Hopsworks.")

    # Convert timestamps to datetime and strip timezone info to make it timezone-naive (matching fetch_live_data.py)
    df["time"] = pd.to_datetime(df["time"]).dt.tz_localize(None)

    cutoff = df["time"].max() - pd.Timedelta(days=days)
    recent = df[df["time"] >= cutoff].copy()
    print(f"Filtered to last {days} days: {len(recent)} rows "
          f"({recent['time'].min()} to {recent['time'].max()})")

    missing_cols = [c for c in RAW_COLUMNS if c not in recent.columns]
    if missing_cols:
        print(f"WARNING: these expected columns are missing from Hopsworks data: {missing_cols}")

    available_cols = [c for c in RAW_COLUMNS if c in recent.columns]
    raw_subset = recent[available_cols].copy()
    raw_subset = raw_subset.sort_values(["city", "time"]).reset_index(drop=True)

    # Format 'time' as string without timezone suffixes before saving to CSV
    raw_subset["time"] = raw_subset["time"].dt.strftime("%Y-%m-%d %H:%M:%S")

    raw_subset.to_csv(LIVE_BUFFER_PATH, index=False)
    print(f"\nSaved {len(raw_subset)} rows -> {LIVE_BUFFER_PATH}")
    print(f"Cities covered: {sorted(raw_subset['city'].unique())}")
    print(f"Time range: {raw_subset['time'].min()} to {raw_subset['time'].max()}")


if __name__ == "__main__":
    main()