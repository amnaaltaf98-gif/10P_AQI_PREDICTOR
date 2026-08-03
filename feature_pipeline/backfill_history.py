"""
Phase 9 - Historical Backfill.

Pulls ~2.5 years of hourly weather + air pollutant data for all 10 cities
from Open-Meteo (both Archive Weather API and Air Quality API), computes AQI
using EPA breakpoints, and saves one CSV per city to data/raw/.

Run this once. Re-run only if you need to extend the date range.

Usage:
    python backfill_history.py
"""

import time
import requests
import pandas as pd

from config import (
    CITIES,
    OPEN_METEO_ARCHIVE_URL,
    OPEN_METEO_AIR_QUALITY_URL,
    BACKFILL_START_DATE,
    BACKFILL_END_DATE,
)
from aqi_utils import compute_aqi

WEATHER_HOURLY_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "surface_pressure",
    "wind_speed_10m",
    "wind_direction_10m",
    "cloud_cover",
    "precipitation",
    "shortwave_radiation",
]

AIR_QUALITY_HOURLY_VARS = [
    "pm2_5",
    "pm10",
    "nitrogen_dioxide",
    "sulphur_dioxide",
    "carbon_monoxide",
    "ozone",
    "dust",
    "aerosol_optical_depth",
]


def fetch_weather(lat, lon, start_date, end_date):
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(WEATHER_HOURLY_VARS),
        "timezone": "Asia/Karachi",
    }
    resp = requests.get(OPEN_METEO_ARCHIVE_URL, params=params, timeout=60)
    resp.raise_for_status()
    return pd.DataFrame(resp.json()["hourly"])


def fetch_air_quality(lat, lon, start_date, end_date):
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(AIR_QUALITY_HOURLY_VARS),
        "timezone": "Asia/Karachi",
    }
    resp = requests.get(OPEN_METEO_AIR_QUALITY_URL, params=params, timeout=60)
    resp.raise_for_status()
    return pd.DataFrame(resp.json()["hourly"])


def backfill_city(city_name, lat, lon, start_date, end_date):
    print(f"Fetching {city_name}...")

    weather_df = fetch_weather(lat, lon, start_date, end_date)
    air_df = fetch_air_quality(lat, lon, start_date, end_date)

    df = pd.merge(weather_df, air_df, on="time", how="inner")
    df["time"] = pd.to_datetime(df["time"])
    df["city"] = city_name

    df["aqi"] = df.apply(
        lambda row: compute_aqi(pm25=row.get("pm2_5"), pm10=row.get("pm10")), axis=1
    )

    return df


def main():
    all_rows = []

    for city_name, info in CITIES.items():
        try:
            df = backfill_city(city_name, info["lat"], info["lon"], BACKFILL_START_DATE, BACKFILL_END_DATE)
            out_path = f"../data/raw_{city_name.lower()}.csv"
            df.to_csv(out_path, index=False)
            print(f"  saved {len(df)} rows -> {out_path}")
            all_rows.append(df)
        except Exception as e:
            print(f"  FAILED for {city_name}: {e}")

        # be polite to the free API, avoid rate limiting
        time.sleep(2)

    if all_rows:
        combined = pd.concat(all_rows, ignore_index=True)
        combined.to_csv("../data/raw_all_cities.csv", index=False)
        print(f"\nDone. Combined dataset: {len(combined)} rows across {combined['city'].nunique()} cities.")


if __name__ == "__main__":
    main()
