"""
Phase 7 - Feature Engineering.

IMPORTANT: every lag and rolling feature is computed with `.groupby("city")` first.
Without this, row N+1 of one city can pick up the previous city's last row as its
"1 hour ago" value once the dataframe is sorted by time alone. That cross-city leakage
quietly inflates model performance during training and falls apart in production, since
live data always arrives one city at a time. This script is written so it can't happen.

Input: a combined dataframe like data/raw_all_cities.csv (output of backfill_history.py)
Output: data/features_all_cities.csv, ready to load into Hopsworks or a training script.

Usage:
    python feature_engineering.py
"""

import pandas as pd
import numpy as np


LAG_HOURS = [1, 3, 6, 12, 24]
ROLLING_WINDOWS = [3, 6, 24]


def add_time_features(df):
    df["hour"] = df["time"].dt.hour.astype("int64")
    df["day"] = df["time"].dt.day.astype("int64")
    df["month"] = df["time"].dt.month.astype("int64")
    df["weekday"] = df["time"].dt.weekday.astype("int64")
    df["is_weekend"] = df["weekday"].isin([5, 6]).astype("int64")
    df["season"] = df["month"].map(_month_to_season)
    return df


def _month_to_season(month):
    if month in (12, 1, 2):
        return "winter"
    if month in (3, 4, 5):
        return "spring"
    if month in (6, 7, 8):
        return "summer"
    return "autumn"


def add_lag_features(df):
    """Grouped by city so lags never cross city boundaries."""
    df = df.sort_values(["city", "time"])
    grouped = df.groupby("city")["aqi"]

    for lag in LAG_HOURS:
        df[f"aqi_lag_{lag}h"] = grouped.shift(lag)

    return df


def add_rolling_features(df):
    """Grouped by city so rolling stats never cross city boundaries."""
    df = df.sort_values(["city", "time"])
    grouped = df.groupby("city")["aqi"]

    for window in ROLLING_WINDOWS:
        # shift(1) first so the rolling window never includes the current (target-adjacent) row
        rolling = grouped.shift(1).groupby(df["city"]).rolling(window)
        df[f"aqi_roll_mean_{window}h"] = rolling.mean().reset_index(level=0, drop=True)
        df[f"aqi_roll_max_{window}h"] = rolling.max().reset_index(level=0, drop=True)
        df[f"aqi_roll_min_{window}h"] = rolling.min().reset_index(level=0, drop=True)
        df[f"aqi_roll_std_{window}h"] = rolling.std().reset_index(level=0, drop=True)

    return df


def add_derived_features(df):
    df["pm_ratio"] = df["pm2_5"] / df["pm10"].replace(0, np.nan)
    df["wind_pollution_interaction"] = df["wind_speed_10m"] * df["pm2_5"]

    df = df.sort_values(["city", "time"])
    df["aqi_change_rate"] = df.groupby("city")["aqi"].diff()
    df["temp_diff"] = df.groupby("city")["temperature_2m"].diff()
    df["humidity_diff"] = df.groupby("city")["relative_humidity_2m"].diff()

    return df


def add_targets(df):
    """
    Prediction targets: AQI 24h, 48h, and 72h ahead.
    Computed with a NEGATIVE shift (i.e. looking forward), grouped by city so the
    label for the last rows of one city never comes from the next city's early rows.
    """
    df = df.sort_values(["city", "time"])
    grouped = df.groupby("city")["aqi"]

    df["target_aqi_24h"] = grouped.shift(-24)
    df["target_aqi_48h"] = grouped.shift(-48)
    df["target_aqi_72h"] = grouped.shift(-72)

    return df


def build_features(df, drop_incomplete=True):
    """
    drop_incomplete=True (training):
        drops rows missing enough lag history OR missing a future target -
        both are required to train on a row.
    drop_incomplete=False (live/inference):
        keeps every row. Live rows will naturally have empty target_aqi_* columns
        since the future hasn't happened yet - that's fine, we only need lags for prediction.
    """
    df = df.copy()
    df["time"] = pd.to_datetime(df["time"])

    df = add_time_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_derived_features(df)
    df = add_targets(df)

    if drop_incomplete:
        # rows without enough history for lags, or without a future target yet, aren't usable for training
        df = df.dropna(subset=[f"aqi_lag_{max(LAG_HOURS)}h", "target_aqi_72h"])

    return df.reset_index(drop=True)


def main():
    raw = pd.read_csv("../data/raw_all_cities.csv")
    features = build_features(raw)
    features.to_csv("../data/features_all_cities.csv", index=False)
    print(f"Built {len(features)} feature rows across {features['city'].nunique()} cities.")
    print(f"Columns: {list(features.columns)}")


if __name__ == "__main__":
    main()