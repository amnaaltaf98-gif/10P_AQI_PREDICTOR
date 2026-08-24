"""
Builds a model-ready feature row for LIVE predictions - the same feature set
train.py trains on, computed from recent raw history instead of Hopsworks'
full offline dataset.

Why this needs to exist: train.py's model expects ~50 features built in two
layers:
    Layer 1 (feature_engineering.py) - already computed for every row stored
        in Hopsworks: time features, lag_1h-24h, rolling_3h/6h/24h, pm_ratio,
        wind_pollution_interaction, aqi_change_rate, temp_diff, humidity_diff.
    Layer 2 (train.py's engineer_advanced_features) - computed fresh at
        TRAINING time only, never stored: cyclical encodings, cross-city
        spatial transport, ewma, roll_48h/72h, lag_48h/168h, momentum diffs,
        dispersion/stagnation indices, climatology.

The live pipeline (fetch_live_data.py) only ever produced layer 1. A model
trained on both layers can't predict anything without both being computed
the same way at serving time too - that's what this module does.

Usage (called by the FastAPI service, not run directly):
    from inference_features import build_prediction_features
    X, cities_order = build_prediction_features(recent_history_df, current_readings_df)
    preds = model.predict(X)
"""

import sys
import numpy as np
import pandas as pd

sys.path.append("../training_pipeline")
sys.path.append(".")

from feature_engineering import build_features
from train import (
    engineer_advanced_features,
    RAW_FEATURE_COLUMNS,
    NEW_ENGINEERED_COLUMNS,
    CITY_COORDS,
)

KNOWN_SEASONS = ["winter", "spring", "summer", "autumn"]

# need > 168h of history per city for the weekly lag to be non-null, plus
# margin for the 24h rolling windows computed on top of that
RECOMMENDED_HISTORY_HOURS = 24 * 10


def _month_to_season(m):
    if m in (12, 1, 2):
        return "winter"
    if m in (3, 4, 5):
        return "spring"
    if m in (6, 7, 8):
        return "summer"
    return "autumn"


def _add_calendar_columns(df):
    """Derives hour/day/month/weekday/is_weekend/season from 'time' if not already present."""
    df = df.copy()
    df["time"] = pd.to_datetime(df["time"])
    if "hour" not in df.columns:
        df["hour"] = df["time"].dt.hour.astype("int64")
    if "day" not in df.columns:
        df["day"] = df["time"].dt.day.astype("int64")
    if "month" not in df.columns:
        df["month"] = df["time"].dt.month.astype("int64")
    if "weekday" not in df.columns:
        df["weekday"] = df["time"].dt.weekday.astype("int64")
    if "is_weekend" not in df.columns:
        df["is_weekend"] = df["weekday"].isin([5, 6]).astype("int64")
    if "season" not in df.columns:
        df["season"] = df["month"].map(_month_to_season)
    return df


def build_prediction_features(recent_history_df, current_readings_df):
    """
    recent_history_df: recent multi-city RAW readings (city, time, aqi, and
        all raw weather/pollutant columns) - ideally RECOMMENDED_HISTORY_HOURS
        worth per city. Less is fine, it just means longer lag/rolling
        features (168h, 72h) will be NaN for that city, which XGBoost
        handles natively - not a hard failure, just reduced signal.
    current_readings_df: the just-fetched current readings for all cities
        (same raw schema), representing "now" - the row(s) we actually want
        a prediction for.

    Returns:
        X: DataFrame, one row per city, every feature the model expects,
           ready for model.predict(X)
        cities_order: list of city names in the same row order as X, so
           predictions can be mapped back to the city they belong to
    """
    combined = pd.concat([recent_history_df, current_readings_df], ignore_index=True)
    combined = combined.drop_duplicates(subset=["city", "time"], keep="last")
    combined = combined.sort_values(["city", "time"]).reset_index(drop=True)
    combined = _add_calendar_columns(combined)

    # Live readings legitimately have None for dust/aerosol_optical_depth
    # (OpenWeather doesn't provide them). Mixing None with floats during the
    # merge can leave a column as 'object' dtype instead of numeric, which
    # XGBoost correctly refuses to train/predict on. Force every non-key
    # column to proper numeric (None/object -> NaN, which the model handles
    # natively).
    non_numeric_cols = {"time", "city", "season"}
    for col in combined.columns:
        if col not in non_numeric_cols:
            combined[col] = pd.to_numeric(combined[col], errors="coerce")

    # Layer 1: same lag/rolling/derived features as the stored Hopsworks data.
    # drop_incomplete=False - we want the CURRENT row even though it has no
    # future target (there is no future target, we're predicting it).
    layer1 = build_features(combined, drop_incomplete=False)

    # Layer 2: the same features train.py adds on top at training time.
    layer2 = engineer_advanced_features(layer1)

    # Climatology: unlike training (which must compute this from the TRAIN
    # split only to avoid leakage), at pure prediction time there's no
    # leakage concern - the future is genuinely unknown, so we can safely
    # use ALL available history.
    clim = (
        layer2.groupby(["city", "month", "hour"])["aqi"]
        .mean()
        .rename("aqi_climatology")
        .reset_index()
    )
    city_fallback = layer2.groupby("city")["aqi"].mean().rename("aqi_climatology_fallback").reset_index()

    latest_time = pd.to_datetime(current_readings_df["time"]).max()
    latest_rows = layer2[layer2["time"] == latest_time].copy()

    if latest_rows.empty:
        raise ValueError(
            f"No rows found at the latest timestamp {latest_time} after merging - "
            "check that current_readings_df's 'time' values match what's in the combined data."
        )

    latest_rows = latest_rows.merge(clim, on=["city", "month", "hour"], how="left")
    latest_rows = latest_rows.merge(city_fallback, on="city", how="left")
    latest_rows["aqi_climatology"] = latest_rows["aqi_climatology"].fillna(latest_rows["aqi_climatology_fallback"])
    latest_rows = latest_rows.drop(columns=["aqi_climatology_fallback"])

    feature_cols = [c for c in RAW_FEATURE_COLUMNS if c in latest_rows.columns] + NEW_ENGINEERED_COLUMNS + ["aqi_climatology"]
    feature_cols = list(dict.fromkeys(feature_cols))

    missing = [c for c in feature_cols if c not in latest_rows.columns]
    if missing:
        raise ValueError(f"Missing expected feature columns at prediction time: {missing}")

    X = latest_rows[feature_cols].copy()

    # Lock in the SAME categories used at training time. XGBoost with
    # enable_categorical=True needs consistent category coding - an unseen
    # or differently-ordered category set at predict time can cause the
    # model to misinterpret which city/season a row belongs to.
    known_cities = list(CITY_COORDS.keys())
    X["city"] = pd.Categorical(X["city"].astype(str), categories=known_cities)
    X["season"] = pd.Categorical(X["season"].astype(str), categories=KNOWN_SEASONS)

    cities_order = latest_rows["city"].tolist()

    return X.reset_index(drop=True), cities_order