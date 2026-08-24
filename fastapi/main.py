"""
Pearls AQI Predictor - Prediction API

Endpoints:
    GET /health              - service + model status check
    GET /cities               - list of supported cities
    GET /predict               - forecast for ALL cities
    GET /predict/{city}         - forecast for one city

How a prediction actually happens:
    1. Fetch CURRENT readings for all 9 cities (same fetch logic the hourly
       pipeline uses - Open-Meteo weather + OpenWeather pollutants)
    2. Load RECENT HISTORY from data/live_buffer.csv (needed for lag/rolling/
       weekly features - a single current reading alone isn't enough)
    3. Run both feature layers via feature_pipeline/inference_features.py
       (same features the model was trained on - this is what makes the
       prediction valid, not just runnable)
    4. Load the 3 trained XGBoost models (24h/48h/72h) and predict

Run locally:
    cd fastapi
    uvicorn main:app --reload
"""

import os
import sys
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException

sys.path.append("../feature_pipeline")
sys.path.append("../training_pipeline")

from config import CITIES
from fetch_live_data import fetch_all_cities_raw
from inference_features import build_prediction_features, RECOMMENDED_HISTORY_HOURS

app = FastAPI(title="Pearls AQI Predictor API")

LIVE_BUFFER_PATH = "../data/live_buffer.csv"

MODEL_PATHS = {
    "24h": "../models/target_aqi_24h_delta_xgb.pkl",
    "48h": "../models/target_aqi_48h_delta_xgb.pkl",
    "72h": "../models/target_aqi_72h_delta_xgb.pkl",
}

# PREDICT_DELTA was False for the models currently saved - they predict
# absolute AQI directly. If train.py's PREDICT_DELTA toggle ever changes,
# this needs to change with it (and current_aqi would need adding back to
# the prediction instead of just clipping).
PREDICT_DELTA = False

_models = {}


def get_model(horizon):
    if horizon not in _models:
        path = MODEL_PATHS[horizon]
        if not os.path.exists(path):
            raise HTTPException(status_code=503, detail=f"Model for {horizon} not found at {path} - train it first.")
        _models[horizon] = joblib.load(path)
    return _models[horizon]


def load_recent_history():
    if not os.path.exists(LIVE_BUFFER_PATH):
        raise HTTPException(
            status_code=503,
            detail=(
                f"{LIVE_BUFFER_PATH} not found. Run fetch_live_data.py at least once "
                "(or restore it from git) before predictions can work - it's the recent "
                "history the model needs for lag/rolling/weekly features."
            ),
        )
    history_df = pd.read_csv(LIVE_BUFFER_PATH)
    history_df["time"] = pd.to_datetime(history_df["time"], format="mixed")
    return history_df


def run_predictions():
    """Fetches current data, builds features, predicts all 3 horizons for all cities."""
    current_readings_df = fetch_all_cities_raw()
    if current_readings_df.empty:
        raise HTTPException(status_code=502, detail="Could not fetch current readings for any city.")

    recent_history_df = load_recent_history()

    history_span_hours = (
        pd.to_datetime(current_readings_df["time"]).max() - recent_history_df["time"].min()
    ).total_seconds() / 3600
    if history_span_hours < RECOMMENDED_HISTORY_HOURS:
        print(
            f"WARNING: only {history_span_hours:.0f}h of history available "
            f"(recommend {RECOMMENDED_HISTORY_HOURS}h+) - weekly/longer-range "
            "features will be degraded (NaN) for some or all cities until the "
            "buffer has accumulated more history."
        )

    X, cities_order = build_prediction_features(recent_history_df, current_readings_df)

    current_aqi_by_city = dict(zip(current_readings_df["city"], current_readings_df["aqi"]))

    results = {}
    for horizon in ["24h", "48h", "72h"]:
        model = get_model(horizon)
        raw_preds = model.predict(X)

        if PREDICT_DELTA:
            current_aqi_array = np.array([current_aqi_by_city[c] for c in cities_order])
            preds = np.clip(current_aqi_array + raw_preds, 0, None)
        else:
            preds = np.clip(raw_preds, 0, None)

        for city, pred in zip(cities_order, preds):
            results.setdefault(city, {})[horizon] = round(float(pred), 1)

    return results, current_aqi_by_city


@app.get("/health")
def health():
    models_present = {h: os.path.exists(p) for h, p in MODEL_PATHS.items()}
    buffer_present = os.path.exists(LIVE_BUFFER_PATH)
    return {
        "status": "ok" if all(models_present.values()) and buffer_present else "degraded",
        "models_present": models_present,
        "live_buffer_present": buffer_present,
    }


@app.get("/cities")
def list_cities():
    return {"cities": sorted(CITIES.keys())}


@app.get("/predict")
def predict_all():
    forecasts, current_aqi_by_city = run_predictions()
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "predictions": [
            {"city": city, "current_aqi": round(float(current_aqi_by_city.get(city, 0)), 1), "forecast": forecast}
            for city, forecast in forecasts.items()
        ],
    }


@app.get("/predict/{city}")
def predict_one(city: str):
    if city not in CITIES:
        raise HTTPException(status_code=404, detail=f"Unknown city '{city}'. See /cities for the valid list.")

    forecasts, current_aqi_by_city = run_predictions()

    if city not in forecasts:
        raise HTTPException(status_code=502, detail=f"Prediction failed for {city} - check server logs.")

    return {
        "city": city,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "current_aqi": round(float(current_aqi_by_city.get(city, 0)), 1),
        "forecast": forecasts[city],
    }