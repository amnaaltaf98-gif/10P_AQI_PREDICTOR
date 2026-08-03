"""
Phase 16 - FastAPI Prediction Service.

Endpoints:
    GET /health
    GET /city            -> list supported cities
    GET /features/{city} -> latest feature row for a city (stubbed until Hopsworks is wired in)
    GET /predict/{city}  -> AQI forecast for 24h/48h/72h
    GET /model           -> which model is currently loaded per horizon

Run locally:
    uvicorn main:app --reload
"""

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

import sys
sys.path.append("../feature_pipeline")
from config import CITIES

app = FastAPI(title="Pearls AQI Predictor API")

MODEL_PATHS = {
    "24h": "../models/target_aqi_24h_model.pkl",
    "48h": "../models/target_aqi_48h_model.pkl",
    "72h": "../models/target_aqi_72h_model.pkl",
}

_loaded_models = {}


def get_model(horizon):
    if horizon not in _loaded_models:
        try:
            _loaded_models[horizon] = joblib.load(MODEL_PATHS[horizon])
        except FileNotFoundError:
            raise HTTPException(status_code=503, detail=f"Model for {horizon} not trained yet")
    return _loaded_models[horizon]


def get_latest_features(city):
    """
    Stub: replace with a real Hopsworks feature-store read once that's wired in.
    Should return a single-row DataFrame matching FEATURE_COLUMNS in train.py.
    """
    raise HTTPException(status_code=501, detail="Feature store lookup not implemented yet")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/city")
def list_cities():
    return {"cities": list(CITIES.keys())}


@app.get("/features/{city}")
def features(city: str):
    if city not in CITIES:
        raise HTTPException(status_code=404, detail="Unknown city")
    row = get_latest_features(city)
    return row.to_dict(orient="records")[0]


@app.get("/predict/{city}")
def predict(city: str):
    if city not in CITIES:
        raise HTTPException(status_code=404, detail="Unknown city")

    row = get_latest_features(city)
    forecast = {}
    for horizon in ["24h", "48h", "72h"]:
        model = get_model(horizon)
        forecast[horizon] = float(model.predict(row)[0])

    return {"city": city, "forecast": forecast}


@app.get("/model")
def model_info():
    return {h: p for h, p in MODEL_PATHS.items()}
