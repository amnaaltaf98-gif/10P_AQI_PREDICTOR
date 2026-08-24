"""
Pearls AQI Predictor - Prediction API

Endpoints:

    GET /health
    GET /cities
    GET /predict
    GET /predict/{city}

Run locally from the project root:

    uvicorn fastapi.main:app --reload
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException


# =========================================================
# Deployment-safe project paths
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

FEATURE_PIPELINE_DIR = (
    BASE_DIR / "feature_pipeline"
)

TRAINING_PIPELINE_DIR = (
    BASE_DIR / "training_pipeline"
)

if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

if str(FEATURE_PIPELINE_DIR) not in sys.path:
    sys.path.append(str(FEATURE_PIPELINE_DIR))

if str(TRAINING_PIPELINE_DIR) not in sys.path:
    sys.path.append(str(TRAINING_PIPELINE_DIR))


from config import CITIES

from fetch_live_data import (
    fetch_all_cities_raw
)

from inference_features import (
    build_prediction_features,
    RECOMMENDED_HISTORY_HOURS
)


app = FastAPI(
    title="Pearls AQI Predictor API"
)


# =========================================================
# Deployment-safe data/model paths
# =========================================================

LIVE_BUFFER_PATH = (
    BASE_DIR /
    "data" /
    "live_buffer.csv"
)

MODEL_PATHS = {

    "24h": (
        BASE_DIR /
        "models" /
        "target_aqi_24h_delta_xgb.pkl"
    ),

    "48h": (
        BASE_DIR /
        "models" /
        "target_aqi_48h_delta_xgb.pkl"
    ),

    "72h": (
        BASE_DIR /
        "models" /
        "target_aqi_72h_delta_xgb.pkl"
    ),

}


# PREDICT_DELTA was False for the currently saved models.
PREDICT_DELTA = False


_models = {}


# =========================================================
# MODEL LOADING
# =========================================================

def get_model(horizon):

    if horizon not in _models:

        path = MODEL_PATHS[horizon]

        if not path.exists():

            raise HTTPException(
                status_code=503,
                detail=(
                    f"Model for {horizon} not found "
                    f"at {path}"
                )
            )

        _models[horizon] = joblib.load(
            path
        )

    return _models[horizon]


# =========================================================
# HISTORY LOADING
# =========================================================

def load_recent_history():

    if not LIVE_BUFFER_PATH.exists():

        raise HTTPException(

            status_code=503,

            detail=(
                f"{LIVE_BUFFER_PATH} not found. "
                "Run fetch_live_data.py at least once "
                "before predictions can work."
            )

        )

    history_df = pd.read_csv(
        LIVE_BUFFER_PATH
    )

    history_df["time"] = pd.to_datetime(
        history_df["time"],
        format="mixed"
    )

    return history_df


# =========================================================
# PREDICTION LOGIC
# =========================================================

def run_predictions():

    """
    Fetches current data,
    builds features,
    predicts all 3 horizons
    for all cities.
    """

    current_readings_df = (
        fetch_all_cities_raw()
    )

    if current_readings_df.empty:

        raise HTTPException(
            status_code=502,
            detail=(
                "Could not fetch current readings "
                "for any city."
            )
        )


    recent_history_df = (
        load_recent_history()
    )


    history_span_hours = (

        pd.to_datetime(
            current_readings_df["time"]
        ).max()

        -

        recent_history_df[
            "time"
        ].min()

    ).total_seconds() / 3600


    if (
        history_span_hours
        <
        RECOMMENDED_HISTORY_HOURS
    ):

        print(

            f"WARNING: only "
            f"{history_span_hours:.0f}h "
            f"of history available "
            f"(recommend "
            f"{RECOMMENDED_HISTORY_HOURS}h+)"
        )


    X, cities_order = (
        build_prediction_features(

            recent_history_df,

            current_readings_df

        )
    )


    current_aqi_by_city = dict(

        zip(

            current_readings_df["city"],

            current_readings_df["aqi"]

        )

    )


    results = {}


    for horizon in [

        "24h",

        "48h",

        "72h"

    ]:


        model = get_model(
            horizon
        )


        raw_preds = model.predict(
            X
        )


        if PREDICT_DELTA:


            current_aqi_array = np.array([

                current_aqi_by_city[c]

                for c in cities_order

            ])


            preds = np.clip(

                current_aqi_array
                +
                raw_preds,

                0,

                None

            )


        else:


            preds = np.clip(

                raw_preds,

                0,

                None

            )


        for city, pred in zip(

            cities_order,

            preds

        ):


            results.setdefault(

                city,

                {}

            )[horizon] = round(

                float(pred),

                1

            )


    return (

        results,

        current_aqi_by_city

    )


# =========================================================
# HEALTH
# =========================================================

@app.get("/health")
def health():

    models_present = {

        h: path.exists()

        for h, path

        in MODEL_PATHS.items()

    }


    buffer_present = (
        LIVE_BUFFER_PATH.exists()
    )


    return {

        "status": (
            "ok"

            if (
                all(
                    models_present.values()
                )
                and
                buffer_present
            )

            else
            "degraded"
        ),

        "models_present":
            models_present,

        "live_buffer_present":
            buffer_present,

    }


# =========================================================
# CITIES
# =========================================================

@app.get("/cities")
def list_cities():

    return {

        "cities":
            sorted(
                CITIES.keys()
            )

    }


# =========================================================
# ALL PREDICTIONS
# =========================================================

@app.get("/predict")
def predict_all():

    forecasts, current_aqi_by_city = (
        run_predictions()
    )


    return {

        "generated_at":

            datetime.now(
                timezone.utc
            ).isoformat(),

        "predictions": [

            {

                "city":
                    city,

                "current_aqi":

                    round(
                        float(
                            current_aqi_by_city.get(
                                city,
                                0
                            )
                        ),
                        1
                    ),

                "forecast":
                    forecast

            }

            for city, forecast

            in forecasts.items()

        ]

    }


# =========================================================
# SINGLE CITY PREDICTION
# =========================================================

@app.get("/predict/{city}")
def predict_one(city: str):

    if city not in CITIES:

        raise HTTPException(

            status_code=404,

            detail=(
                f"Unknown city "
                f"'{city}'. "
                "See /cities for "
                "valid cities."
            )

        )


    forecasts, current_aqi_by_city = (
        run_predictions()
    )


    if city not in forecasts:

        raise HTTPException(

            status_code=502,

            detail=(
                f"Prediction failed for "
                f"{city}"
            )

        )


    return {

        "city":
            city,

        "generated_at":

            datetime.now(
                timezone.utc
            ).isoformat(),

        "current_aqi":

            round(

                float(

                    current_aqi_by_city.get(
                        city,
                        0
                    )

                ),

                1

            ),

        "forecast":
            forecasts[city]

    }