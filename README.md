# Pearls AQI Predictor

Live demo: [Pearls AQI Predictor · Streamlit](https://10pearls-aqi-predictor-amna-altaf.streamlit.app/)

A production-style air quality forecasting system designed for major Pakistani cities. The project combines historical climate and pollutant data, engineered forecasting features, and machine learning models to estimate AQI at multiple future horizons. It delivers both a prediction API and an interactive dashboard for monitoring current conditions and forecasting the next 24, 48, and 72 hours.

## Overview

The AQI Predictor is built to answer a practical question: how polluted will the air become in the next few days, and how confident can that forecast be? The solution uses a complete ML pipeline, from data acquisition and feature engineering to model training and deployment.

At a high level, the system:

- pulls historical and live meteorological and air-quality data
- computes AQI from pollutant concentrations using EPA-style thresholds
- engineers lag, rolling, seasonal, and environmental features
- trains gradient boosting models for multi-horizon forecasting
- exposes predictions through a REST API
- presents live and forecasted AQI in a Streamlit dashboard

This is a full-stack forecasting project with a clear separation between data engineering, model training, inference, and user-facing visualization.

## Supported Cities

The project is configured for the following cities in Pakistan:

- Karachi
- Lahore
- Islamabad
- Rawalpindi
- Faisalabad
- Multan
- Quetta
- Hyderabad
- Sialkot

## Key Features

### Multi-horizon forecasting
The model stack predicts AQI for:

- 24 hours ahead
- 48 hours ahead
- 72 hours ahead

### Feature-rich modeling
The feature pipeline builds a wide range of predictive signals, including:

- hourly, daily, and seasonal time features
- lagged AQI values
- rolling AQI statistics
- meteorological interactions
- pollutant ratios and environmental trend indicators
- city-aware spatial and drift-related features

### City-aware data integrity
The feature engineering logic is implemented to avoid cross-city leakage by grouping calculations by city before shift and rolling operations. This keeps temporal modeling realistic and prevents artifacts caused by mixing observations from different cities.

### Live AQI workflow
The project has a live data path that fetches current pollutant readings and merges them with recent historical coverage so inference can produce valid forecast features for the present moment.

### Interactive monitoring dashboard
The Streamlit interface lets users:

- inspect live AQI conditions
- view city-level historical trends
- explore forecast outputs
- review model explainability and feature behavior
- assess the current monitoring state in a visually rich dashboard

## Project Architecture

```text
Open-Meteo archive + live pollutant sources
        |
        v
feature_pipeline/
  - data acquisition
  - AQI calculations
  - feature engineering
  - live inference feature preparation
        |
        v
training_pipeline/
  - model training
  - validation and scoring
  - model export
        |
        v
models/
  - trained forecast models
        |
        v
fastapi/
  - prediction API
        |
        v
streamlit/
  - dashboard and analytics UI
```

## Repository Structure

```text
10P_AQI_PROJECT/
├── data/
│   ├── raw_all_cities.csv
│   ├── features_all_cities.csv
│   ├── live_buffer.csv
│   └── ...
├── feature_pipeline/
│   ├── aqi_utils.py
│   ├── audit_data_quality.py
│   ├── backfill_extend_earlier.py
│   ├── backfill_history.py
│   ├── config.py
│   ├── feature_engineering.py
│   ├── fetch_live_data.py
│   ├── inference_features.py
│   ├── reseed_buffer_from_hopsworks.py
│   └── upload_to_hopsworks.py
├── training_pipeline/
│   └── train.py
├── fastapi/
│   └── main.py
├── streamlit/
│   └── app.py
├── models/
│   └── metrics.json
├── notebooks/
├── requirements.txt
├── requirements-pipeline.txt
├── runtime.txt
├── README.md
└── .env
```

## Tech Stack

### Data and feature engineering
- Python
- pandas
- NumPy
- requests
- python-dotenv
- Open-Meteo APIs
- OpenWeather Air Pollution API

### Machine learning
- XGBoost
- scikit-learn
- joblib
- SHAP for explainability

### Application layer
- FastAPI
- Streamlit
- Plotly
- PyDeck

### MLOps and deployment support
- Hopsworks for feature storage and model registry workflows
- GitHub automation support for periodic updates

## Data Sources

The project uses a layered data strategy built around both historical and live information.

### Historical data
Historical weather and environmental data are pulled from Open-Meteo archives. This provides a consistent time-series foundation for backfilling long-running historical records and building the model feature set.

### Live data
Current pollutant readings are collected using the OpenWeather Air Pollution API. These observations are used to create fresh inference inputs and keep forecasting aligned with real-time conditions.

### AQI calculation
The project computes AQI values from pollutant measurements, especially PM2.5 and PM10, using EPA-style breakpoint logic. This ensures that the AQI values used in features, targets, and live inference are derived consistently across the pipeline.

## Model Training Workflow

The training pipeline is centered in [training_pipeline/train.py](training_pipeline/train.py). It performs the following tasks:

1. loads feature data for the supported cities
2. prepares training targets for 24h, 48h, and 72h horizons
3. engineers additional model-compatible features
4. evaluates models using validation metrics
5. trains XGBoost regressors for each forecast horizon
6. saves the trained models and evaluation outputs

The model configuration includes strong gradient boosting settings designed for tabular, time-aware regression tasks. Training is optimized for performance while maintaining interpretability and stability across multiple forecast horizons.

## Inference and API Layer

The REST API is implemented in [fastapi/main.py](fastapi/main.py). It provides endpoints for:

- service health checks
- listing supported cities
- generating all-city forecasts
- generating single-city forecasts

The API workflow follows a realistic production sequence:

1. fetch current readings for cities
2. load recent historical buffer data
3. generate inference features using the same logic as model training
4. run the trained XGBoost models for each forecast horizon
5. return AQI predictions in a clean JSON structure

This keeps the forecasting process aligned with the actual training methodology rather than relying on a simplified one-off prediction path.

## Dashboard

The dashboard in [streamlit/app.py](streamlit/app.py) is a modern analytical front end for the AQI predictor. It includes:

- live AQI summaries
- city-wise monitoring views
- forecast summaries for the next 24/48/72 hours
- historical trend exploration
- model explainability and interpretability views
- clean, responsive visual design

The app is designed to be both operationally useful and easy to navigate for users monitoring air quality across multiple cities.

## Local Setup

### Prerequisites

- Python 3.11
- pip
- access to the project environment

### 1. Create a virtual environment

```bash
python -m venv venv
```

On Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### 2. Install dependencies

For the dashboard and basic runtime features:

```bash
pip install -r requirements.txt
```

For the full data pipeline, training workflow, API, and model tooling:

```bash
pip install -r requirements-pipeline.txt
```

### 3. Configure environment variables

Create a root-level `.env` file and set the required values:

```env
OPENWEATHER_API_KEY=your_openweather_key
HOPSWORKS_API_KEY=your_hopsworks_key
HOPSWORKS_PROJECT_NAME=your_hopsworks_project
```

You may also configure an `API_BASE_URL` value if the dashboard should point to a deployed FastAPI service for forecast data.

## Running the Project

### Feature pipeline
From the project root:

```bash
cd feature_pipeline
python backfill_history.py
python feature_engineering.py
```

These steps build the historical dataset and generate engineered features used in training and inference.

### Model training

```bash
cd training_pipeline
python train.py
```

This trains the forecasting models and exports metrics and model artifacts for the active project.

### FastAPI service

```bash
cd fastapi
uvicorn main:app --reload
```

The API will be available locally on the default FastAPI port, typically:

```text
http://localhost:8000
```

### Streamlit dashboard

```bash
cd streamlit
streamlit run app.py
```

The dashboard typically opens at:

```text
http://localhost:8501
```

## Data Pipeline Workflow

The project is designed to support an end-to-end forecast lifecycle:

1. historical weather and pollution records are collected
2. AQI is computed and standardized
3. feature engineering creates time-aware prediction signals
4. XGBoost models are trained on the prepared dataset
5. live data is fetched and transformed into inference-ready features
6. forecasts are served through the API and displayed in the dashboard

This creates a reusable and maintainable forecasting foundation rather than a one-off notebook workflow.

## Production-Ready Characteristics

The project is structured to be practical for ongoing operations:

- modular separation of responsibilities across pipelines and app layers
- reproducible feature generation for training and inference
- model artifact persistence for deployment reuse
- API access for integration with other dashboards or services
- interactive visualization for operational monitoring

## Best Practices and Usage Notes

- Keep the data pipeline and training tools in a dedicated Python environment to avoid dependency conflicts.
- Keep the dashboard runtime lightweight and focused on display and interaction.
- Use the trained models consistently with the same feature-generation logic as the training pipeline.
- Monitor the live buffer and recent feature history to preserve forecasting quality over time.

## Summary

The Pearls AQI Predictor is a comprehensive air-quality forecasting project that connects environmental data, advanced feature engineering, and machine learning to deliver practical AQI forecasts for multiple cities in Pakistan. It combines robust modeling with a deployable API and a polished user-facing dashboard, making it suitable for operational air-quality monitoring and forward-looking environmental planning.
