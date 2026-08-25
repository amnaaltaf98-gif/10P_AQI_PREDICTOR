# Pearls AQI Predictor

Serverless, end-to-end ML system that forecasts AQI 24h / 48h / 72h ahead for 10 major Pakistani cities:
Karachi, Lahore, Islamabad, Rawalpindi, Faisalabad, Multan, Peshawar, Quetta, Hyderabad, Sialkot.

## Architecture

```
Open-Meteo Archive (historical weather + air quality, 2+ years, free, no key)
        |
feature_engineering.py  (lag/rolling/time/derived features, no cross-city leakage)
        |
Hopsworks Feature Store
        |
train.py  (XGBoost, compared against a persistence baseline)
        |
Hopsworks Model Registry
        |
FastAPI (/predict) <---- hourly live pull from OpenWeather Air Pollution API
        |
Streamlit Dashboard (Live AQI, Forecast, EDA, SHAP explainability, alerts)
```

Automation via GitHub Actions:
- **Hourly**: `feature_pipeline/fetch_live_data.py` pulls current pollutant readings and pushes to Hopsworks.
- **Daily**: `training_pipeline/train.py` retrains on the latest data and registers the best model.

## Folder structure

```
AQI-Predictor/
  data/                   raw + feature CSVs (gitignored except .gitkeep)
  feature_pipeline/       config, AQI math, backfill, live fetch, feature engineering
  training_pipeline/      train.py (model comparison + SHAP)
  models/                 saved .pkl models + SHAP plots
  fastapi/                prediction API
  streamlit/              dashboard
  .github/workflows/      hourly + daily automation
  notebooks/              EDA notebooks (add your own)
```

## Setup

1. For the Streamlit dashboard, create a Python 3.11 virtual environment and install the minimal dependencies:
   ```
   pip install -r requirements.txt
   ```

The dashboard can run without Hopsworks or FastAPI. It reads the tracked `data/live_buffer.csv` fallback for the current AQI and historical views. Forecasts are shown only when `API_BASE_URL` is configured with the URL of a separately deployed FastAPI service.

The feature pipeline, model training, and FastAPI service use additional packages and should run in a separate environment. Do not add those packages to the Streamlit Cloud `requirements.txt`; that makes deployment depend on build-heavy and Python-version-sensitive libraries.
2. Copy `.env.example` to `.env` and fill in:
   - `OPENWEATHER_API_KEY` (free tier is fine, needed only for hourly live data)
        - `HOPSWORKS_API_KEY` and `HOPSWORKS_PROJECT_NAME` (create a free project at hopsworks.ai; needed only by the pipeline)
        - `API_BASE_URL` (optional; the deployed FastAPI URL used for forecasts)
3. Add the same three values as GitHub repo secrets (Settings -> Secrets and variables -> Actions) so the workflows can run.

## Run order (first time)

```bash
cd feature_pipeline
python backfill_history.py       # ~2.5 years of hourly data for all 10 cities, takes a few minutes
python feature_engineering.py    # builds lag/rolling/time features + targets

cd ../training_pipeline
python train.py                  # trains and compares models, saves best + SHAP plots

cd ../fastapi
uvicorn main:app --reload        # optional prediction API on localhost:8000

cd ../streamlit
streamlit run app.py             # dashboard on localhost:8501
```

After the first backfill, the hourly/daily GitHub Actions workflows keep everything current automatically.

## Data sources

- **Open-Meteo Archive API** and **Open-Meteo Air Quality API**: historical weather + pollutants, free, no key, unlimited range. Used for backfill so history and later forecasts share the same format.
- **OpenWeather Air Pollution API**: current pollutant readings, used only for the hourly live pipeline.
- AQI itself is computed from PM2.5/PM10 using standard EPA breakpoints (`feature_pipeline/aqi_utils.py`), so historical and live values are calculated identically.

## Notes on the feature engineering leakage fix

Every lag, rolling, and target column is computed with `.groupby("city")` before any shift/rolling operation.
Sorting the full dataframe by time alone and then computing lags naively lets the last row of one city
leak into the first "lag" of the next city once cities are stacked. This was tested in
`feature_pipeline/feature_engineering.py` and confirmed fixed: see the module docstring for the reasoning.
