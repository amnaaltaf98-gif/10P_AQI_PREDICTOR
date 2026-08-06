"""
Shared config for the Pearls AQI Predictor.
Keep this as the single source of truth for cities and API settings.
"""

import os
from dotenv import load_dotenv

load_dotenv()  # reads the .env file in the project root and loads keys into os.environ

# 10 major Pakistani cities with coordinates (needed for Open-Meteo and OpenWeather)
CITIES = {
    "Karachi":    {"lat": 24.8607, "lon": 67.0011, "province": "Sindh"},
    "Lahore":     {"lat": 31.5497, "lon": 74.3436, "province": "Punjab"},
    "Islamabad":  {"lat": 33.6844, "lon": 73.0479, "province": "ICT"},
    "Rawalpindi": {"lat": 33.5651, "lon": 73.0169, "province": "Punjab"},
    "Faisalabad": {"lat": 31.4504, "lon": 73.1350, "province": "Punjab"},
    "Multan":     {"lat": 30.1575, "lon": 71.5249, "province": "Punjab"},
    "Quetta":     {"lat": 30.1798, "lon": 66.9750, "province": "Balochistan"},
    "Hyderabad":  {"lat": 25.3960, "lon": 68.3578, "province": "Sindh"},
    "Sialkot":    {"lat": 32.4945, "lon": 74.5229, "province": "Punjab"},
}

# API keys (set these as environment variables, never hardcode them)
OPENWEATHER_API_KEY = os.environ.get("OPENWEATHER_API_KEY", "")
HOPSWORKS_API_KEY = os.environ.get("HOPSWORKS_API_KEY", "")
HOPSWORKS_PROJECT_NAME = os.environ.get("HOPSWORKS_PROJECT_NAME", "")

# Open-Meteo endpoints (no key needed)
OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
OPEN_METEO_AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"

# OpenWeather endpoint (needs API key) - used for hourly LIVE updates only
OPENWEATHER_AIR_POLLUTION_URL = "https://api.openweathermap.org/data/2.5/air_pollution"

# Hopsworks feature group settings
FEATURE_GROUP_NAME = "aqi_features"
FEATURE_GROUP_VERSION = 1

# Historical backfill window (Phase 9)
BACKFILL_START_DATE = "2024-01-01"
BACKFILL_END_DATE = "2026-07-30"  # keep a day behind "today" to avoid incomplete data

# EPA breakpoints for PM2.5 -> AQI conversion (24-hr average, µg/m3)
# Used when we need to compute AQI ourselves instead of relying on a provider's AQI field
PM25_BREAKPOINTS = [
    (0.0, 12.0, 0, 50),
    (12.1, 35.4, 51, 100),
    (35.5, 55.4, 101, 150),
    (55.5, 150.4, 151, 200),
    (150.5, 250.4, 201, 300),
    (250.5, 350.4, 301, 400),
    (350.5, 500.4, 401, 500),
]

PM10_BREAKPOINTS = [
    (0, 54, 0, 50),
    (55, 154, 51, 100),
    (155, 254, 101, 150),
    (255, 354, 151, 200),
    (355, 424, 201, 300),
    (425, 504, 301, 400),
    (505, 604, 401, 500),
]