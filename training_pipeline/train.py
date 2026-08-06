"""
Serverless-Ready Daily Retraining Pipeline for AQI Forecasting.
Predicts DELTA AQI (AQI_future - AQI_current).
"""
import os
import sys
import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.append("../feature_pipeline")
from config import HOPSWORKS_API_KEY, HOPSWORKS_PROJECT_NAME, FEATURE_GROUP_NAME, FEATURE_GROUP_VERSION

LOCAL_FALLBACK_PATH = "../data/features_all_cities.csv"
LIVE_ACCUMULATED_PATH = "../data/features_live_accumulated.csv"

TARGETS = ["target_aqi_24h", "target_aqi_48h", "target_aqi_72h"]

RAW_FEATURE_COLUMNS = [
    "city", "hour", "day", "month", "weekday", "is_weekend", "season",
    "aqi", "pm2_5", "pm10", "nitrogen_dioxide", "sulphur_dioxide",
    "carbon_monoxide", "ozone", "dust", "aerosol_optical_depth",
    "temperature_2m", "relative_humidity_2m", "surface_pressure",
    "wind_speed_10m", "wind_direction_10m", "cloud_cover", "precipitation",
    "shortwave_radiation",
    "aqi_lag_1h", "aqi_lag_3h", "aqi_lag_6h", "aqi_lag_12h", "aqi_lag_24h",
    "aqi_roll_mean_3h", "aqi_roll_mean_6h", "aqi_roll_mean_24h",
    "aqi_roll_max_3h", "aqi_roll_max_6h", "aqi_roll_max_24h",
    "aqi_roll_min_3h", "aqi_roll_min_6h", "aqi_roll_min_24h",
    "aqi_roll_std_3h", "aqi_roll_std_6h", "aqi_roll_std_24h",
    "pm_ratio", "wind_pollution_interaction", "aqi_change_rate",
    "temp_diff", "humidity_diff",
]

CATEGORICAL_COLUMNS = ["city", "season"]

LGBM_CONFIG = {
    "n_estimators": 500, "learning_rate": 0.03, "max_depth": 6, "num_leaves": 31,
    "subsample": 0.8, "colsample_bytree": 0.8, "n_jobs": -1, "random_state": 42, "verbose": -1,
}

def engineer_advanced_features(df):
    df = df.copy()
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12.0)
    rad = np.radians(df["wind_direction_10m"])
    df["wind_u"] = -df["wind_speed_10m"] * np.sin(rad)
    df["wind_v"] = -df["wind_speed_10m"] * np.cos(rad)
    if "city" in df.columns and "aqi" in df.columns:
        df["aqi_ewma_6h"] = df.groupby("city")["aqi"].transform(lambda x: x.ewm(span=6).mean())
        df["aqi_roll_mean_48h"] = df.groupby("city")["aqi"].transform(lambda x: x.rolling(48, min_periods=1).mean())
        df["aqi_roll_max_72h"] = df.groupby("city")["aqi"].transform(lambda x: x.rolling(72, min_periods=1).max())
        df["aqi_roll_std_48h"] = df.groupby("city")["aqi"].transform(lambda x: x.rolling(48, min_periods=1).std()).fillna(0)
        df["aqi_trend_24h"] = df["aqi"] - df["aqi_lag_24h"]
        df["pressure_trend_12h"] = df["surface_pressure"] - df.groupby("city")["surface_pressure"].shift(12).fillna(0)
        df["temp_inversion_proxy"] = df["temperature_2m"] / (df["wind_speed_10m"] + 0.1)
    for col in CATEGORICAL_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype("category")
    return df

def prepare_xy(df, target_col):
    data = df.dropna(subset=[target_col]).copy()
    data = engineer_advanced_features(data)
    data["target_delta"] = data[target_col] - data["aqi"]
    new_cols = ["hour_sin","hour_cos","month_sin","month_cos","wind_u","wind_v",
        "aqi_ewma_6h","aqi_roll_mean_48h","aqi_roll_max_72h","aqi_roll_std_48h",
        "aqi_trend_24h","pressure_trend_12h","temp_inversion_proxy"]
    feature_cols = [c for c in RAW_FEATURE_COLUMNS if c in data.columns] + new_cols
    X = data[feature_cols].copy()
    return X, data["target_delta"], data[target_col], data["aqi"]

def evaluate(y_true, y_pred):
    return {"MAE": mean_absolute_error(y_true, y_pred),
            "RMSE": mean_squared_error(y_true, y_pred) ** 0.5,
            "R2": r2_score(y_true, y_pred)}

def train_serverless_horizon(df, target_col):
    X, y_delta, y_actual, current_aqi = prepare_xy(df, target_col)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train_delta = y_delta.iloc[:split_idx]
    y_test_actual = y_actual.iloc[split_idx:]
    current_aqi_test = current_aqi.iloc[split_idx:]
    model = LGBMRegressor(**LGBM_CONFIG)
    model.fit(X_train, y_train_delta)
    preds = np.clip(current_aqi_test + model.predict(X_test), 0, None)
    return model, evaluate(y_test_actual, preds)

def load_data():
    """Try Hopsworks first (the live, current source of truth), fall back to the local CSV."""
    try:
        import hopsworks
        print("Loading features from Hopsworks feature store...")
        project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY, project=HOPSWORKS_PROJECT_NAME)
        fs = project.get_feature_store()
        fg = fs.get_feature_group(name=FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)
        df = fg.read()
        print(f"Loaded {len(df)} rows from Hopsworks.")
        return df
    except Exception as e:
        print(f"Could not load from Hopsworks ({e}), falling back to local files...")
        df = pd.read_csv(LOCAL_FALLBACK_PATH)
        print(f"Loaded {len(df)} rows from {LOCAL_FALLBACK_PATH}.")

        if os.path.exists(LIVE_ACCUMULATED_PATH):
            live_df = pd.read_csv(LIVE_ACCUMULATED_PATH)
            print(f"Merging in {len(live_df)} additional rows from {LIVE_ACCUMULATED_PATH}...")
            df["time"] = pd.to_datetime(df["time"], format="mixed")
            live_df["time"] = pd.to_datetime(live_df["time"], format="mixed")
            df = pd.concat([df, live_df], ignore_index=True)
            df = df.drop_duplicates(subset=["city", "time"], keep="last")
            print(f"Combined total: {len(df)} rows.")

        return df

def main():
    os.makedirs("../models", exist_ok=True)
    df = load_data()
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time")
    for target_col in TARGETS:
        model, metrics = train_serverless_horizon(df, target_col)
        joblib.dump(model, f"../models/{target_col}_delta_lgbm.pkl", compress=3)
        print(f"{target_col} -> R2: {metrics['R2']:.4f}")

if __name__ == "__main__":
    main()