"""Serverless-Ready Daily Retraining Pipeline for AQI Forecasting with Spatial Advection.

Predicts DELTA AQI (AQI_future - AQI_current).
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.append("../feature_pipeline")

LOCAL_FALLBACK_PATH = "../data/features_all_cities.csv"
LIVE_ACCUMULATED_PATH = "../data/features_live_accumulated.csv"

TARGETS = ["target_aqi_24h", "target_aqi_48h", "target_aqi_72h"]

# Toggle A: predict absolute AQI directly (False) vs the delta from current AQI (True).
PREDICT_DELTA = False

# Toggle B: None = use all available history. An integer = only train on the most recent N days.
USE_LAST_N_DAYS = None

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
    "upstream_smog_drift", "upstream_aqi_diff",
]

CATEGORICAL_COLUMNS = ["city", "season"]

XGB_CONFIG = {
    "n_estimators": 3000, "learning_rate": 0.02, "max_depth": 7,
    "subsample": 0.8, "colsample_bytree": 0.8,
    "enable_categorical": True, "tree_method": "hist",
    "early_stopping_rounds": 50, "eval_metric": "rmse",
    "n_jobs": -1, "random_state": 42,
}

CITY_COORDS = {
    "Karachi":    (24.8607, 67.0011),
    "Lahore":     (31.5497, 74.3436),
    "Islamabad":  (33.6844, 73.0479),
    "Rawalpindi": (33.5651, 73.0169),
    "Faisalabad": (31.4504, 73.1350),
    "Multan":     (30.1575, 71.5249),
    "Quetta":     (30.1798, 66.9750),
    "Hyderabad":  (25.3960, 68.3578),
    "Sialkot":    (32.4945, 74.5229),
}

def compute_spatial_advection_features(df):
    if "time" not in df.columns or "city" not in df.columns or "aqi" not in df.columns:
        return df

    aqi_pivot = df.pivot(index="time", columns="city", values="aqi")
    u_pivot = df.pivot(index="time", columns="city", values="wind_u")
    v_pivot = df.pivot(index="time", columns="city", values="wind_v")

    cities = [c for c in df["city"].unique() if c in CITY_COORDS]

    city_pairs = {}
    for c_target in cities:
        lat1, lon1 = CITY_COORDS[c_target]
        for c_source in cities:
            if c_target == c_source:
                continue
            lat2, lon2 = CITY_COORDS[c_source]
            d_lat = lat1 - lat2
            d_lon = lon1 - lon2
            dist = np.sqrt(d_lat**2 + d_lon**2)
            if dist > 0:
                north_dir = d_lat / dist
                east_dir = d_lon / dist
                weight = 1.0 / (dist + 0.1)
                city_pairs[(c_target, c_source)] = (east_dir, north_dir, weight)

    drift_df = pd.DataFrame(index=aqi_pivot.index)

    for c_target in cities:
        target_drift = pd.Series(0.0, index=aqi_pivot.index)
        total_weight = pd.Series(1e-5, index=aqi_pivot.index)
        for c_source in cities:
            if c_target == c_source or (c_target, c_source) not in city_pairs:
                continue
            east_dir, north_dir, dist_wt = city_pairs[(c_target, c_source)]
            su = u_pivot[c_source]
            sv = v_pivot[c_source]
            alignment = np.maximum(0, su * east_dir + sv * north_dir)
            transport_wt = alignment * dist_wt
            target_drift += aqi_pivot[c_source] * transport_wt
            total_weight += transport_wt
        drift_df[c_target] = target_drift / total_weight

    drift_long = (
        drift_df.unstack()
        .reset_index()
        .rename(columns={"level_0": "city", 0: "upstream_smog_drift"})
    )

    df = df.merge(drift_long, on=["time", "city"], how="left")
    df["upstream_aqi_diff"] = df["upstream_smog_drift"] - df["aqi"]
    return df


def engineer_advanced_features(df):
    df = df.copy()
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])
        df = df.sort_values(["time", "city"])

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12.0)
    df["day_of_week_sin"] = np.sin(2 * np.pi * df["weekday"] / 7.0)
    df["day_of_week_cos"] = np.cos(2 * np.pi * df["weekday"] / 7.0)

    rad = np.radians(df["wind_direction_10m"])
    df["wind_u"] = -df["wind_speed_10m"] * np.sin(rad)
    df["wind_v"] = -df["wind_speed_10m"] * np.cos(rad)

    df = compute_spatial_advection_features(df)

    if "city" in df.columns and "aqi" in df.columns:
        df["aqi_ewma_6h"] = df.groupby("city")["aqi"].transform(lambda x: x.ewm(span=6).mean())
        df["aqi_roll_mean_48h"] = df.groupby("city")["aqi"].transform(lambda x: x.rolling(48, min_periods=1).mean())
        df["aqi_roll_max_72h"] = df.groupby("city")["aqi"].transform(lambda x: x.rolling(72, min_periods=1).max())
        df["aqi_roll_std_48h"] = df.groupby("city")["aqi"].transform(lambda x: x.rolling(48, min_periods=1).std()).fillna(0)
        df["aqi_trend_24h"] = df["aqi"] - df["aqi_lag_24h"]
        df["pressure_trend_12h"] = (
            df["surface_pressure"] - df.groupby("city")["surface_pressure"].shift(12)
        ).fillna(0)
        df["temp_inversion_proxy"] = df["temperature_2m"] / (df["wind_speed_10m"] + 0.1)

        for lag in [1, 3, 6, 12, 24]:
            df[f"aqi_diff_{lag}h"] = df.groupby("city")["aqi"].diff(lag)

        df["aqi_lag_48h"] = df.groupby("city")["aqi"].shift(48)
        df["aqi_lag_168h"] = df.groupby("city")["aqi"].shift(168)

        df["dispersion_index"] = df["wind_speed_10m"] / (df["relative_humidity_2m"] + 1e-5)
        df["stagnation_index"] = df["surface_pressure"] / (df["wind_speed_10m"] + 1e-5)
        df["pm25_humidity_interaction"] = df["pm2_5"] * df["relative_humidity_2m"]

    for col in CATEGORICAL_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype("category")

    return df


NEW_ENGINEERED_COLUMNS = [
    "hour_sin", "hour_cos", "month_sin", "month_cos",
    "day_of_week_sin", "day_of_week_cos",
    "wind_u", "wind_v",
    "aqi_ewma_6h", "aqi_roll_mean_48h", "aqi_roll_max_72h", "aqi_roll_std_48h",
    "aqi_trend_24h", "pressure_trend_12h", "temp_inversion_proxy",
    "upstream_smog_drift", "upstream_aqi_diff",
    "aqi_diff_1h", "aqi_diff_3h", "aqi_diff_6h", "aqi_diff_12h", "aqi_diff_24h",
    "aqi_lag_48h", "aqi_lag_168h",
    "dispersion_index", "stagnation_index", "pm25_humidity_interaction",
]


def prepare_xy(df, target_col):
    data = df.dropna(subset=[target_col]).copy()
    data = engineer_advanced_features(data)
    data["target_delta"] = data[target_col] - data["aqi"]

    feature_cols = [c for c in RAW_FEATURE_COLUMNS if c in data.columns] + NEW_ENGINEERED_COLUMNS
    feature_cols = list(dict.fromkeys(feature_cols))

    X = data[feature_cols].copy()
    return X, data["target_delta"], data[target_col], data["aqi"]


def evaluate(y_true, y_pred):
    return {"MAE": mean_absolute_error(y_true, y_pred),
            "RMSE": mean_squared_error(y_true, y_pred) ** 0.5,
            "R2": r2_score(y_true, y_pred)}


def add_climatology_feature(X_train, current_aqi_train, X_test):
    train_keys = pd.DataFrame({
        "city": X_train["city"].astype(str).values,
        "month": X_train["month"].values,
        "hour": X_train["hour"].values,
        "aqi": current_aqi_train.values,
    })
    clim = train_keys.groupby(["city", "month", "hour"])["aqi"].mean().rename("aqi_climatology").reset_index()
    city_fallback = train_keys.groupby("city")["aqi"].mean().rename("aqi_climatology_fallback").reset_index()

    def apply_clim(X_subset):
        keys = pd.DataFrame({
            "city": X_subset["city"].astype(str).values,
            "month": X_subset["month"].values,
            "hour": X_subset["hour"].values,
        })
        merged = keys.merge(clim, on=["city", "month", "hour"], how="left")
        merged = merged.merge(city_fallback, on="city", how="left")
        return merged["aqi_climatology"].fillna(merged["aqi_climatology_fallback"]).values

    X_train = X_train.copy()
    X_test = X_test.copy()
    X_train["aqi_climatology"] = apply_clim(X_train)
    X_test["aqi_climatology"] = apply_clim(X_test)
    return X_train, X_test


def train_serverless_horizon(df, target_col):
    X, y_delta, y_actual, current_aqi = prepare_xy(df, target_col)
    
    # 80/20 train-test split ensures validation on most recent period
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_test_actual = y_actual.iloc[split_idx:]
    current_aqi_test = current_aqi.iloc[split_idx:]
    current_aqi_train = current_aqi.iloc[:split_idx]

    X_train, X_test = add_climatology_feature(X_train, current_aqi_train, X_test)

    if PREDICT_DELTA:
        y_train = y_delta.iloc[:split_idx]
    else:
        y_train = y_actual.iloc[:split_idx]

    # Val set for early stopping (last 10% of training data)
    val_cut = int(len(X_train) * 0.9)
    X_fit, X_val = X_train.iloc[:val_cut], X_train.iloc[val_cut:]
    y_fit, y_val = y_train.iloc[:val_cut], y_train.iloc[val_cut:]

    # Train XGBoost
    model = XGBRegressor(**XGB_CONFIG)
    model.fit(X_fit, y_fit, eval_set=[(X_val, y_val)], verbose=False)

    # Save a compact SHAP summary for the dashboard when the optional package
    # is available. Training and predictions do not depend on this plot.
    try:
        import shap
        import matplotlib.pyplot as plt

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_test, show=False, max_display=15)
        plt.tight_layout()
        plot_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", f"shap_{target_col}.png")
        plt.savefig(plot_path, dpi=120, bbox_inches="tight")
        plt.close()
    except Exception as exc:
        print(f"SHAP plot unavailable for {target_col}: {exc}")

    if PREDICT_DELTA:
        preds = np.clip(current_aqi_test + model.predict(X_test), 0, None)
    else:
        preds = np.clip(model.predict(X_test), 0, None)

    # Evaluate Model vs Persistence Baseline
    metrics = evaluate(y_test_actual, preds)
    persistence_metrics = evaluate(y_test_actual, current_aqi_test)
    
    print(f"    XGBoost    : R2={metrics['R2']:.4f}  RMSE={metrics['RMSE']:.2f}  MAE={metrics['MAE']:.2f}")
    print(f"    Persistence: R2={persistence_metrics['R2']:.4f}  RMSE={persistence_metrics['RMSE']:.2f}  MAE={persistence_metrics['MAE']:.2f}")

    return model, metrics


def load_data():
    try:
        import hopsworks
        from config import HOPSWORKS_API_KEY, HOPSWORKS_PROJECT_NAME, FEATURE_GROUP_NAME, FEATURE_GROUP_VERSION
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


import tempfile

def main():
    # 1. Ensure path resolution works regardless of working directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    sys.path.append(os.path.join(project_root, "feature_pipeline"))

    df = load_data()
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time")

    if USE_LAST_N_DAYS is not None:
        cutoff = df["time"].max() - pd.Timedelta(days=USE_LAST_N_DAYS)
        df = df[df["time"] >= cutoff].reset_index(drop=True)

    # 2. Connect to Hopsworks Model Registry
    try:
        import hopsworks
        from config import HOPSWORKS_API_KEY, HOPSWORKS_PROJECT_NAME
        project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY, project=HOPSWORKS_PROJECT_NAME)
        mr = project.get_model_registry()
        use_hopsworks_mr = True
    except Exception as e:
        print(f"Hopsworks Model Registry unavailable ({e}). Saving locally only.")
        use_hopsworks_mr = False

    models_dir = os.path.join(project_root, "models")
    os.makedirs(models_dir, exist_ok=True)

    for target_col in TARGETS:
        print(f"\nTraining for {target_col}...")
        model, metrics = train_serverless_horizon(df, target_col)
        
        file_name = f"{target_col}_delta_xgb.pkl"
        local_filepath = os.path.join(models_dir, file_name)
        
        # Save model locally
        joblib.dump(model, local_filepath, compress=3)

        # Register model in Hopsworks Model Registry
        if use_hopsworks_mr:
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_model_path = os.path.join(tmp_dir, file_name)
                joblib.dump(model, tmp_model_path, compress=3)
                
                hw_model = mr.python.create_model(
                    name=f"aqi_{target_col}_xgb",
                    metrics=metrics,
                    description=f"XGBoost model predicting {target_col}"
                )
                hw_model.save(tmp_model_path)
                print(f"Successfully pushed {target_col} model to Hopsworks Model Registry!")

if __name__ == "__main__":
    main()