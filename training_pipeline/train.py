"""
Phase 11-13 - Train, evaluate, and explain models.

Loads data/features_all_cities.csv, trains a global model per horizon
(24h / 48h / 72h) with `city` as a categorical feature, compares against
several baselines, and saves the best model + a SHAP summary plot.

Usage:
    python train.py
"""

import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

TARGETS = ["target_aqi_24h", "target_aqi_48h", "target_aqi_72h"]

FEATURE_COLUMNS = [
    "city", "hour", "day", "month", "weekday", "is_weekend", "season",
    "pm2_5", "pm10", "nitrogen_dioxide", "sulphur_dioxide", "carbon_monoxide", "ozone",
    "temperature_2m", "relative_humidity_2m", "surface_pressure",
    "wind_speed_10m", "wind_direction_10m", "cloud_cover", "precipitation",
    "aqi_lag_1h", "aqi_lag_3h", "aqi_lag_6h", "aqi_lag_12h", "aqi_lag_24h",
    "aqi_roll_mean_3h", "aqi_roll_mean_6h", "aqi_roll_mean_24h",
    "aqi_roll_max_3h", "aqi_roll_min_3h", "aqi_roll_std_24h",
    "pm_ratio", "wind_pollution_interaction", "aqi_change_rate",
    "temp_diff", "humidity_diff",
]

CATEGORICAL_COLUMNS = ["city", "season"]


def prepare_xy(df, target_col):
    data = df.dropna(subset=[target_col]).copy()
    X = data[FEATURE_COLUMNS].copy()
    y = data[target_col]

    for col in CATEGORICAL_COLUMNS:
        X[col] = X[col].astype("category")

    return X, y


def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true.replace(0, np.nan))) * 100
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "MAPE": mape}


def train_one_horizon(df, target_col):
    print(f"\n=== Training for {target_col} ===")
    X, y = prepare_xy(df, target_col)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Persistence baseline: predict "AQI stays the same as right now"
    baseline_pred = X_test["aqi_lag_1h"]
    results = {"Persistence baseline": evaluate(y_test, baseline_pred)}

    models = {
        "Ridge": Ridge(),
        "RandomForest": RandomForestRegressor(n_estimators=300, max_depth=12, random_state=42, n_jobs=-1),
        "XGBoost": XGBRegressor(
            n_estimators=400, max_depth=6, learning_rate=0.05,
            enable_categorical=True, tree_method="hist", random_state=42,
        ),
        "LightGBM": LGBMRegressor(
            n_estimators=400, max_depth=6, learning_rate=0.05,
            random_state=42, verbose=-1,
        ),
    }

    fitted_models = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        results[name] = evaluate(y_test, preds)
        fitted_models[name] = model
        print(f"{name}: {results[name]}")

    # pick best by RMSE among the real models (skip the baseline)
    best_name = min(
        (n for n in results if n != "Persistence baseline"),
        key=lambda n: results[n]["RMSE"],
    )
    best_model = fitted_models[best_name]
    print(f"Best model for {target_col}: {best_name}")

    return best_model, best_name, results, X_test, y_test


def save_shap_summary(model, X_test, target_col, model_name):
    explainer = shap.TreeExplainer(model) if model_name != "Ridge" else shap.LinearExplainer(model, X_test)
    shap_values = explainer.shap_values(X_test)

    plt.figure()
    shap.summary_plot(shap_values, X_test, show=False)
    plt.title(f"SHAP summary - {target_col}")
    plt.tight_layout()
    out_path = f"../models/shap_{target_col}.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved SHAP plot -> {out_path}")


def main():
    df = pd.read_csv("../data/features_all_cities.csv")
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time")  # time-ordered split, no shuffling across the train/test boundary

    summary = {}
    for target_col in TARGETS:
        best_model, best_name, results, X_test, y_test = train_one_horizon(df, target_col)
        joblib.dump(best_model, f"../models/{target_col}_{best_name}.pkl")
        save_shap_summary(best_model, X_test, target_col, best_name)
        summary[target_col] = {"best_model": best_name, "metrics": results[best_name]}

    print("\n=== Final Summary ===")
    for target_col, info in summary.items():
        print(target_col, info)


if __name__ == "__main__":
    main()
