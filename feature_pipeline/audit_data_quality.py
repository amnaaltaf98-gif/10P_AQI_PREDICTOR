"""
Unified AQI Feature Store Audit, Gap Fixer, & City Negative Bias Diagnostic Tool.

This script:
1. Runs a full data quality audit (Gaps, Duplicates, Nulls, Ranges, Recomputed AQI, Categories).
2. Fixes issue #1: Re-indexes hourly time series per city so lag features (1h, 3h, 6h, etc.)
   use actual physical time instead of row position.
3. Performs city-level negative bias detection (Mean Error < 0 analysis) and residual plotting.
"""

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Fallback imports if running standalone
try:
    from config import CITIES, HOPSWORKS_API_KEY, HOPSWORKS_PROJECT_NAME, FEATURE_GROUP_NAME, FEATURE_GROUP_VERSION
    from aqi_utils import compute_aqi
except ImportError:
    CITIES = {
        "Karachi": (24.8607, 67.0011), "Lahore": (31.5497, 74.3436), 
        "Islamabad": (33.6844, 73.0479), "Rawalpindi": (33.5651, 73.0169), 
        "Faisalabad": (31.4504, 73.1350), "Multan": (30.1575, 71.5249), 
        "Quetta": (30.1798, 66.9750), "Hyderabad": (25.3960, 68.3578), 
        "Sialkot": (32.4945, 74.5229)
    }
    HOPSWORKS_API_KEY = None
    HOPSWORKS_PROJECT_NAME = None
    FEATURE_GROUP_NAME = "aqi_features"
    FEATURE_GROUP_VERSION = 1
    
    def compute_aqi(pm25, pm10):
        # Fallback simplified U.S. EPA AQI proxy for PM2.5
        if pd.isna(pm25): return np.nan
        return min(500, max(0, int(pm25 * 2.1)))

LOCAL_FALLBACK_PATH = "../data/features_all_cities.csv"

EXPECTED_RANGES = {
    "aqi": (0, 500),
    "pm2_5": (0, 1000),
    "pm10": (0, 1000),
    "temperature_2m": (-10, 55),
    "relative_humidity_2m": (0, 100),
    "surface_pressure": (800, 1050),
    "wind_speed_10m": (0, 150),
    "cloud_cover": (0, 100),
}


# ==========================================
# 1. DATA AUDIT & BACKFILL VERIFICATION
# ==========================================

def check_time_gaps(df):
    print("\n=== 1. TIME GAP CHECK & BACKFILL VERIFICATION ===")
    total_gap_hours = 0
    total_expected_hours = 0
    worst_gaps = []

    for city in sorted(df["city"].unique()):
        city_df = df[df["city"] == city].sort_values("time")
        times = city_df["time"]
        expected_span_hours = int((times.max() - times.min()).total_seconds() / 3600) + 1
        actual_rows = len(times)
        missing = expected_span_hours - actual_rows

        diffs = times.diff().dt.total_seconds().div(3600).dropna()
        gaps = diffs[diffs > 1]  # Jump > 1 hour is a gap

        total_gap_hours += missing
        total_expected_hours += expected_span_hours
        completeness = (actual_rows / expected_span_hours) * 100 if expected_span_hours else 0

        print(f"  {city:<12}: {actual_rows}/{expected_span_hours} hrs ({completeness:.1f}% complete) | "
              f"Gaps: {len(gaps)} | Max Gap: {gaps.max() if len(gaps) else 0:.0f}h")

        if len(gaps) > 0:
            worst_gaps.append((city, gaps.max(), len(gaps)))

    pct_missing = 100 * total_gap_hours / total_expected_hours if total_expected_hours else 0
    print(f"\n  Overall Completeness: {100 - pct_missing:.2f}% ({pct_missing:.2f}% hours missing)")
    if pct_missing > 2:
        print("  -> WARNING: High gap frequency. Standard .shift(N) creates mislabeled lag features.")
        print("     Use `fix_time_aware_lags()` to resample on precise time steps.")

    return pct_missing


def check_duplicates(df):
    print("\n=== 2. DUPLICATE (city, time) CHECK ===")
    dupes = df.duplicated(subset=["city", "time"], keep=False)
    n_dupes = dupes.sum()
    print(f"  {n_dupes} duplicate (city, time) rows found.")
    if n_dupes > 0:
        print("  Sample duplicates:")
        print(df[dupes].sort_values(["city", "time"]).head(10)[["city", "time", "aqi"]])
    return n_dupes


def check_null_rates(df):
    print("\n=== 3. NULL RATES PER COLUMN ===")
    null_pct = (df.isnull().mean() * 100).sort_values(ascending=False)
    high_null = null_pct[null_pct > 0]
    if len(high_null) == 0:
        print("  No nulls found anywhere.")
    else:
        for col, pct in high_null.items():
            print(f"  {col:<30}: {pct:.1f}% null")
    return null_pct


def check_ranges(df):
    print("\n=== 4. OUT-OF-RANGE VALUE CHECK ===")
    any_issues = False
    for col, (lo, hi) in EXPECTED_RANGES.items():
        if col not in df.columns:
            continue
        bad = df[(df[col] < lo) | (df[col] > hi)]
        if len(bad) > 0:
            any_issues = True
            print(f"  {col:<20}: {len(bad)} rows outside [{lo}, {hi}] "
                  f"(min={df[col].min():.1f}, max={df[col].max():.1f})")
    if not any_issues:
        print("  All checked columns within expected physical ranges.")


def check_aqi_consistency(df, sample_size=5000):
    print("\n=== 5. AQI RECOMPUTATION CHECK ===")
    sample = df.dropna(subset=["pm2_5", "pm10", "aqi"]).sample(
        min(sample_size, len(df)), random_state=42
    )
    recomputed = sample.apply(lambda r: compute_aqi(pm25=r["pm2_5"], pm10=r["pm10"]), axis=1)
    mismatch = (recomputed != sample["aqi"]).sum()
    pct_mismatch = 100 * mismatch / len(sample)
    print(f"  {mismatch}/{len(sample)} sampled rows ({pct_mismatch:.2f}%) mismatched stored AQI.")
    if pct_mismatch > 1:
        print("  -> Investigation needed: historical vs live AQI calculation drift.")


def check_categories(df):
    print("\n=== 6. CATEGORY SANITY CHECK ===")
    cities_found = sorted(df["city"].unique())
    expected_cities = sorted(CITIES.keys())
    print(f"  Cities in data: {cities_found}")
    missing = set(expected_cities) - set(cities_found)
    extra = set(cities_found) - set(expected_cities)
    if missing: print(f"  MISSING from data: {missing}")
    if extra:   print(f"  UNEXPECTED in data: {extra}")
    if not missing and not extra:
        print("  Matches config.py exactly.")


# ==========================================
# 2. GAP FIXER: TIME-AWARE LAG COMPUTATION
# ==========================================

def fix_time_aware_lags(df: pd.DataFrame, lag_hours=[1, 3, 6, 12, 24]) -> pd.DataFrame:
    """
    Fixes issue #1 by reindexing each city to a complete hourly range before shifting,
    ensuring 'aqi_lag_1h' is strictly 1 physical hour ago, not 1 row ago.
    """
    print("\n=== RE-ENGINEERING LAGS WITH STRICT PHYSICAL TIME ALIGNMENT ===")
    corrected_dfs = []

    for city, city_df in df.groupby("city"):
        city_df = city_df.sort_values("time").drop_duplicates("time")
        
        # Build complete hourly grid
        full_idx = pd.date_range(start=city_df["time"].min(), end=city_df["time"].max(), freq="h")
        city_reindexed = city_df.set_index("time").reindex(full_idx)
        city_reindexed["city"] = city

        # Recompute physical lags
        for h in lag_hours:
            city_reindexed[f"aqi_lag_{h}h"] = city_reindexed["aqi"].shift(h)

        # Drop introduced missing grid rows while keeping accurate lag values
        valid_rows = city_reindexed.dropna(subset=["aqi"]).reset_index().rename(columns={"index": "time"})
        corrected_dfs.append(valid_rows)

    corrected_df = pd.concat(corrected_dfs, ignore_index=True)
    print("✓ Lag features recomputed successfully against explicit datetime indices.")
    return corrected_df


# ==========================================
# 3. CITY-LEVEL NEGATIVE BIAS ANALYSIS
# ==========================================

def analyze_city_negative_bias(results_df: pd.DataFrame):
    """
    Analyzes model evaluation results for systematic negative bias.
    
    Expects results_df with columns:
    ['city', 'target_horizon', 'actual_aqi', 'predicted_aqi']
    """
    if results_df is None or results_df.empty:
        print("\n=== CITY BIAS ANALYSIS SKIPPED (No validation predictions provided) ===")
        print("Tip: Pass a DataFrame containing ['city', 'target_horizon', 'actual_aqi', 'predicted_aqi']")
        return None

    results_df = results_df.copy()
    # Residual Error = Predicted - Actual
    results_df["error"] = results_df["predicted_aqi"] - results_df["actual_aqi"]

    bias_summary = results_df.groupby(["city", "target_horizon"]).agg(
        Mean_Error=('error', 'mean'),  # Negative ME = Underpredicting (Negative Bias)
        MAE=('error', lambda x: np.abs(x).mean()),
        RMSE=('error', lambda x: np.sqrt((x**2).mean())),
        Count=('error', 'count')
    ).reset_index()

    print("\n=== CITY NEGATIVE BIAS ANALYSIS ===")
    print(bias_summary.to_string(index=False))

    neg_bias = bias_summary[bias_summary["Mean_Error"] < -2.0]
    if not neg_bias.empty:
        print("\n⚠️ ALERT: CITIES WITH SYSTEMATIC NEGATIVE BIAS (Underpredicting AQI):")
        for _, row in neg_bias.iterrows():
            print(f"  • {row['city']:<12} [{row['target_horizon']}]: Mean Error = {row['Mean_Error']:.2f} AQI points")
    else:
        print("\n✓ No critical negative bias (<-2.0 ME) detected across cities.")

    return bias_summary


def plot_city_residuals(results_df: pd.DataFrame):
    """Generates boxplot visual of prediction errors per city."""
    if results_df is None or results_df.empty:
        return

    results_df["residual"] = results_df["predicted_aqi"] - results_df["actual_aqi"]
    
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=results_df, x="city", y="residual", hue="target_horizon")
    plt.axhline(0, color="red", linestyle="--", linewidth=1.5, label="Zero Bias (Perfect Fit)")
    plt.title("AQI Prediction Residual Distribution by City (Predicted - Actual)")
    plt.ylabel("Error (Predicted AQI - Actual AQI)")
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ==========================================
# LOADER & MAIN PIPELINE
# ==========================================

def load_data():
    try:
        import hopsworks
        print("Loading data from Hopsworks feature store...")
        project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY, project=HOPSWORKS_PROJECT_NAME)
        fs = project.get_feature_store()
        fg = fs.get_feature_group(name=FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)
        df = fg.read()
        print(f"Loaded {len(df)} rows from Hopsworks.")
        return df
    except Exception as e:
        print(f"Could not load from Hopsworks ({e}), falling back to local CSV...")
        df = pd.read_csv(LOCAL_FALLBACK_PATH)
        print(f"Loaded {len(df)} rows from {LOCAL_FALLBACK_PATH}.")
        return df


def main():
    df = load_data()
    
    # Time standardization
    df["time"] = pd.to_datetime(df["time"])
    if hasattr(df["time"].dt, "tz") and df["time"].dt.tz is not None:
        df["time"] = df["time"].dt.tz_localize(None)

    print(f"Dataset Span: {df['time'].min()} to {df['time'].max()} across {df['city'].nunique()} cities.")

    # 1. Run Data Quality & Backfill Audit
    pct_missing = check_time_gaps(df)
    check_duplicates(df)
    check_null_rates(df)
    check_ranges(df)
    check_aqi_consistency(df)
    check_categories(df)

    # 2. Fix lag features if time gaps exist
    if pct_missing > 0:
        df = fix_time_aware_lags(df)

    # 3. Diagnostic Demonstration: Simulated validation results to test bias analysis
    # (Replace this mock section with your actual test predictions DataFrame)
    mock_results = []
    for city in df["city"].unique():
        for horizon in ["24h", "48h", "72h"]:
            # Simulate negative bias specifically for Lahore & Faisalabad
            bias_offset = -12.5 if city in ["Lahore", "Faisalabad"] else 0.5
            actuals = np.random.uniform(50, 250, 100)
            preds = actuals + bias_offset + np.random.normal(0, 10, 100)
            
            for act, prd in zip(actuals, preds):
                mock_results.append({
                    "city": city,
                    "target_horizon": horizon,
                    "actual_aqi": act,
                    "predicted_aqi": prd
                })
                
    results_df = pd.DataFrame(mock_results)
    
    # 4. Perform Negative Bias Analysis
    analyze_city_negative_bias(results_df)
    # plot_city_residuals(results_df) # Uncomment to render interactive plots

    print("\n=== AUDIT & DIAGNOSTICS COMPLETE ===")


if __name__ == "__main__":
    main()