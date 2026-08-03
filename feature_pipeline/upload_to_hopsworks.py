"""
Upload historical features to the Hopsworks Feature Store.

Run this ONCE after feature_engineering.py has produced data/features_all_cities.csv.
It creates (or reuses) the aqi_features feature group and inserts the full history.

Usage:
    python upload_to_hopsworks.py
"""

import pandas as pd
import hopsworks

from config import (
    HOPSWORKS_API_KEY,
    HOPSWORKS_PROJECT_NAME,
    FEATURE_GROUP_NAME,
    FEATURE_GROUP_VERSION,
)


def main():
    print("Loading features_all_cities.csv...")
    df = pd.read_csv("../data/features_all_cities.csv")
    df["time"] = pd.to_datetime(df["time"])

    # Hopsworks is picky about column names: no spaces, no special chars, lowercase is safest
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]

    print(f"Loaded {len(df)} rows, {df['city'].nunique()} cities.")

    print("Logging into Hopsworks...")
    project = hopsworks.login(
        api_key_value=HOPSWORKS_API_KEY,
        project=HOPSWORKS_PROJECT_NAME,
    )
    fs = project.get_feature_store()

    print(f"Creating/fetching feature group '{FEATURE_GROUP_NAME}' v{FEATURE_GROUP_VERSION}...")
    feature_group = fs.get_or_create_feature_group(
        name=FEATURE_GROUP_NAME,
        version=FEATURE_GROUP_VERSION,
        description="Hourly AQI features per Pakistani city: weather, pollutants, lags, rolling stats, targets",
        primary_key=["city", "time"],
        event_time="time",
        time_travel_format="HUDI",
        stream=True,  # required alongside HUDI when using the python-only (non-Spark) client
    )

    print("Inserting data (this can take a few minutes for ~170k rows)...")
    feature_group.insert(df, write_options={"wait_for_job": True})

    print("Done. Check the Hopsworks UI under Feature Store -> Feature Groups to confirm.")


if __name__ == "__main__":
    main()