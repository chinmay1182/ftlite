import os
import datetime
import pandas as pd
from ftlite import Entity, Feature, FeatureView, FtliteClient


def main():
    print("=== FTLite Time Series Forecasting Example ===")

    # Initialize Ftlite Client targeting temporary directory files
    temp_dir = "temp_forecasting_example"
    os.makedirs(temp_dir, exist_ok=True)
    registry_path = os.path.join(temp_dir, "registry.json")
    online_db_path = os.path.join(temp_dir, "online_store.db")
    parquet_path = os.path.join(temp_dir, "demand_stats.parquet")

    client = FtliteClient(registry_path=registry_path, online_db_path=online_db_path)

    # 1. Define Entity (Grid station)
    station = Entity(name="station_id", value_type="INT64")
    client.register_entity(station)
    print("1. Registered Station Entity.")

    # 2. Register Feature View for demand features
    demand_fv = FeatureView(
        name="station_demand",
        entities=[station],
        features=[
            Feature(name="load_kw", dtype="double"),
            Feature(name="temp_celsius", dtype="double"),
        ],
        source_path=parquet_path,
        timestamp_field="timestamp",
    )
    client.register_feature_view(demand_fv)
    print("2. Registered station_demand FeatureView.")

    # 3. Ingest dynamic time-series features (simulated lag logs)
    # Station 401 load levels every hour
    records_df = pd.DataFrame(
        {
            "station_id": [401, 401, 401, 401],
            "timestamp": [
                "2026-07-14T08:00:00",
                "2026-07-14T09:00:00",
                "2026-07-14T10:00:00",
                "2026-07-14T11:00:00",
            ],
            "load_kw": [450.5, 480.0, 520.2, 510.8],
            "temp_celsius": [22.4, 23.5, 24.8, 25.1],
        }
    )
    client.push("station_demand", records_df)
    print("3. Ingested dynamic time-series observations into Parquet.")

    # 4. Point-In-Time Join to match lags
    # Let's say we want to predict grid load at 11:30 based on what was known at 11:00
    forecast_df = pd.DataFrame(
        {
            "station_id": [401],
            "timestamp": ["2026-07-14T11:30:00"],
        }
    )

    joined_df = client.get_historical_features(
        entity_df=forecast_df,
        features=["station_demand:load_kw", "station_demand:temp_celsius"],
        timestamp_col="timestamp",
    )
    print("\nJoined Historical Forecast Lags (Correct Point-in-Time):")
    print(joined_df)

    # 5. Materialize features to SQLite Online Store
    start_time = datetime.datetime(2026, 7, 14, 0, 0, 0)
    end_time = datetime.datetime(2026, 7, 14, 23, 59, 59)
    client.materialize(start_time, end_time)

    # 6. Get online features for real-time forecasting inference
    online_feats = client.get_online_features(
        entity_keys=[401],
        features=["station_demand:load_kw", "station_demand:temp_celsius"],
    )
    print("\nServed Online Features:")
    for record in online_feats:
        print(record)

    # Cleanup temp files
    try:
        import shutil

        shutil.rmtree(temp_dir)
    except Exception:
        pass


if __name__ == "__main__":
    main()
