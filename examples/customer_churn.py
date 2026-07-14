import os
import datetime
import pandas as pd
from ftlite import Entity, Feature, FeatureView, FtliteClient


def main():
    print("=== FTLite Customer Churn Prediction Example ===")

    # Initialize Ftlite Client targeting temporary directory files
    temp_dir = "temp_churn_example"
    os.makedirs(temp_dir, exist_ok=True)
    registry_path = os.path.join(temp_dir, "registry.json")
    online_db_path = os.path.join(temp_dir, "online_store.db")
    parquet_path = os.path.join(temp_dir, "customer_stats.parquet")

    client = FtliteClient(registry_path=registry_path, online_db_path=online_db_path)

    # 1. Define Entity
    customer = Entity(name="customer_id", value_type="INT64")
    client.register_entity(customer)
    print("1. Registered Customer Entity.")

    # 2. Register Feature View
    activity_fv = FeatureView(
        name="customer_activity",
        entities=[customer],
        features=[
            Feature(name="usage_minutes", dtype="double"),
            Feature(name="support_calls", dtype="int64"),
        ],
        source_path=parquet_path,
        timestamp_field="timestamp",
    )
    client.register_feature_view(activity_fv)
    print("2. Registered customer_activity FeatureView.")

    # 3. Ingest Dynamic Customer Features
    features_df = pd.DataFrame(
        {
            "customer_id": [101, 102, 101, 102],
            "timestamp": [
                "2026-07-01T10:00:00",
                "2026-07-01T10:00:00",
                "2026-07-14T12:00:00",
                "2026-07-14T12:00:00",
            ],
            "usage_minutes": [120.5, 45.2, 210.0, 50.8],
            "support_calls": [1, 3, 0, 4],
        }
    )
    client.push("customer_activity", features_df)
    print("3. Ingested raw feature records into Parquet storage.")

    # 4. Point-In-Time Join for Training Data
    # Let's say we have target labels captured at specific moments
    labels_df = pd.DataFrame(
        {
            "customer_id": [101, 102],
            "timestamp": ["2026-07-14T13:00:00", "2026-07-14T13:00:00"],
            "churned": [0, 1],
        }
    )

    training_df = client.get_historical_features(
        entity_df=labels_df,
        features=["customer_activity:usage_minutes", "customer_activity:support_calls"],
        timestamp_col="timestamp",
    )
    print("\nJoined Historical Training Features (Point-in-Time Correct):")
    print(training_df)

    # 5. Materialize features to Online SQLite store for low-latency serving
    start_time = datetime.datetime(2026, 7, 1)
    end_time = datetime.datetime(2026, 7, 15)
    client.materialize(start_time, end_time)
    print("\n4. Materialized feature views to serving database.")

    # 6. Low-Latency Online Prediction Retrieve
    online_feats = client.get_online_features(
        entity_keys=[101, 102],
        features=["customer_activity:usage_minutes", "customer_activity:support_calls"],
    )
    print("\nServed Online Features:")
    for customer_record in online_feats:
        print(customer_record)

    # Cleanup temp dir files
    try:
        import shutil

        shutil.rmtree(temp_dir)
    except Exception:
        pass


if __name__ == "__main__":
    main()
