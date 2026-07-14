import os
import datetime
import pandas as pd
from ftlite import Entity, Feature, FeatureView, OnDemandFeatureView, FtliteClient


def main():
    print("=== FTLite Fraud Detection Prediction Example ===")

    # Initialize Ftlite Client targeting temporary directory files
    temp_dir = "temp_fraud_example"
    os.makedirs(temp_dir, exist_ok=True)
    registry_path = os.path.join(temp_dir, "registry.json")
    online_db_path = os.path.join(temp_dir, "online_store.db")
    parquet_path = os.path.join(temp_dir, "user_transactions.parquet")

    client = FtliteClient(registry_path=registry_path, online_db_path=online_db_path)

    # 1. Define and Register Entity
    user = Entity(name="user_id", value_type="INT64")
    client.register_entity(user)
    print("1. Registered User Entity.")

    # 2. Register Feature View
    tx_fv = FeatureView(
        name="user_transactions",
        entities=[user],
        features=[
            Feature(name="amount", dtype="double"),
            Feature(name="limit_threshold", dtype="double"),
        ],
        source_path=parquet_path,
        timestamp_field="timestamp",
    )
    client.register_feature_view(tx_fv)
    print("2. Registered user_transactions FeatureView.")

    # 3. Define and Register On-Demand Transformation
    # Compare raw transaction amounts against user limit threshold
    def compare_limit_transform(df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        out["exceeds_limit"] = df["amount"] > df["limit_threshold"]
        return out

    exceeds_limit_feat = Feature(name="exceeds_limit", dtype="boolean")
    odfv = OnDemandFeatureView(
        name="fraud_rules",
        features=[exceeds_limit_feat],
        inputs=["user_transactions:amount", "user_transactions:limit_threshold"],
        transform_fn=compare_limit_transform,
    )
    client.register_on_demand_feature_view(odfv)
    print("3. Registered fraud_rules OnDemandFeatureView.")

    # 4. Ingest raw transaction data
    tx_df = pd.DataFrame(
        {
            "user_id": [201, 202, 201, 202],
            "timestamp": [
                "2026-07-01T10:00:00",
                "2026-07-01T10:00:00",
                "2026-07-14T12:00:00",
                "2026-07-14T12:00:00",
            ],
            "amount": [45.0, 950.0, 1200.0, 30.5],
            "limit_threshold": [500.0, 500.0, 500.0, 500.0],
        }
    )
    client.push("user_transactions", tx_df)
    print("4. Ingested raw transactions into Parquet storage.")

    # 5. Point-In-Time Join for training targets
    obs_df = pd.DataFrame(
        {
            "user_id": [201, 202],
            "timestamp": ["2026-07-14T13:00:00", "2026-07-14T13:00:00"],
        }
    )

    training_df = client.get_historical_features(
        entity_df=obs_df,
        features=["user_transactions:amount", "fraud_rules:exceeds_limit"],
    )
    print("\nJoined Training Features with On-Demand Transformations:")
    print(training_df)

    # 6. Materialize features to SQLite Online Store
    start_time = datetime.datetime(2026, 7, 1)
    end_time = datetime.datetime(2026, 7, 15)
    client.materialize(start_time, end_time)

    # Re-register transform_fn to simulate fresh runtime load serving
    client2 = FtliteClient(registry_path=registry_path, online_db_path=online_db_path)
    client2.registry.get_on_demand_feature_view("fraud_rules").transform_fn = (
        compare_limit_transform
    )

    # 7. Get low-latency online features
    online_feats = client2.get_online_features(
        entity_keys=[201, 202],
        features=["user_transactions:amount", "fraud_rules:exceeds_limit"],
    )
    print("\nServed Online Features:")
    for user_record in online_feats:
        print(user_record)

    # Cleanup temp files
    try:
        import shutil

        shutil.rmtree(temp_dir)
    except Exception:
        pass


if __name__ == "__main__":
    main()
