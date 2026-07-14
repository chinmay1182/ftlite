import os
import datetime
import pandas as pd
from ftlite import Entity, Feature, FeatureView, FtliteClient


def main():
    print("=== FTLite House Price Prediction Example ===")

    # Initialize Ftlite Client targeting temporary directory files
    temp_dir = "temp_house_example"
    os.makedirs(temp_dir, exist_ok=True)
    registry_path = os.path.join(temp_dir, "registry.json")
    online_db_path = os.path.join(temp_dir, "online_store.db")
    parquet_path = os.path.join(temp_dir, "property_market.parquet")

    client = FtliteClient(registry_path=registry_path, online_db_path=online_db_path)

    # 1. Define Entity (ZIP code)
    zipcode = Entity(name="zipcode", value_type="STRING")
    client.register_entity(zipcode)
    print("1. Registered Zipcode Entity.")

    # 2. Register Feature View for ZIP code average price trends
    market_fv = FeatureView(
        name="zip_market_stats",
        entities=[zipcode],
        features=[
            Feature(name="avg_sqft_price", dtype="double"),
            Feature(name="market_index", dtype="double"),
        ],
        source_path=parquet_path,
        timestamp_field="timestamp",
    )
    client.register_feature_view(market_fv)
    print("2. Registered zip_market_stats FeatureView.")

    # 3. Ingest trend features
    stats_df = pd.DataFrame(
        {
            "zipcode": ["94101", "94101", "90210", "90210"],
            "timestamp": [
                "2026-01-01T00:00:00",
                "2026-06-01T00:00:00",
                "2026-01-01T00:00:00",
                "2026-06-01T00:00:00",
            ],
            "avg_sqft_price": [850.0, 920.0, 1200.0, 1350.0],
            "market_index": [1.0, 1.08, 1.2, 1.35],
        }
    )
    client.push("zip_market_stats", stats_df)
    print("3. Ingested raw property stats into Parquet storage.")

    # 4. Point-In-Time Join: Houses sold at specific dates
    houses_sold_df = pd.DataFrame(
        {
            "zipcode": ["94101", "90210", "94101"],
            "timestamp": [
                "2026-03-15T12:00:00",  # should match Jan stats
                "2026-06-15T12:00:00",  # should match June stats
                "2026-07-01T09:00:00",  # should match June stats
            ],
            "house_id": [1, 2, 3],
        }
    )

    joined_df = client.get_historical_features(
        entity_df=houses_sold_df,
        features=["zip_market_stats:avg_sqft_price", "zip_market_stats:market_index"],
        timestamp_col="timestamp",
    )
    print("\nJoined Temporal Historical Features (Point-in-Time Correct):")
    print(joined_df)

    # 5. Materialize features to SQLite Online Store
    start_time = datetime.datetime(2026, 1, 1)
    end_time = datetime.datetime(2026, 7, 10)
    client.materialize(start_time, end_time)

    # 6. Get online features
    online_feats = client.get_online_features(
        entity_keys=["94101", "90210"],
        features=["zip_market_stats:avg_sqft_price", "zip_market_stats:market_index"],
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
