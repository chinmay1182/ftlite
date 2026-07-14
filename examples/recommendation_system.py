import os
import pandas as pd
from ftlite import Entity, Feature, FeatureView, FtliteClient


def main():
    print("=== FTLite Recommendation System Example ===")

    # Initialize Ftlite Client targeting temporary directory files
    temp_dir = "temp_rec_example"
    os.makedirs(temp_dir, exist_ok=True)
    registry_path = os.path.join(temp_dir, "registry.json")
    online_db_path = os.path.join(temp_dir, "online_store.db")
    parquet_v1 = os.path.join(temp_dir, "clicks_v1.parquet")
    parquet_v2 = os.path.join(temp_dir, "clicks_v2.parquet")

    client = FtliteClient(registry_path=registry_path, online_db_path=online_db_path)

    # 1. Define Entity
    user = Entity(name="user_id", value_type="INT64")
    client.register_entity(user)
    print("1. Registered User Entity.")

    # 2. Register v1 and v2 Feature Views (Versioning)
    clicks_v1 = FeatureView(
        name="user_clicks",
        version="v1",
        entities=[user],
        features=[Feature(name="click_count", dtype="int64")],
        source_path=parquet_v1,
        timestamp_field="timestamp",
    )
    client.register_feature_view(clicks_v1)

    clicks_v2 = FeatureView(
        name="user_clicks",
        version="v2",
        entities=[user],
        features=[
            Feature(name="click_count", dtype="int64"),
            Feature(name="last_category", dtype="string"),
        ],
        source_path=parquet_v2,
        timestamp_field="timestamp",
    )
    client.register_feature_view(clicks_v2)
    print("2. Registered user_clicks@v1 and user_clicks@v2 FeatureViews.")

    # 3. Ingest separate datasets for v1 and v2
    v1_df = pd.DataFrame(
        {
            "user_id": [301, 302],
            "timestamp": ["2026-07-14T12:00:00", "2026-07-14T12:00:00"],
            "click_count": [15, 8],
        }
    )
    client.push("user_clicks@v1", v1_df)

    v2_df = pd.DataFrame(
        {
            "user_id": [301, 302],
            "timestamp": ["2026-07-14T12:00:00", "2026-07-14T12:00:00"],
            "click_count": [25, 12],
            "last_category": ["Electronics", "Books"],
        }
    )
    client.push("user_clicks@v2", v2_df)
    print("3. Ingested data for v1 and v2 into separate Parquet sources.")

    # 4. Point-In-Time Joins showing fallbacks and syntax
    obs_df = pd.DataFrame(
        {
            "user_id": [301, 302],
            "timestamp": ["2026-07-14T13:00:00", "2026-07-14T13:00:00"],
        }
    )

    # 4a. Query without version (should fallback to v2 - the latest version)
    df_fallback = client.get_historical_features(
        entity_df=obs_df,
        features=["user_clicks:click_count", "user_clicks:last_category"],
    )
    print("\nFallback to Latest Version (v2):")
    print(df_fallback)

    # 4b. Query specifying version explicitly using different syntaxes
    # Style 1: user_clicks@v1:click_count
    df_style1 = client.get_historical_features(
        entity_df=obs_df, features=["user_clicks@v1:click_count"]
    )
    print("\nStyle 1 Lookup (user_clicks@v1:click_count):")
    print(df_style1)

    # Style 2: user_clicks:click_count@v1
    df_style2 = client.get_historical_features(
        entity_df=obs_df, features=["user_clicks:click_count@v1"]
    )
    print("\nStyle 2 Lookup (user_clicks:click_count@v1):")
    print(df_style2)

    # Cleanup temp files
    try:
        import shutil

        shutil.rmtree(temp_dir)
    except Exception:
        pass


if __name__ == "__main__":
    main()
