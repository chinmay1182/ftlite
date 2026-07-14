import os
import datetime
import pandas as pd
from ftlite import Entity, Feature, FeatureView, OnDemandFeatureView, FtliteClient


def test_ftlite_end_to_end(tmp_path):
    # Setup temporary paths for registry and online db
    registry_path = tmp_path / "registry.json"
    online_db_path = tmp_path / "online_store.db"
    parquet_path = tmp_path / "customer_features.parquet"

    # Initialize client
    client = FtliteClient(
        registry_path=str(registry_path), online_db_path=str(online_db_path)
    )

    # 1. Define Entity
    customer = Entity(name="customer_id", value_type="INT64")
    client.register_entity(customer)

    # 2. Define Features and FeatureView
    age = Feature(name="age", dtype="int32")
    balance = Feature(name="balance", dtype="float64")

    fv = FeatureView(
        name="customer_fv",
        entities=[customer],
        features=[age, balance],
        source_path=str(parquet_path),
        timestamp_field="timestamp",
    )
    client.register_feature_view(fv)

    # 3. Ingest historical feature data
    feature_data = pd.DataFrame(
        {
            "customer_id": [1001, 1001, 1002],
            "timestamp": [
                "2026-07-14T10:00:00",
                "2026-07-14T12:00:00",
                "2026-07-14T11:00:00",
            ],
            "age": [30, 30, 25],
            "balance": [100.0, 150.0, 50.0],
        }
    )
    client.push("customer_fv", feature_data)

    # 4. Get Historical Features (Point-in-time Join)
    observation_df = pd.DataFrame(
        {
            "customer_id": [1001, 1001, 1002],
            "timestamp": [
                "2026-07-14T11:00:00",
                "2026-07-14T13:00:00",
                "2026-07-14T11:30:00",
            ],
        }
    )

    historical_features = client.get_historical_features(
        entity_df=observation_df, features=["age", "balance"], timestamp_col="timestamp"
    )

    assert len(historical_features) == 3

    row1 = historical_features[
        (historical_features["customer_id"] == 1001)
        & (historical_features["timestamp"] == "2026-07-14T11:00:00")
    ].iloc[0]
    assert row1["balance"] == 100.0
    assert row1["age"] == 30

    row2 = historical_features[
        (historical_features["customer_id"] == 1001)
        & (historical_features["timestamp"] == "2026-07-14T13:00:00")
    ].iloc[0]
    assert row2["balance"] == 150.0
    assert row2["age"] == 30

    # 5. Materialize to Online Store
    start_time = datetime.datetime(2026, 7, 14, 0, 0, 0)
    end_time = datetime.datetime(2026, 7, 14, 23, 59, 59)
    client.materialize(start_time, end_time)

    # 6. Retrieve Online Features
    online_features = client.get_online_features(
        entity_keys=[1001, 1002], features=["age", "balance"]
    )

    assert len(online_features) == 2

    cust1001 = next(item for item in online_features if item["entity_id"] == 1001)
    assert cust1001["age"] == 30
    assert cust1001["balance"] == 150.0

    cust1002 = next(item for item in online_features if item["entity_id"] == 1002)
    assert cust1002["age"] == 25
    assert cust1002["balance"] == 50.0


def test_edge_cases(tmp_path):
    registry_path = tmp_path / "registry.json"
    online_db_path = tmp_path / "online_store.db"
    parquet_path = tmp_path / "customer_features.parquet"

    client = FtliteClient(
        registry_path=str(registry_path), online_db_path=str(online_db_path)
    )

    customer = Entity(name="customer_id", value_type="INT64")
    client.register_entity(customer)

    age = Feature(name="age", dtype="int32")
    fv = FeatureView(
        name="customer_fv",
        entities=[customer],
        features=[age],
        source_path=str(parquet_path),
        timestamp_field="timestamp",
    )
    client.register_feature_view(fv)

    # Ingest single feature update at 12:00
    feature_data = pd.DataFrame(
        {"customer_id": [1001], "timestamp": ["2026-07-14T12:00:00"], "age": [30]}
    )
    client.push("customer_fv", feature_data)

    # Observation times:
    # 1. Observation BEFORE any updates exist (should be null/NaN)
    # 2. Observation for a MISSING customer (should be null/NaN)
    # 3. Normal observation
    observation_df = pd.DataFrame(
        {
            "customer_id": [1001, 9999, 1001],
            "timestamp": [
                "2026-07-14T11:00:00",  # before update
                "2026-07-14T13:00:00",  # missing customer
                "2026-07-14T13:00:00",  # normal match
            ],
        }
    )

    historical_features = client.get_historical_features(
        entity_df=observation_df, features=["age"], timestamp_col="timestamp"
    )

    assert len(historical_features) == 3

    # Customer 1001 at 11:00 (before update) -> age should be NaN/None
    val0 = historical_features[
        (historical_features["customer_id"] == 1001)
        & (historical_features["timestamp"] == "2026-07-14T11:00:00")
    ].iloc[0]["age"]
    assert val0 is None or pd.isna(val0)

    # Customer 9999 at 13:00 (missing customer) -> age should be NaN/None
    val1 = historical_features[
        (historical_features["customer_id"] == 9999)
        & (historical_features["timestamp"] == "2026-07-14T13:00:00")
    ].iloc[0]["age"]
    assert val1 is None or pd.isna(val1)

    # Customer 1001 at 13:00 (match) -> age should be 30
    val2 = historical_features[
        (historical_features["customer_id"] == 1001)
        & (historical_features["timestamp"] == "2026-07-14T13:00:00")
    ].iloc[0]["age"]
    assert val2 == 30

    # Materialize and test online edge cases
    start_time = datetime.datetime(2026, 7, 14, 0, 0, 0)
    end_time = datetime.datetime(2026, 7, 14, 23, 59, 59)
    client.materialize(start_time, end_time)

    online_features = client.get_online_features(
        entity_keys=[1001, 9999], features=["age"]
    )

    assert len(online_features) == 2

    cust1001 = next(item for item in online_features if item["entity_id"] == 1001)
    assert cust1001["age"] == 30

    cust9999 = next(item for item in online_features if item["entity_id"] == 9999)
    assert cust9999["age"] is None


def test_on_demand_transformations(tmp_path):
    registry_path = tmp_path / "registry.json"
    online_db_path = tmp_path / "online_store.db"
    parquet_path = tmp_path / "customer_features.parquet"

    client = FtliteClient(
        registry_path=str(registry_path), online_db_path=str(online_db_path)
    )

    customer = Entity(name="customer_id", value_type="INT64")
    client.register_entity(customer)

    balance = Feature(name="balance", dtype="float64")
    fv = FeatureView(
        name="customer_fv",
        entities=[customer],
        features=[balance],
        source_path=str(parquet_path),
        timestamp_field="timestamp",
    )
    client.register_feature_view(fv)

    # Ingest feature data
    feature_data = pd.DataFrame(
        {
            "customer_id": [1001, 1002],
            "timestamp": ["2026-07-14T12:00:00", "2026-07-14T12:00:00"],
            "balance": [100.0, 50.0],
        }
    )
    client.push("customer_fv", feature_data)

    # Define On Demand transformation function: adds 100 to balance
    def add_hundred_transform(df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        out["balance_plus_hundred"] = df["balance"] + 100.0
        return out

    # Define On Demand Feature View
    plus_hundred_feat = Feature(name="balance_plus_hundred", dtype="float64")
    odfv = OnDemandFeatureView(
        name="plus_hundred_fv",
        features=[plus_hundred_feat],
        inputs=["customer_fv:balance"],
        transform_fn=add_hundred_transform,
    )
    client.register_on_demand_feature_view(odfv)

    # 1. Historical retrieval with On Demand Feature
    observation_df = pd.DataFrame(
        {
            "customer_id": [1001, 1002],
            "timestamp": ["2026-07-14T13:00:00", "2026-07-14T13:00:00"],
        }
    )

    hist_res = client.get_historical_features(
        entity_df=observation_df, features=["balance", "balance_plus_hundred"]
    )

    assert len(hist_res) == 2
    row1 = hist_res[hist_res["customer_id"] == 1001].iloc[0]
    assert row1["balance"] == 100.0
    assert row1["balance_plus_hundred"] == 200.0

    row2 = hist_res[hist_res["customer_id"] == 1002].iloc[0]
    assert row2["balance"] == 50.0
    assert row2["balance_plus_hundred"] == 150.0

    # 2. Online retrieval with On Demand Feature
    start_time = datetime.datetime(2026, 7, 14, 0, 0, 0)
    end_time = datetime.datetime(2026, 7, 14, 23, 59, 59)
    client.materialize(start_time, end_time)

    # Need to re-register transform_fn because it is not saved to registry json file
    # We will simulate loading the registry from disk
    client2 = FtliteClient(
        registry_path=str(registry_path), online_db_path=str(online_db_path)
    )
    # Re-register transform_fn to simulate runtime script behavior
    client2.registry.get_on_demand_feature_view("plus_hundred_fv").transform_fn = (
        add_hundred_transform
    )

    online_res = client2.get_online_features(
        entity_keys=[1001, 1002], features=["balance", "balance_plus_hundred"]
    )

    assert len(online_res) == 2

    cust1001 = next(item for item in online_res if item["entity_id"] == 1001)
    assert cust1001["balance"] == 100.0
    assert cust1001["balance_plus_hundred"] == 200.0

    cust1002 = next(item for item in online_res if item["entity_id"] == 1002)
    assert cust1002["balance"] == 50.0
    assert cust1002["balance_plus_hundred"] == 150.0


def test_advanced_features(tmp_path):
    import polars as pl

    registry_path = tmp_path / "registry.json"
    online_db_path = tmp_path / "online_store.db"
    parquet_path1 = tmp_path / "user_v1.parquet"
    parquet_path2 = tmp_path / "user_v2.parquet"

    client = FtliteClient(
        registry_path=str(registry_path), online_db_path=str(online_db_path)
    )

    user_ent = Entity(name="user_id", value_type="INT64")
    client.register_entity(user_ent)

    # 1. Versioning setup
    # v1 features
    score_v1 = Feature(name="score", dtype="int32")
    fv_v1 = FeatureView(
        name="user_fv",
        version="v1",
        entities=[user_ent],
        features=[score_v1],
        source_path=str(parquet_path1),
        timestamp_field="timestamp",
    )
    client.register_feature_view(fv_v1)

    # v2 features
    score_v2 = Feature(name="score", dtype="int32")
    fv_v2 = FeatureView(
        name="user_fv",
        version="v2",
        entities=[user_ent],
        features=[score_v2],
        source_path=str(parquet_path2),
        timestamp_field="timestamp",
    )
    client.register_feature_view(fv_v2)

    # Push different data to v1 and v2
    data_v1 = pd.DataFrame(
        {"user_id": [1], "timestamp": ["2026-07-14T12:00:00"], "score": [50]}
    )
    data_v2 = pd.DataFrame(
        {"user_id": [1], "timestamp": ["2026-07-14T12:00:00"], "score": [99]}
    )

    client.push("user_fv@v1", data_v1)
    client.push("user_fv@v2", data_v2)

    # Define observation df
    obs_df = pd.DataFrame({"user_id": [1], "timestamp": ["2026-07-14T13:00:00"]})

    # Test Fallback (Should resolve to latest: v2)
    df_fallback = client.get_historical_features(
        entity_df=obs_df, features=["user_fv:score"]
    )
    assert df_fallback.iloc[0]["user_fv:score"] == 99

    # Test Specific Version Lookup (Style 1: fv_name@version:feature_name)
    df_v1_style1 = client.get_historical_features(
        entity_df=obs_df, features=["user_fv@v1:score"]
    )
    assert df_v1_style1.iloc[0]["user_fv@v1:score"] == 50

    # Test Specific Version Lookup (Style 2: fv_name:feature_name@version)
    df_v1_style2 = client.get_historical_features(
        entity_df=obs_df, features=["user_fv:score@v1"]
    )
    assert df_v1_style2.iloc[0]["user_fv:score@v1"] == 50

    # 2. Polars output test
    df_polars = client.get_historical_features(
        entity_df=obs_df, features=["user_fv@v2:score"], output_format="polars"
    )
    assert isinstance(df_polars, pl.DataFrame)
    assert df_polars.row(0)[0] == 1  # user_id

    # 3. Caching test
    cache_dir = tmp_path / "cache"
    # First request - warm cache
    df_cached1 = client.get_historical_features(
        entity_df=obs_df,
        features=["user_fv@v1:score"],
        cache=True,
        cache_dir=str(cache_dir),
    )
    assert df_cached1.iloc[0]["user_fv@v1:score"] == 50

    # Change offline value to simulate out-of-sync
    data_v1_new = pd.DataFrame(
        {"user_id": [1], "timestamp": ["2026-07-14T12:00:00"], "score": [999]}
    )
    # Write directly to file to bypass cache invalidation
    data_v1_new.to_parquet(str(parquet_path1))

    # Second request - should read from cache (getting 50 instead of 999)
    df_cached2 = client.get_historical_features(
        entity_df=obs_df,
        features=["user_fv@v1:score"],
        cache=True,
        cache_dir=str(cache_dir),
    )
    assert df_cached2.iloc[0]["user_fv@v1:score"] == 50

    # Third request - no cache (should read 999)
    df_no_cached = client.get_historical_features(
        entity_df=obs_df, features=["user_fv@v1:score"], cache=False
    )
    assert df_no_cached.iloc[0]["user_fv@v1:score"] == 999

    # Cache clear test
    client.get_historical_features(
        entity_df=obs_df,
        features=["user_fv@v1:score"],
        cache=True,
        cache_dir=str(cache_dir),
    )
    assert len(os.listdir(str(cache_dir))) > 0
    client.clear_cache(cache_dir=str(cache_dir))
    assert len(os.listdir(str(cache_dir))) == 0

    # 4. Lineage tracing
    lineage = client.get_feature_lineage("user_fv@v1:score")
    assert lineage["name"] == "user_fv@v1:score"
    assert lineage["type"] == "standard"
    assert lineage["feature_view"] == "user_fv@v1"
    assert lineage["source_path"] == str(parquet_path1)


def test_cache_eviction_bounds(tmp_path):
    from ftlite.client import clean_cache
    import time

    cache_dir = tmp_path / "cache"
    os.makedirs(str(cache_dir), exist_ok=True)

    # 1. Create 5 mock cache files
    for i in range(5):
        with open(str(cache_dir / f"file_{i}.parquet"), "w") as f:
            f.write("mock")

    # 2. Run clean_cache with max_files = 3
    clean_cache(str(cache_dir), max_files=3)
    assert len(os.listdir(str(cache_dir))) == 3

    # 3. Modify time of one file to be very old
    old_file = cache_dir / "file_old.parquet"
    with open(str(old_file), "w") as f:
        f.write("mock")
    # Set modification time to 2 days ago (172800 seconds)
    os.utime(str(old_file), (time.time() - 172800, time.time() - 172800))

    # 4. Run clean_cache with TTL = 1 day (86400 seconds)
    clean_cache(str(cache_dir), ttl_seconds=86400, max_files=10)
    # The old file should be gone!
    assert not os.path.exists(str(old_file))
