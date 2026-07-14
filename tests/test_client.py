import os
import datetime
import pandas as pd
import numpy as np
import pytest
from ftlite import Entity, Feature, FeatureView, OnDemandFeatureView, FtliteClient

def test_ftlite_end_to_end(tmp_path):
    # Setup temporary paths for registry and online db
    registry_path = tmp_path / "registry.json"
    online_db_path = tmp_path / "online_store.db"
    parquet_path = tmp_path / "customer_features.parquet"

    # Initialize client
    client = FtliteClient(
        registry_path=str(registry_path),
        online_db_path=str(online_db_path)
    )

    # 1. Define Entity
    customer = Entity(
        name="customer_id",
        value_type="INT64"
    )
    client.register_entity(customer)

    # 2. Define Features and FeatureView
    age = Feature(name="age", dtype="int32")
    balance = Feature(name="balance", dtype="float64")
    
    fv = FeatureView(
        name="customer_fv",
        entities=[customer],
        features=[age, balance],
        source_path=str(parquet_path),
        timestamp_field="timestamp"
    )
    client.register_feature_view(fv)

    # 3. Ingest historical feature data
    feature_data = pd.DataFrame({
        "customer_id": [1001, 1001, 1002],
        "timestamp": [
            "2026-07-14T10:00:00",
            "2026-07-14T12:00:00",
            "2026-07-14T11:00:00"
        ],
        "age": [30, 30, 25],
        "balance": [100.0, 150.0, 50.0]
    })
    client.push("customer_fv", feature_data)

    # 4. Get Historical Features (Point-in-time Join)
    observation_df = pd.DataFrame({
        "customer_id": [1001, 1001, 1002],
        "timestamp": [
            "2026-07-14T11:00:00",
            "2026-07-14T13:00:00",
            "2026-07-14T11:30:00"
        ]
    })

    historical_features = client.get_historical_features(
        entity_df=observation_df,
        features=["age", "balance"],
        timestamp_col="timestamp"
    )

    assert len(historical_features) == 3
    
    row1 = historical_features[(historical_features["customer_id"] == 1001) & (historical_features["timestamp"] == "2026-07-14T11:00:00")].iloc[0]
    assert row1["balance"] == 100.0
    assert row1["age"] == 30

    row2 = historical_features[(historical_features["customer_id"] == 1001) & (historical_features["timestamp"] == "2026-07-14T13:00:00")].iloc[0]
    assert row2["balance"] == 150.0
    assert row2["age"] == 30

    # 5. Materialize to Online Store
    start_time = datetime.datetime(2026, 7, 14, 0, 0, 0)
    end_time = datetime.datetime(2026, 7, 14, 23, 59, 59)
    client.materialize(start_time, end_time)

    # 6. Retrieve Online Features
    online_features = client.get_online_features(
        entity_keys=[1001, 1002],
        features=["age", "balance"]
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
        registry_path=str(registry_path),
        online_db_path=str(online_db_path)
    )

    customer = Entity(name="customer_id", value_type="INT64")
    client.register_entity(customer)

    age = Feature(name="age", dtype="int32")
    fv = FeatureView(
        name="customer_fv",
        entities=[customer],
        features=[age],
        source_path=str(parquet_path),
        timestamp_field="timestamp"
    )
    client.register_feature_view(fv)

    # Ingest single feature update at 12:00
    feature_data = pd.DataFrame({
        "customer_id": [1001],
        "timestamp": ["2026-07-14T12:00:00"],
        "age": [30]
    })
    client.push("customer_fv", feature_data)

    # Observation times:
    # 1. Observation BEFORE any updates exist (should be null/NaN)
    # 2. Observation for a MISSING customer (should be null/NaN)
    # 3. Normal observation
    observation_df = pd.DataFrame({
        "customer_id": [1001, 9999, 1001],
        "timestamp": [
            "2026-07-14T11:00:00",  # before update
            "2026-07-14T13:00:00",  # missing customer
            "2026-07-14T13:00:00"   # normal match
        ]
    })

    historical_features = client.get_historical_features(
        entity_df=observation_df,
        features=["age"],
        timestamp_col="timestamp"
    )

    assert len(historical_features) == 3

    # Row 0: customer 1001 at 11:00 (before update) -> age should be NaN/None
    val0 = historical_features.iloc[0]["age"]
    assert val0 is None or pd.isna(val0)

    # Row 1: customer 9999 at 13:00 (missing customer) -> age should be NaN/None
    val1 = historical_features.iloc[1]["age"]
    assert val1 is None or pd.isna(val1)

    # Row 2: customer 1001 at 13:00 (match) -> age should be 30
    assert historical_features.iloc[2]["age"] == 30

    # Materialize and test online edge cases
    start_time = datetime.datetime(2026, 7, 14, 0, 0, 0)
    end_time = datetime.datetime(2026, 7, 14, 23, 59, 59)
    client.materialize(start_time, end_time)

    online_features = client.get_online_features(
        entity_keys=[1001, 9999],
        features=["age"]
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
        registry_path=str(registry_path),
        online_db_path=str(online_db_path)
    )

    customer = Entity(name="customer_id", value_type="INT64")
    client.register_entity(customer)

    balance = Feature(name="balance", dtype="float64")
    fv = FeatureView(
        name="customer_fv",
        entities=[customer],
        features=[balance],
        source_path=str(parquet_path),
        timestamp_field="timestamp"
    )
    client.register_feature_view(fv)

    # Ingest feature data
    feature_data = pd.DataFrame({
        "customer_id": [1001, 1002],
        "timestamp": ["2026-07-14T12:00:00", "2026-07-14T12:00:00"],
        "balance": [100.0, 50.0]
    })
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
        transform_fn=add_hundred_transform
    )
    client.register_on_demand_feature_view(odfv)

    # 1. Historical retrieval with On Demand Feature
    observation_df = pd.DataFrame({
        "customer_id": [1001, 1002],
        "timestamp": ["2026-07-14T13:00:00", "2026-07-14T13:00:00"]
    })

    hist_res = client.get_historical_features(
        entity_df=observation_df,
        features=["balance", "balance_plus_hundred"]
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
        registry_path=str(registry_path),
        online_db_path=str(online_db_path)
    )
    # Re-register transform_fn to simulate runtime script behavior
    client2.registry.get_on_demand_feature_view("plus_hundred_fv").transform_fn = add_hundred_transform

    online_res = client2.get_online_features(
        entity_keys=[1001, 1002],
        features=["balance", "balance_plus_hundred"]
    )

    assert len(online_res) == 2
    
    cust1001 = next(item for item in online_res if item["entity_id"] == 1001)
    assert cust1001["balance"] == 100.0
    assert cust1001["balance_plus_hundred"] == 200.0

    cust1002 = next(item for item in online_res if item["entity_id"] == 1002)
    assert cust1002["balance"] == 50.0
    assert cust1002["balance_plus_hundred"] == 150.0
