import os
import time
import datetime
import pandas as pd
from ftlite import Entity, Feature, FeatureView, FtliteClient


def main():
    print("=== FTLite Comparative Performance Benchmark ===")

    temp_dir = "temp_benchmark_dir"
    os.makedirs(temp_dir, exist_ok=True)
    registry_path = os.path.join(temp_dir, "registry.json")
    online_db_path = os.path.join(temp_dir, "online_store.db")
    parquet_path = os.path.join(temp_dir, "bench_data.parquet")

    client = FtliteClient(registry_path=registry_path, online_db_path=online_db_path)

    # 1. Setup entities and feature views
    user = Entity(name="user_id", value_type="INT64")
    client.register_entity(user)

    fv = FeatureView(
        name="user_stats",
        entities=[user],
        features=[
            Feature(name="metric_a", dtype="double"),
            Feature(name="metric_b", dtype="int64"),
        ],
        source_path=parquet_path,
        timestamp_field="timestamp",
    )
    client.register_feature_view(fv)

    # Generate 5,000 baseline records
    print("Generating 5,000 mock baseline records...")
    base_df = pd.DataFrame(
        {
            "user_id": list(range(1, 5001)),
            "timestamp": ["2026-07-14T10:00:00"] * 5000,
            "metric_a": [1.23 * i for i in range(5000)],
            "metric_b": list(range(5000)),
        }
    )
    client.push("user_stats", base_df)

    # Generate 1,000 query observation entities
    obs_df = pd.DataFrame(
        {
            "user_id": list(range(1, 1001)),
            "timestamp": ["2026-07-14T12:00:00"] * 1000,
        }
    )

    # Benchmark 1: Cache Miss vs Cache Hit
    print("\n1. Running Cache Miss vs Cache Hit Join...")
    # Cache Miss (First run)
    start_miss = time.perf_counter()
    client.get_historical_features(
        entity_df=obs_df,
        features=["user_stats:metric_a", "user_stats:metric_b"],
        cache=True,
        cache_dir=os.path.join(temp_dir, "cache"),
    )
    time_miss = time.perf_counter() - start_miss
    print(f"   Cache Miss time: {time_miss:.4f} seconds")

    # Cache Hit (Second run)
    start_hit = time.perf_counter()
    client.get_historical_features(
        entity_df=obs_df,
        features=["user_stats:metric_a", "user_stats:metric_b"],
        cache=True,
        cache_dir=os.path.join(temp_dir, "cache"),
    )
    time_hit = time.perf_counter() - start_hit
    speedup = time_miss / max(time_hit, 1e-6)
    print(f"   Cache Hit time:  {time_hit:.4f} seconds (Speedup: {speedup:.1f}x)")

    # Benchmark 2: Pandas vs Polars
    # Check if Polars is installed
    try:
        import polars as pl

        _ = pl
        has_polars = True
    except ImportError:
        has_polars = False

    if has_polars:
        print("\n2. Running Pandas vs Polars extraction benchmark...")
        # Pandas Output format
        start_pd = time.perf_counter()
        _ = client.get_historical_features(
            entity_df=obs_df,
            features=["user_stats:metric_a"],
            output_format="pandas",
            cache=False,
        )
        time_pd = time.perf_counter() - start_pd

        # Polars Output format
        start_pl = time.perf_counter()
        _ = client.get_historical_features(
            entity_df=obs_df,
            features=["user_stats:metric_a"],
            output_format="polars",
            cache=False,
        )
        time_pl = time.perf_counter() - start_pl

        print(f"   Pandas DataFrame output time: {time_pd:.4f} seconds")
        print(f"   Polars DataFrame output time: {time_pl:.4f} seconds")
    else:
        print("\n2. Polars is not installed (skipping Pandas vs Polars benchmark).")

    # Benchmark 3: SQLite Online Serving Latency
    print("\n3. Serving Online Low-Latency serving benchmark...")
    # Materialize data
    client.materialize(
        datetime.datetime(2026, 7, 14, 0, 0, 0),
        datetime.datetime(2026, 7, 14, 23, 59, 59),
    )

    # Query online store for 100 random user retrievals
    serving_keys = list(range(1, 101))
    start_serve = time.perf_counter()
    _ = client.get_online_features(
        entity_keys=serving_keys,
        features=["user_stats:metric_a", "user_stats:metric_b"],
    )
    time_serve = time.perf_counter() - start_serve
    latency_per_key_ms = (time_serve / 100) * 1000
    print(f"   SQLite Serving time for 100 users: {time_serve:.4f} seconds")
    print(f"   Latency per user retrieval:       {latency_per_key_ms:.2f} ms")

    # Cleanup temp benchmark directory
    try:
        import shutil

        shutil.rmtree(temp_dir)
    except Exception:
        pass


if __name__ == "__main__":
    main()
