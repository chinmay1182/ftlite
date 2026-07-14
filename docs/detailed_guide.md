# FTLite Detailed Guide & API Reference

This guide covers advanced usage, detailed API references, and configuration options for FTLite.

---

## 1. Feature Transformations (On-Demand Feature Views)

On-demand feature views allow you to perform runtime feature transformations using standard Python code, ensuring consistency between training and online serving.

```python
from ftlite import Feature, OnDemandFeatureView

# 1. Define transformation function
def calculate_calls_per_minute(df):
    df["calls_per_minute"] = df["support_calls"] / (df["usage_minutes"] + 1)
    return df

# 2. Register On-Demand Feature View
ondemand_fv = OnDemandFeatureView(
    name="calls_per_minute_transform",
    inputs=[
        "customer_activity:usage_minutes",
        "customer_activity:support_calls"
    ],
    features=[
        Feature(name="calls_per_minute", dtype="float64")
    ],
    transform_fn=calculate_calls_per_minute
)
client.register_on_demand_feature_view(ondemand_fv)
```

---

## 2. Ingesting Features

You can ingest new offline feature values dynamically using the `push` API. It automatically updates local Parquet data storage:

```python
import pandas as pd

new_data = pd.DataFrame({
    "customer_id": [1001, 1002],
    "timestamp": ["2026-07-14T12:00:00", "2026-07-14T12:00:00"],
    "usage_minutes": [145.2, 89.0],
    "support_calls": [0, 2],
    "active_days_in_week": [5, 3]
})

client.push(feature_view_name="customer_activity", df=new_data)
```

---

## 3. Materializing to Online Store

To use low-latency online feature serving, sync features from the offline store to the SQLite database over a given temporal window:

```python
import datetime

# Sync last 30 days of data
client.materialize(
    start_time=datetime.datetime(2026, 6, 1),
    end_time=datetime.datetime(2026, 7, 1)
)
```

---

## 4. Feature Versioning & Fallbacks

Define versioned feature views to support model experimentation. Queries automatically resolve to the latest version if no version suffix is provided.

```python
# Define v1 and v2 of a feature view
fv_v1 = FeatureView(
    name="customer_fv",
    version="v1",
    ...
)
fv_v2 = FeatureView(
    name="customer_fv",
    version="v2",
    ...
)

# Queries resolve to the latest version (v2) by default:
client.get_historical_features(entity_df, ["customer_fv:balance"])

# Or pin to specific versions using either notation:
client.get_historical_features(entity_df, ["customer_fv@v1:balance"])
client.get_historical_features(entity_df, ["customer_fv:balance@v1"])
```

---

## 5. Polars Support

Polars DataFrame integration is available as an optional install extra.

```bash
pip install ftlite[polars]
```

Pass or retrieve Polars relations natively:

```python
# Pass Polars DataFrame and get a Polars DataFrame back
hist_df = client.get_historical_features(
    entity_df=polars_entity_df,
    features=["customer_fv:balance"],
    output_format="polars"
)
```

---

## 6. Point-in-Time Join Caching

Save redundant execution steps by caching point-in-time correct joins locally:

```python
# Enables warm caching. Expired or excess files are self-evicted.
client.get_historical_features(
    entity_df,
    ["customer_fv:balance"],
    cache=True,
    cache_ttl=86400  # 24 hour TTL (default)
)
```

Programmatically or via CLI, clear the cache storage:

```python
client.clear_cache()
```

---

## 7. Feature Lineage Tracing

Trace upstream features, intermediate on-demand transform functions, and source parquet files:

```python
lineage = client.get_feature_lineage("calls_per_minute_transform:calls_per_minute")
# Returns a structured dictionary describing the transformation path.
```

Or trace features interactively via the command line.

---

## 8. Command Line Interface (CLI)

FTLite includes a command line utility to inspect your feature registry, clear cached files, and execute materialization commands.

```bash
# Initialize registry and workspace directory
ftlite init --registry path/to/registry.json --online-db path/to/online_store.db

# List all registered components
ftlite list --registry path/to/registry.json

# Trace dependency lineage of a feature
ftlite lineage customer_fv:balance

# Clear local parquet cache files
ftlite cache-clear

# Materialize features from the CLI
ftlite materialize \
  --registry path/to/registry.json \
  --online-db path/to/online_store.db \
  --start 2026-06-01T00:00:00 \
  --end 2026-07-01T00:00:00
```
