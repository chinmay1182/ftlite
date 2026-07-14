# FTLite

> FTLite is a batteries-included, local-first feature store for Python that helps you build reproducible machine learning pipelines without Kubernetes, Redis, or cloud infrastructure.

Whether you're training a churn model, fraud detector, recommendation system, or forecasting pipeline, FTLite helps ensure your training features match what would have been available at prediction time—preventing data leakage while keeping the developer experience simple.

---

## Why FTLite?

Modern feature stores are powerful, but often require significant infrastructure and operational overhead.

FTLite is designed for:

* Individual developers
* Researchers
* Students
* Startups
* Small ML teams
* Local-first workflows

No servers.

No Kubernetes.

No metadata databases.

Just Python.

---

## Features

* Lightweight feature definitions
* Point-in-time correct historical feature retrieval
* Offline feature store powered by DuckDB + Parquet
* Online feature store powered by SQLite
* Zero infrastructure setup
* Local feature registry
* Simple Python API
* Fast local development
* Type hints and clean API design

---

## Architecture

```text
                    +----------------+
                    |  Raw Features  |
                    +----------------+
                            |
                            v
                    +----------------+
                    | Offline Store  |
                    | DuckDB/Parquet |
                    +----------------+
                            |
          Historical        | Materialize
         Feature Joins      v
                    +----------------+
                    | Online Store   |
                    |    SQLite      |
                    +----------------+
                            |
                            v
                      Model Inference
```

---

## Project Structure

```text
ftlite/
│
├── ftlite/
│   ├── __init__.py
│   ├── feature.py          # Entity, Feature, FeatureView, OnDemandFeatureView
│   ├── offline_store.py    # DuckDB + temporal ASOF joins
│   ├── online_store.py     # SQLite online serving store
│   ├── registry.py         # tracks and persists metadata
│   ├── ingestion.py        # appends features to Parquet files
│   ├── client.py           # main FtliteClient orchestrator
│   └── cli.py              # CLI entry point
│
├── tests/
│   ├── __init__.py
│   └── test_client.py
│
├── examples/
│   └── churn_example/
│       ├── __init__.py
│       ├── generate_data.py
│       └── run_churn_store.py
│
├── docs/
│   └── index.md
│
├── pyproject.toml
├── README.md
└── LICENSE
```

---

## Installation

```bash
pip install ftlite
```

Or install from source:

```bash
git clone https://github.com/<your-username>/ftlite.git

cd ftlite

pip install -e .
```

---

## Quick Example

### Define and Register Entity & Features

```python
from ftlite import Entity, Feature, FeatureView, FtliteClient

# Initialize client
client = FtliteClient()

# Define Entity
customer = Entity(
    name="customer_id",
    value_type="INT64"
)
client.register_entity(customer)

# Define Features and FeatureView
age = Feature(name="age", dtype="int32")
balance = Feature(name="balance", dtype="float64")

fv = FeatureView(
    name="customer_fv",
    entities=[customer],
    features=[age, balance],
    source_path="data/customer_features.parquet",
    timestamp_field="timestamp"
)
client.register_feature_view(fv)
```

### Historical Features (Point-in-Time Join)

```python
# Join historical features to observation entity data
historical_df = client.get_historical_features(
    entity_df=entity_df,
    features=[
        "customer_fv:age",
        "customer_fv:balance"
    ],
    timestamp_col="timestamp"
)
```

### Online Features (Low-Latency Serving)

```python
# Retrieve online features for real-time predictions
online_features = client.get_online_features(
    entity_keys=[1001],
    features=[
        "customer_fv:age",
        "customer_fv:balance"
    ]
)
```

---

# Detailed Documentation

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

## 7. Feature Lineage Tracing

Trace upstream features, intermediate on-demand transform functions, and source parquet files:

```python
lineage = client.get_feature_lineage("calls_per_minute_transform:calls_per_minute")
# Returns a structured dictionary describing the transformation path.
```

Or trace features interactively via the command line.

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

---

# Tech Stack

| Component     | Technology     |
| ------------- | -------------- |
| Language      | Python         |
| Offline Store | DuckDB         |
| Storage       | Parquet        |
| Online Store  | SQLite         |
| Testing       | Pytest         |
| Formatting    | Ruff + Black   |
| CI            | GitHub Actions |

---

# Example Use Cases

* Customer churn prediction
* Fraud detection
* Recommendation systems
* Credit risk modeling
* Demand forecasting
* Time-series feature engineering

---

# Contributing

Contributions are welcome!

If you'd like to contribute:

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Open a Pull Request

Please make sure all tests pass before submitting.

---

# Vision

FTLite aims to become the simplest feature store for Python.

Instead of managing infrastructure, developers should spend their time building better machine learning models.

---

# License

MIT License.

---

## ⭐ Support the Project

If you find FTLite useful, please consider giving the repository a ⭐ on GitHub. It helps others discover the project and supports future development.
