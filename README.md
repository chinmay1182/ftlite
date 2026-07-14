# FTLite

> A lightweight, zero-infrastructure feature store for machine learning.

FTLite is an open-source Python library that makes feature management simple without requiring Kubernetes, Spark, Redis, or cloud infrastructure. It focuses on one thing that matters most for production ML: **point-in-time correct feature retrieval**.

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

# Roadmap

## Phase 1

* Entity definitions
* Feature definitions
* Local registry

## Phase 2

* Offline feature store
* Historical feature retrieval
* Point-in-time joins
* Training dataset generation

## Phase 3

* SQLite online store
* Materialization
* Online inference API

## Phase 4

* CLI
* Better documentation
* Real-world examples
* Type hints
* Unit tests

## Phase 5

* Redis backend
* Feature transformations
* Feature validation
* Versioning
* PyPI release

---

# Design Goals

* Lightweight
* Local-first
* Easy to understand
* Zero infrastructure
* Production-friendly
* Strong correctness guarantees
* Great developer experience

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
