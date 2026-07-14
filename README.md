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
│   ├── feature.py
│   ├── offline_store.py
│   ├── online_store.py
│   ├── ingestion.py
│   ├── registry.py
│   ├── client.py
│   └── utils.py
│
├── tests/
├── examples/
├── docs/
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

### Define an Entity

```python
from ftlite import Entity

customer = Entity(
    name="customer_id",
    dtype="int64"
)
```

### Define Features

```python
from ftlite import Feature

age = Feature(
    name="age",
    dtype="int32",
    entity=customer
)

balance = Feature(
    name="balance",
    dtype="float64",
    entity=customer
)
```

### Register Features

```python
from ftlite import Registry

registry = Registry()

registry.register(age)
registry.register(balance)
```

### Historical Features

```python
client.get_historical_features(
    entity_df,
    features=[
        "age",
        "balance"
    ]
)
```

### Online Features

```python
client.get_online_features(
    entity_ids=[1001],
    features=[
        "age",
        "balance"
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
