# FTLite Documentation

Welcome to FTLite documentation.

FTLite is a lightweight, zero-infrastructure feature store for machine learning.

## Key Concepts

- **Entity**: Represents the primary key used to identify features (e.g. `customer_id`, `driver_id`).
- **Feature**: An individual property or signal (e.g. `age`, `average_daily_spend`).
- **Feature View**: A collection of related features tied to one or more entities and a data source (e.g. a Parquet file).
- **Offline Store**: Stores historical features. Powered by DuckDB + Parquet. Used for generating training datasets with point-in-time correct joins.
- **Online Store**: Stores the latest feature values. Powered by SQLite. Used for low-latency feature serving at prediction time.
- **Registry**: Maintains definitions of entities and feature views, serialized to a local `.ftlite/registry.json` file.

## Getting Started

See [examples/churn_example/run_churn_store.py](file:///c:/Users/Admin/Documents/ftlite/examples/churn_example/run_churn_store.py) for a complete end-to-end example using a realistic customer churn dataset.

For an in-depth reference on all API capabilities (On-Demand Transformations, Versioning, Caching, Lineage, and CLI usage), read the [Detailed Guide](file:///c:/Users/Admin/Documents/ftlite/docs/detailed_guide.md).
