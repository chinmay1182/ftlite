# FTLite

**The simplest local-first feature store for Python.**

*Define features once. Train models without data leakage. Serve features in milliseconds.*

**No Kubernetes. No Redis. No Cloud. Just Python.**

[![PyPI Version](https://img.shields.io/pypi/v/ftlite.svg)](https://pypi.org/project/ftlite/)
[![License](https://img.shields.io/pypi/l/ftlite.svg)](https://github.com/chinmay1182/ftlite/blob/main/LICENSE)

---

## Why do I need a Feature Store?

| Without FTLite | With FTLite |
| :--- | :--- |
| ❌ Feature engineering logic is duplicated across train & serve | ✅ **Define features once** and reuse them everywhere |
| ❌ Training and inference compute different feature values | ✅ **Ensure consistency** between offline & online models |
| ❌ Data leakage silently hurts real-world model performance | ✅ **Prevent data leakage** with temporal point-in-time joins |
| ❌ Feature versions are difficult to manage and swap | ✅ **Track lineage & version** features out-of-the-box |

---

## FTLite vs. Feast

| Capability | FTLite | Feast |
| :--- | :---: | :---: |
| **Local-first / Zero-infrastructure** | ✅ | ⚠️ *Complex* |
| **Kubernetes Required** | ❌ | *Often* |
| **Redis Required** | ❌ | *Usually* |
| **DuckDB (Offline Joins)** | ✅ | ❌ |
| **SQLite (Online Serving)** | ✅ | ❌ |
| **Point-in-Time Joins** | ✅ | ✅ |
| **Feature Versioning** | ✅ | ✅ |
| **Zero Setup** | ✅ | ❌ |

---

## Architecture

<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 220" width="100%" height="auto">
  <defs>
    <linearGradient id="blueGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#4F46E5" />
      <stop offset="100%" stop-color="#3B82F6" />
    </linearGradient>
    <linearGradient id="greenGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#10B981" />
      <stop offset="100%" stop-color="#059669" />
    </linearGradient>
  </defs>
  <rect width="100%" height="100%" fill="#0F172A" rx="12"/>
  <line x1="230" y1="110" x2="310" y2="110" stroke="#475569" stroke-width="3" stroke-dasharray="6"/>
  <polygon points="310,105 320,110 310,115" fill="#475569"/>
  <path d="M 480 110 L 570 110" fill="none" stroke="#475569" stroke-width="3" stroke-dasharray="6"/>
  <polygon points="570,105 580,110 570,115" fill="#475569"/>
  <rect x="50" y="60" width="180" height="100" rx="10" fill="url(#blueGrad)"/>
  <text x="140" y="105" font-family="system-ui, sans-serif" font-weight="bold" font-size="16" fill="#FFFFFF" text-anchor="middle">Offline Data</text>
  <text x="140" y="130" font-family="system-ui, sans-serif" font-size="12" fill="#E0F2FE" text-anchor="middle">Local Parquet Files</text>
  <rect x="320" y="40" width="160" height="140" rx="10" fill="#1E293B" stroke="#334155" stroke-width="2"/>
  <text x="400" y="80" font-family="system-ui, sans-serif" font-weight="bold" font-size="18" fill="#F8FAFC" text-anchor="middle">FTLite Client</text>
  <text x="400" y="115" font-family="system-ui, sans-serif" font-size="13" fill="#38BDF8" text-anchor="middle">DuckDB (Offline)</text>
  <text x="400" y="145" font-family="system-ui, sans-serif" font-size="13" fill="#34D399" text-anchor="middle">SQLite (Online)</text>
  <rect x="580" y="60" width="170" height="100" rx="10" fill="url(#greenGrad)"/>
  <text x="665" y="105" font-family="system-ui, sans-serif" font-weight="bold" font-size="16" fill="#FFFFFF" text-anchor="middle">ML Application</text>
  <text x="665" y="130" font-family="system-ui, sans-serif" font-size="12" fill="#ECFDF5" text-anchor="middle">Training &amp; Inference</text>
</svg>

---

## Installation

```bash
pip install ftlite
```

To enable native Polars DataFrame outputs:
```bash
pip install ftlite[polars]
```

---

## 2-Minute Quick Example

### 1. Define and Register Features
```python
from ftlite import Entity, Feature, FeatureView, FtliteClient

client = FtliteClient()

# Identify entity
customer = Entity(name="customer_id", value_type="INT64")
client.register_entity(customer)

# Define feature view mapping to local Parquet file
fv = FeatureView(
    name="customer_stats",
    entities=[customer],
    features=[
        Feature(name="balance", dtype="double"),
        Feature(name="active_days", dtype="int64")
    ],
    source_path="customer_data.parquet",
    timestamp_field="timestamp"
)
client.register_feature_view(fv)
```

### 2. Historical Join (Prevent Data Leakage)
```python
# Pass raw entity observations and timestamps
obs_df = pd.DataFrame({
    "customer_id": [1001],
    "timestamp": ["2026-07-14T12:00:00"]
})

hist_df = client.get_historical_features(
    entity_df=obs_df,
    features=["customer_stats:balance", "customer_stats:active_days"]
)
```

#### Visualizing Point-in-Time Join:
```text
Input (Entity Observations):
| customer_id | timestamp           |
| ----------- | ------------------- |
| 1001        | 2026-07-14T12:00:00 |

Output (PIT Correct Joined Features):
| customer_id | timestamp           | customer_stats:balance | customer_stats:active_days |
| ----------- | ------------------- | ---------------------- | -------------------------- |
| 1001        | 2026-07-14T12:00:00 | 5250.75                | 14                         |
```

### 3. Materialize & Serve Low-Latency Online Features
```python
import datetime

# Sync last 30 days offline features to SQLite serving store
client.materialize(
    start_time=datetime.datetime(2026, 6, 15),
    end_time=datetime.datetime(2026, 7, 15)
)

# Fetch low-latency prediction features in <1ms
online_features = client.get_online_features(
    entity_keys=[1001],
    features=["customer_stats:balance"]
)
# Returns: [{'entity_id': 1001, 'customer_stats:balance': 5250.75}]
```

---

## Feature List

* ✅ **Point-in-Time Correct Joins**: Built on DuckDB temporal ASOF joins to prevent data leakage.
* ✅ **Low-Latency Online Serving**: SQLite-backed serving engine fetching features in sub-milliseconds.
* ✅ **One-Line Feature Versioning**: Track feature versions explicitly with automatic fallbacks.
* ✅ **Zero-Configuration Caching**: Automatically cache historical query computations.
* ✅ **Feature Lineage Tracing**: View recursive upstream dependencies easily via CLI or Python API.
* ✅ **Optional Polars Support**: Perform lightning-fast queries with native Polars outputs.
* ✅ **Python-First API**: Simple object-oriented structures with 100% type annotations.

---

## Complete ML Examples

We provide fully self-contained ML examples inside the `examples/` directory:

* 📊 **[Customer Churn](file:///c:/Users/Admin/Documents/ftlite/examples/customer_churn.py)**: End-to-end model training and online inference.
* 🛡️ **[Fraud Detection](file:///c:/Users/Admin/Documents/ftlite/examples/fraud_detection.py)**: Dynamic query-time On-Demand transformations.
* 🏠 **[House Price Prediction](file:///c:/Users/Admin/Documents/ftlite/examples/house_price_prediction.py)**: Zip-code statistics using temporal joins.
* 🔍 **[Recommendation System](file:///c:/Users/Admin/Documents/ftlite/examples/recommendation_system.py)**: Version fallback and lookup syntaxes.
* 📈 **[Time Series Forecasting](file:///c:/Users/Admin/Documents/ftlite/examples/time_series_forecasting.py)**: Target lag feature generations.
* ⚡ **[Performance Benchmarks](file:///c:/Users/Admin/Documents/ftlite/examples/benchmark.py)**: Cache hit vs. miss speedups, and Pandas vs. Polars extraction timings.

---

## Detailed Documentation

For comprehensive guides on On-Demand Transformations, versioning patterns, cache management, lineage tracking, and command-line scripts, refer to the **[Detailed Guide & API Reference](file:///c:/Users/Admin/Documents/ftlite/docs/detailed_guide.md)**.

---

## License

MIT License.
