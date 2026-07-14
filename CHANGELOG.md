# Changelog

All notable changes to the FTLite project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.2.0] - 2026-07-14

### Added
- **Feature Versioning**: Optional `version` support on feature views (e.g. `user_fv@v1`, `user_fv@v2`), allowing query version suffixes and default latest-version fallback lookups.
- **Optional Polars Support**: Introduced lazy Polars dataframe format support (`output_format="polars"`) behind the `ftlite[polars]` optional install extra.
- **Local Join Caching**: Hash-based local `.ftlite/cache/` storage for historical joins with 24h TTL eviction policy and maximum 100 cache files quota.
- **Cache Eviction CLI**: New `ftlite cache-clear` command to clear local cache.
- **Feature Lineage Tracing**: Recursive dependency lineage traversal graph API and CLI (`ftlite lineage <feature_name>`).
- **Comprehensive Examples**: Added 5 machine learning example applications (Churn, Fraud, Housing, Recommendations, Forecasting) and a custom comparative performance benchmarking script.

### Removed
- Unused experimental drift metrics (PSI) to maintain a lightweight core package positioning.

---

## [0.1.3] - 2026-07-14

### Changed
- Improved package metadata and README documentation detailing local installation.

---

## [0.1.2] - 2026-07-14

### Fixed
- Registry serialization updates to support versioned feature views.

---

## [0.1.1] - 2026-07-14

### Fixed
- Fixed float-upcasting bug in Pandas historical joins; serving integer types now correctly retains nullable integer formats (`Int64`).

---

## [0.1.0] - 2026-07-14

### Added
- Initial release of FTLite: A lightweight, local-first feature store for ML.
- Basic Entity, Feature, FeatureView, and OnDemandFeatureView registrations.
- DuckDB-powered point-in-time correct historical joins.
- SQLite-powered low-latency online serving.
- Click-based command-line interface.
