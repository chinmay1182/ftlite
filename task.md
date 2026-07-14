# FTLite Implementation Checklist

- [x] Feature Transformations (On-Demand Feature Views)
  - [x] Define `OnDemandFeatureView` in `feature.py`
  - [x] Add serialization & registry support in `registry.py`
  - [x] Resolve dependencies & apply transformations in `client.py`
- [x] Command-Line Interface (CLI)
  - [x] Move `click` to main dependencies and define script entry point in `pyproject.toml`
  - [x] Create `cli.py` with init, materialize, list commands
- [x] Packaging & CI
  - [x] Create `LICENSE` file
  - [x] Create GitHub Actions workflow file `.github/workflows/ci.yml`
- [x] Advanced Testing
  - [x] Expand `tests/test_client.py` with edge cases, multiple entities, and on-demand transformations
  - [x] Run test suite & verify churn example runs
