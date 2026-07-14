# Contributing to FTLite

First off, thank you for considering contributing to FTLite! It's people like you who make open source a great place to build tools.

---

## Code of Conduct

By participating in this project, you agree to abide by our Contributor Covenant Code of Conduct (detailed in `CODE_OF_CONDUCT.md`).

---

## Local Development Setup

To set up FTLite locally for development:

1. **Fork and Clone** the repository:
   ```bash
   git clone https://github.com/<your-username>/ftlite.git
   cd ftlite
   ```

2. **Create a Virtual Environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install Dependencies** (including development and optional extras):
   ```bash
   pip install -e .[dev,polars]
   ```

---

## Coding Standards & Style

We use `black` for formatting and `ruff` for linting. Please run these check commands before submitting any pull request:

* **Format Code**:
  ```bash
  black .
  ```
* **Lint Check**:
  ```bash
  ruff check .
  ```

---

## Running Unit Tests

We use `pytest` for all unit testing. Run the test suite using:

```bash
pytest -v
```

Ensure all tests pass successfully before committing any code changes.
