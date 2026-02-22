# CI/CD Pipelines

## CI Pipeline (`ci.yml`)

Triggered on: Pull requests and pushes to `main`.

| Stage | Tool | Description | Fails if |
|-------|------|-------------|----------|
| **Lint** | `ruff` | Code style and import order | Any violation |
| **Type Check** | `mypy --strict` | Static type analysis | Any error |
| **Unit Tests** | `pytest` | Python 3.11 + 3.12 matrix | Any failure |
| **Coverage** | `pytest-cov` | Line coverage report | Coverage < 80% |
| **Build** | `python -m build` | Package build + install verify | Build fails |

## CD Pipeline (`cd.yml`)

Triggered on: Push to `main` (excludes docs-only changes).

| Stage | Tool | Description |
|-------|------|-------------|
| **Deploy** | `huggingface_hub` | Upload to HuggingFace Space |
| **Smoke Test** | `urllib` | Verify space returns HTTP 200 |

## Required Secrets

| Secret | Description |
|--------|-------------|
| `HF_TOKEN` | HuggingFace write token for Space deployment |

## Local Debugging

```bash
# Run the same CI checks locally
ruff check mangomas_demo/ tests/
mypy mangomas_demo/
pytest tests/ -v --cov=mangomas_demo --cov-fail-under=80
pip install -e . && python -c "import mangomas_demo; print('ok')"
```
