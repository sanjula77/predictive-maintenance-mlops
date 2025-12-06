# CI/CD Setup Guide

This document describes the Continuous Integration and Continuous Deployment setup for the Predictive Maintenance MLOps project.

## Overview

The project includes automated CI/CD pipelines using GitHub Actions for:
- Code quality checks
- Automated testing
- Docker image building
- Security scanning
- API testing

## Development Dependencies

Install development dependencies:

```bash
pip install -r requirements-dev.txt
```

### Included Tools

- **Testing**: pytest, pytest-cov, pytest-asyncio, pytest-mock, httpx
- **Code Quality**: black, flake8, pylint, mypy, isort
- **Security**: safety (dependency scanning)
- **Documentation**: sphinx

## Testing Infrastructure

### Test Structure

```
tests/
├── conftest.py              # Pytest fixtures and configuration
├── test_config.py           # Configuration tests
├── test_models.py           # Model architecture tests
├── test_api.py              # FastAPI endpoint tests
├── test_data_preprocessing.py  # Data preprocessing tests
└── test_utils.py            # Utility function tests
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_api.py

# Run with verbose output
pytest -v
```

## Code Quality Tools

### Black (Code Formatting)

```bash
# Format code
black src/ tests/

# Check formatting
black --check src/ tests/
```

### Flake8 (Linting)

```bash
# Run linter
flake8 src/ tests/
```

### MyPy (Type Checking)

```bash
# Type check
mypy src/
```

### Isort (Import Sorting)

```bash
# Sort imports
isort src/ tests/
```

## GitHub Actions Workflows

### 1. CI - Lint & Test (`ci.yml`)

Runs on every push and pull request:
- Code formatting check (Black)
- Linting (Flake8)
- Type checking (MyPy)
- Unit tests (Pytest)
- Coverage reporting

### 2. Build Docker Image (`docker-build.yml`)

Runs on push to main:
- Builds Docker image
- Tests Docker build
- Optionally pushes to registry

### 3. Test API (`test-api.yml`)

Runs on API-related changes:
- Starts API in Docker
- Tests all endpoints
- Validates responses

### 4. Security Checks (`security.yml`)

Runs weekly:
- Dependency vulnerability scanning
- Code security analysis

## Local CI/CD Simulation

Run CI checks locally before pushing:

```bash
# Format code
black src/ tests/

# Lint
flake8 src/ tests/

# Type check
mypy src/

# Run tests
pytest --cov=src

# Security scan
safety check
```

## Configuration Files

- **`pytest.ini`**: Pytest configuration and coverage settings
- **`pyproject.toml`**: Tool configurations (Black, isort, mypy, pytest)
- **`.flake8`**: Flake8 linting rules
- **`.github/workflows/`**: GitHub Actions workflows

## Best Practices

1. **Run tests before committing**: `pytest`
2. **Format code**: `black src/ tests/`
3. **Check types**: `mypy src/`
4. **Fix linting issues**: `flake8 src/ tests/`
5. **Update dependencies**: Regularly update `requirements-dev.txt`

## Troubleshooting

### Tests Failing

```bash
# Run with verbose output
pytest -v

# Run specific test
pytest tests/test_api.py::test_health_check -v

# Check coverage
pytest --cov=src --cov-report=term-missing
```

### Type Checking Issues

```bash
# Check specific file
mypy src/api/main.py

# Ignore missing imports
mypy src/ --ignore-missing-imports
```

### Linting Errors

```bash
# Show all errors
flake8 src/ --show-source

# Ignore specific rules
flake8 src/ --ignore=E501,W503
```

