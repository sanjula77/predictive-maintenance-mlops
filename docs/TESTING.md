# Testing Documentation

## Table of Contents

- [Overview](#overview)
- [Test Structure](#test-structure)
- [Running Tests](#running-tests)
- [Test Coverage](#test-coverage)
- [Writing Tests](#writing-tests)
- [Best Practices](#best-practices)
- [CI/CD Integration](#cicd-integration)
- [Troubleshooting](#troubleshooting)

---

## Overview

This project uses **pytest** as the primary testing framework, following industry best practices for Python testing. The test suite ensures code quality, reliability, and maintainability through comprehensive unit and integration tests.

### Testing Philosophy

- **Test-Driven Development (TDD)**: Write tests before or alongside implementation
- **Comprehensive Coverage**: Aim for high coverage of critical business logic
- **Fast Execution**: Tests should run quickly to enable rapid feedback
- **Isolation**: Tests should be independent and not rely on external state
- **Clarity**: Test names and structure should clearly express intent

### Test Framework Stack

- **pytest**: Primary testing framework
- **pytest-cov**: Coverage reporting
- **pytest-asyncio**: Async test support
- **pytest-mock**: Mocking utilities
- **httpx**: HTTP client for API testing

---

## Test Structure

### Directory Layout

```
tests/
├── __init__.py                 # Test package initialization
├── conftest.py                 # Shared fixtures and configuration
├── test_api.py                 # API endpoint tests
├── test_config.py              # Configuration tests
├── test_data_preprocessing.py  # Data preprocessing tests
├── test_models.py              # Model architecture tests
└── test_utils.py               # Utility function tests
```

### Test Organization

Tests are organized by module/component:

- **`test_config.py`**: Configuration validation and path checks
- **`test_models.py`**: Model architecture creation and forward pass tests
- **`test_data_preprocessing.py`**: Data transformation and sequence generation
- **`test_api.py`**: FastAPI endpoint integration tests
- **`test_utils.py`**: Utility function tests (device detection, seeding)

### Test Naming Convention

- **Files**: `test_<module_name>.py`
- **Classes**: `Test<ComponentName>`
- **Methods**: `test_<functionality>_<expected_behavior>`

Example:
```python
class TestModelArchitectures:
    def test_lstm_creation(self):
        """Test LSTM model can be created."""
        ...
```

---

## Running Tests

### Prerequisites

Install development dependencies:

```bash
pip install -r requirements-dev.txt
```

### Basic Test Execution

Run all tests:
```bash
pytest tests/ -v
```

Run specific test file:
```bash
pytest tests/test_models.py -v
```

Run specific test class:
```bash
pytest tests/test_config.py::TestConfigHyperparameters -v
```

Run specific test method:
```bash
pytest tests/test_models.py::TestModelArchitectures::test_lstm_creation -v
```

### Test Execution Options

**Verbose output:**
```bash
pytest tests/ -v
```

**Show print statements:**
```bash
pytest tests/ -v -s
```

**Stop on first failure:**
```bash
pytest tests/ -v -x
```

**Run tests matching pattern:**
```bash
pytest tests/ -v -k "test_lstm"
```

**Run tests in parallel (requires pytest-xdist):**
```bash
pytest tests/ -v -n auto
```

### Test Markers

Markers allow selective test execution:

```bash
# Run only unit tests
pytest tests/ -v -m unit

# Run only integration tests
pytest tests/ -v -m integration

# Skip slow tests
pytest tests/ -v -m "not slow"
```

Available markers (defined in `pytest.ini`):
- `@pytest.mark.unit`: Unit tests
- `@pytest.mark.integration`: Integration tests
- `@pytest.mark.slow`: Tests that take longer to run

---

## Test Coverage

### Running Coverage Reports

**Terminal report:**
```bash
pytest tests/ -v --cov=src --cov-report=term-missing
```

**HTML report:**
```bash
pytest tests/ -v --cov=src --cov-report=html
# Open htmlcov/index.html in browser
```

**XML report (for CI/CD):**
```bash
pytest tests/ -v --cov=src --cov-report=xml
```

**Combined report:**
```bash
pytest tests/ -v --cov=src --cov-report=term-missing --cov-report=html --cov-report=xml
```

### Coverage Goals

| Component | Target Coverage | Current Status |
|-----------|----------------|----------------|
| Core Models | 100% | ✅ 100% |
| Configuration | 100% | ✅ 100% |
| Utilities | 100% | ✅ 100% |
| Data Preprocessing | >80% | ⚠️ 57% |
| API Endpoints | >70% | ⚠️ 28% |
| Scripts | N/A | N/A (CLI tools) |

### Coverage Interpretation

- **100%**: All code paths tested
- **80-99%**: Good coverage, minor gaps acceptable
- **50-79%**: Adequate, but improvements needed
- **<50%**: Insufficient, requires attention

**Note**: 100% coverage doesn't guarantee bug-free code. Focus on testing critical business logic and edge cases.

---

## Writing Tests

### Test Structure

Follow the **AAA Pattern** (Arrange, Act, Assert):

```python
def test_example(self):
    """Test description."""
    # Arrange: Set up test data and conditions
    input_data = create_test_data()
    expected_output = 42
    
    # Act: Execute the code under test
    result = function_under_test(input_data)
    
    # Assert: Verify the results
    assert result == expected_output
```

### Test Fixtures

Use `conftest.py` for shared fixtures:

```python
# conftest.py
@pytest.fixture
def sample_data_path():
    """Return path to test data directory."""
    return Path(__file__).parent.parent / "data" / "raw"

# test_file.py
def test_something(sample_data_path):
    """Use the fixture."""
    data_file = sample_data_path / "test.txt"
    assert data_file.exists()
```

### Testing Models

Example model test:

```python
def test_lstm_forward(self, sample_input):
    """Test LSTM forward pass produces correct output shape."""
    # Arrange
    model = RUL_LSTM(input_size=sample_input.shape[2])
    model.eval()
    
    # Act
    with torch.no_grad():
        output = model(sample_input)
    
    # Assert
    assert output.shape == (sample_input.shape[0], 1)
    assert not torch.isnan(output).any()
```

### Testing API Endpoints

Example API test:

```python
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

def test_health_check(self):
    """Test health endpoint returns 200."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] in ["healthy", "ok"]
```

### Testing with Mocks

Use mocks to isolate units under test:

```python
from unittest.mock import patch, MagicMock

def test_function_with_external_call(self):
    """Test function that calls external service."""
    with patch('module.external_service') as mock_service:
        mock_service.return_value = "mocked_result"
        result = function_under_test()
        assert result == "expected"
        mock_service.assert_called_once()
```

### Testing Exceptions

Test error handling:

```python
def test_invalid_input_raises_error(self):
    """Test that invalid input raises appropriate error."""
    with pytest.raises(ValueError, match="Invalid input"):
        function_under_test(invalid_data)
```

---

## Best Practices

### 1. Test Independence

- ✅ Each test should be independent
- ✅ Tests should not rely on execution order
- ✅ Clean up after tests (use fixtures)

```python
# Good: Independent test
def test_calculation(self):
    result = calculate(5, 3)
    assert result == 8

# Bad: Depends on previous test
def test_second_calculation(self):
    # Assumes previous test ran
    assert global_state == expected
```

### 2. Descriptive Test Names

- ✅ Use clear, descriptive names
- ✅ Follow pattern: `test_<what>_<condition>_<expected_result>`

```python
# Good
def test_calculate_rul_returns_positive_values(self):
    ...

# Bad
def test_rul(self):
    ...
```

### 3. One Assertion Per Test (When Possible)

- ✅ Focus each test on one behavior
- ✅ Use multiple assertions only when testing related conditions

```python
# Good: Single behavior
def test_model_output_shape(self):
    output = model(input)
    assert output.shape == expected_shape

# Also Good: Related assertions
def test_model_output_valid(self):
    output = model(input)
    assert output.shape == expected_shape
    assert not torch.isnan(output).any()
    assert output.min() >= 0
```

### 4. Use Fixtures for Setup

- ✅ Extract common setup into fixtures
- ✅ Use fixture scopes appropriately

```python
@pytest.fixture
def trained_model():
    """Create and return a trained model."""
    model = create_model()
    train_model(model)
    return model

def test_prediction(trained_model):
    """Test prediction with trained model."""
    result = trained_model.predict(test_data)
    assert result is not None
```

### 5. Test Edge Cases

- ✅ Test boundary conditions
- ✅ Test error conditions
- ✅ Test empty/null inputs

```python
def test_sequence_generation_empty_dataframe(self):
    """Test sequence generation handles empty dataframe."""
    empty_df = pd.DataFrame()
    with pytest.raises(ValueError):
        generate_sequences(empty_df, seq_len=10)
```

### 6. Keep Tests Fast

- ✅ Avoid slow operations in unit tests
- ✅ Use mocks for external services
- ✅ Mark slow tests appropriately

```python
@pytest.mark.slow
def test_full_training_pipeline(self):
    """Full training test - marked as slow."""
    # This test takes time
    ...
```

### 7. Document Test Purpose

- ✅ Use docstrings to explain test purpose
- ✅ Add comments for complex test logic

```python
def test_rul_calculation_decreasing(self):
    """
    Test that RUL decreases as cycles increase.
    
    For each engine, RUL should start at max_cycles and
    decrease to 0 as we approach the failure point.
    """
    ...
```

---

## CI/CD Integration

### Automated Testing

Tests run automatically on:
- **Push to main/develop branches**
- **Pull requests**
- **Scheduled runs** (optional)

### CI Pipeline Steps

1. **Linting**: Code quality checks (flake8, black, isort)
2. **Type Checking**: Static type analysis (mypy)
3. **Unit Tests**: Fast, isolated tests
4. **Integration Tests**: Component interaction tests
5. **Coverage Report**: Generate and upload coverage

### Viewing CI Results

1. Go to GitHub repository
2. Click **Actions** tab
3. Select workflow run
4. Review test results and coverage

### Local CI Simulation

Run all CI checks locally:

```bash
# 1. Linting
flake8 src --max-complexity=10
black --check src/ tests/
isort --check-only src/ tests/

# 2. Type checking
mypy src/ --ignore-missing-imports

# 3. Tests with coverage
pytest tests/ -v --cov=src --cov-report=xml --cov-report=term-missing
```

---

## Troubleshooting

### Common Issues

#### Tests Fail Locally but Pass in CI

**Possible causes:**
- Different Python versions
- Missing dependencies
- Environment variables not set

**Solution:**
```bash
# Ensure dev dependencies installed
pip install -r requirements-dev.txt

# Check Python version
python --version  # Should match CI (3.10)

# Run with verbose output
pytest tests/ -v -s
```

#### Import Errors

**Error:** `ModuleNotFoundError: No module named 'src'`

**Solution:**
```bash
# Run from project root
cd /path/to/predictive-maintenance-mlops
pytest tests/ -v

# Or install package in development mode
pip install -e .
```

#### Coverage Not Working

**Error:** Coverage shows 0%

**Solution:**
```bash
# Ensure pytest-cov is installed
pip install pytest-cov

# Run with explicit coverage
pytest tests/ --cov=src --cov-report=term
```

#### Tests Hang or Timeout

**Possible causes:**
- Infinite loops
- Waiting for external services
- Resource contention

**Solution:**
```bash
# Run with timeout
pytest tests/ --timeout=30

# Run specific test to isolate
pytest tests/test_specific.py::test_function -v -s
```

#### Flaky Tests

**Symptoms:** Tests pass/fail intermittently

**Common causes:**
- Race conditions
- Shared state
- Time-dependent logic

**Solution:**
- Use fixtures for setup/teardown
- Avoid global state
- Use deterministic test data
- Add retry logic if needed (use `pytest-rerunfailures`)

---

## Test Maintenance

### Regular Tasks

1. **Review Coverage Reports**: Monthly review of coverage trends
2. **Update Tests**: Keep tests in sync with code changes
3. **Remove Obsolete Tests**: Delete tests for removed features
4. **Refactor Tests**: Improve test clarity and maintainability

### Test Review Checklist

- [ ] All new features have corresponding tests
- [ ] Tests are fast (< 1 second per test)
- [ ] Tests are independent and can run in any order
- [ ] Test names clearly describe what they test
- [ ] Edge cases and error conditions are covered
- [ ] Coverage meets project standards
- [ ] No hardcoded paths or environment-specific values

---

## Additional Resources

### Documentation

- [pytest Documentation](https://docs.pytest.org/)
- [pytest-cov Documentation](https://pytest-cov.readthedocs.io/)
- [Python Testing Best Practices](https://docs.python-guide.org/writing/tests/)

### Tools

- **pytest**: Testing framework
- **pytest-cov**: Coverage reporting
- **pytest-xdist**: Parallel test execution
- **pytest-mock**: Mocking utilities
- **pytest-timeout**: Test timeout management

### Example Test Files

Refer to existing test files for examples:
- `tests/test_models.py`: Model testing patterns
- `tests/test_api.py`: API endpoint testing
- `tests/test_data_preprocessing.py`: Data transformation testing

---

## Contributing

When adding new features:

1. **Write tests first** (TDD approach) or alongside code
2. **Ensure tests pass** before submitting PR
3. **Maintain or improve coverage**
4. **Follow naming conventions**
5. **Add docstrings** to test methods

### Pull Request Checklist

- [ ] Tests added for new functionality
- [ ] All existing tests pass
- [ ] Coverage maintained or improved
- [ ] Tests follow project conventions
- [ ] Documentation updated if needed

---

**Last Updated**: 2024
**Maintained By**: Development Team

