# Project Structure

This document describes the organization of the Predictive Maintenance MLOps project.

## Directory Layout

```
predictive-maintenance-mlops/
├── src/                          # Source code
│   ├── __init__.py
│   ├── config.py                # Configuration constants
│   ├── utils/                   # Utility functions
│   │   └── __init__.py
│   ├── data/                    # Data handling
│   │   ├── __init__.py
│   │   ├── load_data.py        # Data loading functions
│   │   └── preprocessing.py    # Preprocessing functions
│   ├── models/                  # Model architectures
│   │   ├── __init__.py
│   │   └── architectures.py   # Model definitions
│   ├── api/                     # FastAPI application
│   │   ├── main.py             # API endpoints
│   │   └── schemas.py          # Pydantic schemas
│   ├── train.py                # Training script
│   ├── train_with_versioning.py # Training with versioning
│   ├── train_all.py            # Train all models
│   ├── evaluate.py              # Evaluation script
│   ├── predict.py              # Prediction script
│   ├── compare_models.py       # Model comparison
│   ├── model_registry.py       # Model versioning
│   ├── register_model.py       # Register model version
│   ├── list_models.py          # List model versions
│   └── load_model_version.py   # Load model version
├── notebooks/                   # Jupyter notebooks
│   └── 01_dataset_exploration.ipynb
├── data/                        # Data directory
│   ├── raw/                    # Raw data files
│   └── processed/              # Processed data
├── models/                     # Trained models
│   ├── v1/                     # Version 1
│   │   ├── model.pth
│   │   ├── scaler.pkl
│   │   └── metadata.json
│   └── v2/                     # Version 2
│       └── ...
├── tests/                       # Unit tests
├── docs/                        # Documentation
│   ├── USAGE.md
│   ├── MODEL_REGISTRY.md
│   ├── TRAINING.md
│   └── API.md
├── .gitignore                   # Git ignore rules
├── requirements.txt             # Python dependencies
├── setup.py                     # Package setup
├── README.md                    # Main documentation
├── LICENSE                      # License file
└── CONTRIBUTING.md              # Contributing guidelines
```

## Key Files

### Configuration
- `src/config.py` - Centralized configuration

### Training Scripts
- `src/train.py` - Basic training
- `src/train_with_versioning.py` - Training with versioning
- `src/train_all.py` - Train all models

### Model Management
- `src/model_registry.py` - Versioning system core
- `src/register_model.py` - Register models
- `src/list_models.py` - List versions
- `src/load_model_version.py` - Load versions

### Evaluation & Prediction
- `src/evaluate.py` - Evaluate models
- `src/predict.py` - Make predictions
- `src/compare_models.py` - Compare models

### API
- `src/api/main.py` - FastAPI application

## Best Practices

1. **Modularity**: Each module has a single responsibility
2. **Configuration**: All config in `src/config.py`
3. **Versioning**: Models automatically versioned
4. **Documentation**: Docstrings in all functions
5. **Type Hints**: Used throughout codebase
6. **Testing**: Tests in `tests/` directory

