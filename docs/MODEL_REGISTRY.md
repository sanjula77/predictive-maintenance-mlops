# Model Registry & Versioning Guide

This document explains the model registry and versioning system.

## Overview

The model registry automatically manages model versions with:
- Model weights (`.pth`)
- Preprocessing scaler (`.pkl`)
- Metadata (`.json`) with metrics, config, and timestamp

## Quick Start

```bash
# Train with automatic versioning
python -m src.train_with_versioning --model lstm --epochs 20

# List all versions
python -m src.list_models

# Load a specific version
python -m src.load_model_version --version 1 --model-type lstm
```

## Detailed Usage

See [MODEL_REGISTRY_USAGE.md](MODEL_REGISTRY_USAGE.md) for complete documentation.

