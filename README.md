# Predictive Maintenance MLOps

A production-ready MLOps pipeline for predictive maintenance using Remaining Useful Life (RUL) prediction on CMAPSS aircraft engine data.

## 🚀 Features

- **Multiple Model Architectures**: LSTM, BiLSTM, GRU, and Transformer models
- **Model Versioning**: Automatic version management with metadata tracking
- **Production-Ready Scripts**: Clean, modular codebase following best practices
- **REST API**: FastAPI-based inference API
- **Comprehensive Evaluation**: Built-in model comparison and metrics

## 📁 Project Structure

```
predictive-maintenance-mlops/
├── src/                    # Source code
│   ├── data/              # Data loading and preprocessing
│   ├── models/            # Model architectures
│   ├── api/               # FastAPI application
│   ├── train.py           # Training script
│   ├── evaluate.py        # Evaluation script
│   ├── predict.py         # Prediction script
│   └── model_registry.py  # Model versioning system
├── notebooks/             # Jupyter notebooks for exploration
├── data/                  # Data directory
│   ├── raw/              # Raw data files
│   └── processed/        # Processed data
├── models/               # Trained models (versioned)
│   └── v1/, v2/, ...    # Version directories
├── tests/                # Unit tests
├── docs/                 # Documentation
└── requirements.txt      # Python dependencies
```

## 🛠️ Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd predictive-maintenance-mlops
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

## 📊 Quick Start

### 1. Train a Model

```bash
# Train with automatic versioning (recommended)
python -m src.train_with_versioning --model lstm --epochs 20

# Or train without versioning
python -m src.train --model lstm --epochs 20
```

### 2. Evaluate Model

```bash
python -m src.evaluate --model lstm
```

### 3. List All Model Versions

```bash
python -m src.list_models
```

### 4. Make Predictions

```bash
python -m src.predict --model lstm --input data.csv --output predictions.csv
```

## 🌐 API Inference Service

### Quick Start

```bash
# Start the API server
uvicorn src.api.main:app --reload

# Or using Docker
docker-compose up
```

### API Endpoints

- **Health Check**: `GET /` or `GET /health`
- **List Models**: `GET /models`
- **Model Info**: `GET /models/{version}`
- **Predict RUL**: `POST /predict?version=1&model_type=lstm`

### Example Request

```bash
curl -X POST "http://localhost:8000/predict?version=1&model_type=lstm" \
  -H "Content-Type: application/json" \
  -d @example_request.json
```

### Interactive Documentation

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 📚 Documentation

- [Usage Guide](docs/USAGE.md) - Detailed usage instructions
- [Model Registry](docs/MODEL_REGISTRY.md) - Model versioning system
- [API Documentation](docs/API.md) - Complete API reference
- [Deployment Guide](docs/DEPLOYMENT.md) - Deploy to cloud platforms

## 🧪 Model Architectures

- **LSTM**: Standard LSTM for sequence prediction
- **BiLSTM**: Bidirectional LSTM for better context
- **GRU**: Gated Recurrent Unit (faster training)
- **Transformer**: Transformer encoder for attention-based prediction

## 📈 Model Versioning

Models are automatically versioned with:
- Model weights (`.pth`)
- Preprocessing scaler (`.pkl`)
- Metadata (`.json`) with metrics, config, and timestamp

```bash
# Register an existing model
python -m src.register_model --model-path models/rul_lstm.pth --model-type lstm --rmse 24.04 --mae 16.81

# List all versions
python -m src.list_models

# Load a specific version
python -m src.load_model_version --version 1 --model-type lstm
```

## 🔧 Configuration

All configuration is centralized in `src/config.py`:
- Hyperparameters (epochs, batch size, learning rate)
- Model architecture parameters
- Data paths
- Feature columns

## 🧪 Testing

```bash
# Run tests
pytest tests/
```

## 📝 License

[Add your license here]

## 🤝 Contributing

[Add contributing guidelines]

## 📧 Contact

[Add contact information]

