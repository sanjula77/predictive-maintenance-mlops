# Predictive Maintenance MLOps

[![CI - Lint & Test](https://github.com/yourusername/predictive-maintenance-mlops/workflows/CI%20-%20Lint%20%26%20Test/badge.svg)](https://github.com/yourusername/predictive-maintenance-mlops/actions/workflows/ci.yml)
[![Build Docker Image](https://github.com/yourusername/predictive-maintenance-mlops/workflows/Build%20Docker%20Image/badge.svg)](https://github.com/yourusername/predictive-maintenance-mlops/actions/workflows/docker-build.yml)
[![Test API](https://github.com/yourusername/predictive-maintenance-mlops/workflows/Test%20API/badge.svg)](https://github.com/yourusername/predictive-maintenance-mlops/actions/workflows/test-api.yml)
[![Security Checks](https://github.com/yourusername/predictive-maintenance-mlops/workflows/Security%20Checks/badge.svg)](https://github.com/yourusername/predictive-maintenance-mlops/actions/workflows/security.yml)

> **Note**: Replace `yourusername` in the badge URLs with your actual GitHub username/organization.

A production-ready MLOps pipeline for predictive maintenance using Remaining Useful Life (RUL) prediction on CMAPSS aircraft engine data.

## 🚀 Features

- **Multiple Model Architectures**: LSTM, BiLSTM, GRU, and Transformer models
- **MLflow Integration**: Complete experiment tracking and model registry
- **Model Versioning**: Automatic version management with metadata tracking
- **Production-Ready API**: FastAPI-based REST API with Docker support
- **Auto-Promotion**: Automatically promotes best model to Production
- **Comprehensive Evaluation**: Built-in model comparison and metrics
- **Docker Support**: Fully containerized for easy deployment

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Training Models](#training-models)
- [API Inference Service](#api-inference-service)
- [Docker Deployment](#docker-deployment)
- [Documentation](#documentation)
- [Model Architectures](#model-architectures)
- [Contributing](#contributing)
- [License](#license)

## 🛠️ Installation

### Prerequisites

- Python 3.10+
- pip
- (Optional) CUDA-capable GPU for faster training
- (Optional) Docker for containerized deployment

### Setup

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

### 1. Train Models with MLflow (Recommended)

```bash
# Train all models and auto-promote best to Production
python -m src.train_with_mlflow --epochs 20 --auto-promote
```

This will:
- Train all 4 model architectures
- Track experiments in MLflow
- Register all models
- Compare and promote best model automatically

### 2. Start API Server

```bash
# With MLflow (uses Production model)
USE_MLFLOW=true uvicorn src.api.main:app --reload

# Or with Docker
docker-compose up -d
```

### 3. Make Predictions

```bash
# Test API
curl http://localhost:8000/health

# Make prediction
curl -X POST "http://localhost:8000/predict/simple" \
  -H "Content-Type: application/json" \
  -d @example_request.json
```

### 4. View MLflow UI

```bash
mlflow ui
# Open http://localhost:5000
```

## 📁 Project Structure

```
predictive-maintenance-mlops/
├── src/                    # Source code
│   ├── api/               # FastAPI application
│   │   ├── main.py        # API endpoints
│   │   └── schemas.py     # Pydantic schemas
│   ├── data/              # Data loading and preprocessing
│   ├── models/            # Model architectures
│   ├── train.py           # Basic training script
│   ├── train_with_versioning.py  # Training with versioning
│   ├── train_with_mlflow.py      # MLflow training (recommended)
│   ├── evaluate.py        # Evaluation script
│   ├── predict.py         # Prediction script
│   ├── model_registry.py  # Legacy model versioning
│   └── mlflow_utils.py    # MLflow integration utilities
├── tests/                 # Unit tests
├── docs/                  # Documentation
│   ├── API.md            # API documentation
│   ├── TRAINING.md       # Training guide
│   ├── MLFLOW_GUIDE.md   # MLflow integration
│   ├── DOCKER.md         # Docker deployment
│   └── ...
├── docker/                # Docker helper scripts
├── data/                  # Data directory
│   ├── raw/              # Raw data files
│   └── processed/        # Processed data
├── models/               # Trained models (legacy registry)
├── mlruns/               # MLflow tracking data
├── notebooks/            # Jupyter notebooks
├── Dockerfile            # Docker image definition
├── docker-compose.yml    # Docker Compose configuration
└── requirements.txt      # Python dependencies
```

## 🎯 Training Models

### Training Options

**Option 1: MLflow Training (Recommended for Production)**
```bash
python -m src.train_with_mlflow --epochs 20 --auto-promote
```

**Option 2: Training with Versioning**
```bash
python -m src.train_with_versioning --model lstm --epochs 20
```

**Option 3: Basic Training**
```bash
python -m src.train --model lstm --epochs 20
```

### Available Models

- `lstm` - Standard LSTM
- `bilstm` - Bidirectional LSTM
- `gru` - GRU (recommended for speed)
- `transformer` - Transformer encoder

See [Training Guide](docs/TRAINING.md) for detailed information.

## 🌐 API Inference Service

### Quick Start

```bash
# Start API server
uvicorn src.api.main:app --reload

# Or using Docker
docker-compose up -d
```

### API Endpoints

- **Health Check**: `GET /` or `GET /health`
- **List Models**: `GET /models`
- **Model Info**: `GET /models/{version}`
- **Predict RUL**: `POST /predict` or `POST /predict/simple`

### Interactive Documentation

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Example Request

```bash
curl -X POST "http://localhost:8000/predict/simple" \
  -H "Content-Type: application/json" \
  -d @example_request.json
```

See [API Documentation](docs/API.md) for complete reference.

## 🐳 Docker Deployment

### Quick Start

```bash
# Build and start
docker-compose up -d

# Or use helper scripts
./docker/start.sh  # Linux/Mac
docker\start.bat   # Windows
```

### Features

- ✅ MLflow integration (automatic Production model loading)
- ✅ Volume mounting for models and MLflow data
- ✅ Health checks included
- ✅ Production-ready configuration

### Commands

```bash
# Start
docker-compose up -d

# Stop
docker-compose down

# View logs
docker-compose logs -f

# Rebuild
docker-compose up -d --build
```

See [Docker Guide](docs/DOCKER.md) for detailed instructions.

## 📚 Documentation

Comprehensive documentation is available in the `docs/` directory:

- **[Usage Guide](docs/USAGE.md)** - Complete usage instructions
- **[Training Guide](docs/TRAINING.md)** - Training workflows and best practices
- **[Model Registry Guide](docs/MODEL_REGISTRY.md)** - Model versioning system
- **[API Documentation](docs/API.md)** - Complete API reference with examples
- **[MLflow Guide](docs/MLFLOW_GUIDE.md)** - MLflow integration and workflows
- **[Docker Guide](docs/DOCKER.md)** - Docker deployment and management
- **[Deployment Guide](docs/DEPLOYMENT.md)** - Production deployment strategies
- **[CI/CD Guide](docs/CI_CD.md)** - Continuous Integration setup
- **[Testing Guide](docs/TESTING.md)** - Testing framework and practices

## 🧪 Model Architectures

The project supports four deep learning architectures:

- **LSTM**: Standard Long Short-Term Memory network
- **BiLSTM**: Bidirectional LSTM for better context understanding
- **GRU**: Gated Recurrent Unit (faster training, similar performance)
- **Transformer**: Transformer encoder with attention mechanism

All models are configured with:
- Sequence length: 30 cycles
- Input features: 24 (3 operational settings + 21 sensors)
- Hidden size: 64
- Number of layers: 2
- Dropout: 0.2

## 🔄 MLflow Workflow

### Training with MLflow

```bash
# Train all models and auto-promote best
python -m src.train_with_mlflow --epochs 20 --auto-promote
```

### View Experiments

```bash
mlflow ui
# Open http://localhost:5000
```

### Model Management

- **View Models**: MLflow UI → Models
- **Promote to Production**: Add alias "production" to desired version
- **Compare Models**: Select multiple runs in MLflow UI

### API Integration

```bash
# API automatically uses Production model
USE_MLFLOW=true uvicorn src.api.main:app
```

See [MLflow Guide](docs/MLFLOW_GUIDE.md) for detailed information.

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_api.py -v
```

See [Testing Guide](docs/TESTING.md) for more information.

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- CMAPSS dataset from NASA
- PyTorch for deep learning framework
- MLflow for MLOps capabilities
- FastAPI for API framework

## 📞 Support

For issues, questions, or contributions:
- Open an issue on GitHub
- Check the [documentation](docs/)
- Review [FAQ](docs/USAGE.md#troubleshooting)

---

**Status**: ✅ Production-ready MLOps pipeline with MLflow integration
