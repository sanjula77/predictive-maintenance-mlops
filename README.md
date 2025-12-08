<div align="center">

# 🔧 Predictive Maintenance MLOps

**Production-ready MLOps pipeline for Remaining Useful Life (RUL) prediction**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3.0-orange.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![MLflow](https://img.shields.io/badge/MLflow-2.8+-0194E2.svg)](https://mlflow.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

[Features](#-features) • [Quick Start](#-quick-start) • [Documentation](#-documentation) • [API Reference](#-api-reference) • [Contributing](#-contributing)

</div>

---

## 📖 Overview

A comprehensive MLOps pipeline for predictive maintenance using deep learning models to predict the Remaining Useful Life (RUL) of aircraft engines. Built with production-grade tools including MLflow for experiment tracking, FastAPI for serving predictions, and Docker for containerized deployment.

### Key Highlights

- 🚀 **Production-Ready**: Fully containerized with Docker and CI/CD integration
- 📊 **MLflow Integration**: Complete experiment tracking and model registry
- 🤖 **Multiple Architectures**: LSTM, BiLSTM, GRU, and Transformer models
- 🔄 **Auto-Promotion**: Automatically promotes best-performing models to production
- 📈 **Comprehensive Evaluation**: Built-in model comparison and metrics tracking
- 🐳 **Docker Support**: One-command deployment with Docker Compose
- 🔒 **CI/CD Pipeline**: Automated testing and deployment via GitHub Actions

---

## ✨ Features

### Model Training & Management
- **Multiple Deep Learning Architectures**: LSTM, BiLSTM, GRU, and Transformer
- **MLflow Integration**: Experiment tracking, model registry, and versioning
- **Automatic Model Promotion**: Best model automatically promoted to production
- **Model Comparison**: Built-in tools for comparing model performance
- **Hyperparameter Tracking**: All training parameters logged automatically

### API & Deployment
- **RESTful API**: FastAPI-based inference service with automatic documentation
- **Docker Support**: Fully containerized for easy deployment
- **Health Checks**: Built-in monitoring and health check endpoints
- **Interactive Docs**: Swagger UI and ReDoc for API exploration
- **Production Ready**: Optimized for production workloads

### DevOps & Automation
- **CI/CD Pipeline**: Automated testing and deployment
- **GitHub Actions**: Continuous integration and deployment workflows
- **Automated Testing**: Comprehensive test suite with coverage reporting
- **Version Control**: Model versioning and metadata tracking

---

## 🚀 Quick Start

### Prerequisites

- **Python** 3.10 or higher
- **pip** package manager
- **Docker** (optional, for containerized deployment)
- **CUDA-capable GPU** (optional, for faster training)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/predictive-maintenance-mlops.git
   cd predictive-maintenance-mlops
   ```

2. **Create and activate virtual environment**
   ```bash
   # Create virtual environment
   python -m venv venv
   
   # Activate virtual environment
   # On Windows:
   venv\Scripts\activate
   # On Linux/Mac:
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Quick Start Guide

#### 1. Train Models with MLflow (Recommended)

Train all model architectures and automatically promote the best model to production:

```bash
python -m src.train_with_mlflow --epochs 20 --auto-promote
```

This command will:
- ✅ Train all 4 model architectures (LSTM, BiLSTM, GRU, Transformer)
- ✅ Track experiments in MLflow
- ✅ Register all models in the model registry
- ✅ Compare models and automatically promote the best one to production

#### 2. Start the API Server

**Option A: Direct Python execution**
```bash
# With MLflow (uses Production model automatically)
USE_MLFLOW=true uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

**Option B: Docker Compose (Recommended)**
```bash
docker-compose up -d
```

#### 3. Make Predictions

Test the API health endpoint:
```bash
curl http://localhost:8000/health
```

Make a prediction:
```bash
curl -X POST "http://localhost:8000/predict/simple" \
  -H "Content-Type: application/json" \
  -d @example_request.json
```

#### 4. View MLflow UI

Access the MLflow tracking UI to view experiments and models:
```bash
mlflow ui
# Open http://localhost:5000 in your browser
```

---

## 📁 Project Structure

```
predictive-maintenance-mlops/
├── src/                          # Source code
│   ├── api/                      # FastAPI application
│   │   ├── main.py              # API endpoints and routes
│   │   └── schemas.py           # Pydantic request/response schemas
│   ├── data/                     # Data loading and preprocessing
│   │   ├── load_data.py         # Data loading utilities
│   │   └── preprocessing.py     # Data preprocessing pipeline
│   ├── models/                   # Model architectures
│   │   ├── architectures.py     # LSTM, BiLSTM, GRU, Transformer
│   │   └── __init__.py
│   ├── train.py                  # Basic training script
│   ├── train_with_versioning.py # Training with versioning
│   ├── train_with_mlflow.py     # MLflow training (recommended)
│   ├── evaluate.py              # Model evaluation script
│   ├── predict.py               # Standalone prediction script
│   ├── model_registry.py        # Legacy model versioning
│   ├── mlflow_utils.py          # MLflow integration utilities
│   └── config.py                # Configuration management
├── tests/                        # Test suite
│   ├── test_api.py              # API endpoint tests
│   ├── test_models.py           # Model architecture tests
│   ├── test_data_preprocessing.py
│   └── conftest.py              # Pytest configuration
├── docs/                         # Comprehensive documentation
│   ├── API.md                   # Complete API reference
│   ├── TRAINING.md              # Training workflows guide
│   ├── MLFLOW_GUIDE.md          # MLflow integration guide
│   ├── DOCKER.md                # Docker deployment guide
│   ├── CI_CD_DEPLOYMENT.md      # CI/CD setup guide
│   └── ...                      # Additional documentation
├── docker/                       # Docker helper scripts
│   ├── start.sh / start.bat     # Start containers
│   ├── stop.sh / stop.bat       # Stop containers
│   └── test.sh / test.bat       # Run tests in containers
├── data/                         # Data directory
│   ├── raw/                     # Raw CMAPSS dataset files
│   └── processed/               # Processed data files
├── models/                       # Trained models (legacy registry)
│   └── v*/                      # Versioned model directories
├── notebooks/                    # Jupyter notebooks
│   └── RUL_Exploration_and_Modeling.ipynb
├── Dockerfile                    # Docker image definition
├── docker-compose.yml            # Docker Compose configuration
├── requirements.txt              # Python dependencies
├── requirements-dev.txt          # Development dependencies
├── pytest.ini                   # Pytest configuration
└── README.md                     # This file
```

---

## 🎯 Training Models

### Training Options

The project provides three training approaches:

#### Option 1: MLflow Training (Recommended)

Best for production use with experiment tracking and model registry:

```bash
python -m src.train_with_mlflow --epochs 20 --auto-promote
```

**Features:**
- Automatic experiment tracking
- Model registry integration
- Automatic best model promotion
- Comprehensive metrics logging

#### Option 2: Training with Versioning

Training with manual version control:

```bash
python -m src.train_with_versioning --model lstm --epochs 20
```

#### Option 3: Basic Training

Simple training without versioning:

```bash
python -m src.train --model lstm --epochs 20
```

### Available Model Architectures

| Model | Description | Use Case |
|-------|-------------|----------|
| `lstm` | Standard Long Short-Term Memory network | General purpose sequence modeling |
| `bilstm` | Bidirectional LSTM | Better context understanding |
| `gru` | Gated Recurrent Unit | Faster training, similar performance |
| `transformer` | Transformer encoder with attention | State-of-the-art performance |

### Model Configuration

All models are configured with:
- **Sequence Length**: 30 cycles
- **Input Features**: 24 (3 operational settings + 21 sensors)
- **Hidden Size**: 64
- **Number of Layers**: 2
- **Dropout**: 0.2

📚 **Detailed Guide**: See [Training Guide](docs/TRAINING.md) for comprehensive training documentation.

---

## 🌐 API Reference

### Quick Start

Start the API server:

```bash
# Direct execution
uvicorn src.api.main:app --reload

# Or with Docker
docker-compose up -d
```

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Root endpoint |
| `GET` | `/health` | Health check endpoint |
| `GET` | `/models` | List all available models |
| `GET` | `/models/{version}` | Get model information |
| `POST` | `/predict` | Make RUL prediction (full format) |
| `POST` | `/predict/simple` | Make RUL prediction (simplified format) |

### Interactive Documentation

Once the API is running, access interactive documentation:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Example Request

```bash
curl -X POST "http://localhost:8000/predict/simple" \
  -H "Content-Type: application/json" \
  -d @example_request.json
```

**Example Request Body:**
```json
{
  "sensor_readings": [
    {
      "op_setting_1": 0.0,
      "op_setting_2": 0.0,
      "op_setting_3": 100.0,
      "sensor_1": 518.67,
      "sensor_2": 641.82,
      ...
    }
  ]
}
```

📚 **Complete API Documentation**: See [API Documentation](docs/API.md) for detailed endpoint reference and examples.

---

## 🐳 Docker Deployment

### Quick Start

Deploy using Docker Compose:

```bash
# Start all services
docker-compose up -d

# Or use helper scripts
./docker/start.sh      # Linux/Mac
docker\start.bat       # Windows
```

### Docker Features

- ✅ MLflow integration (automatic Production model loading)
- ✅ Volume mounting for models and MLflow data
- ✅ Health checks included
- ✅ Production-ready configuration
- ✅ Hot-reload support for development

### Common Commands

```bash
# Start services
docker-compose up -d

# Stop services
docker-compose down

# View logs
docker-compose logs -f

# Rebuild containers
docker-compose up -d --build

# Run tests in container
docker-compose exec api pytest

# Access container shell
docker-compose exec api bash
```

📚 **Detailed Guide**: See [Docker Guide](docs/DOCKER.md) for comprehensive Docker documentation.

---

## 🔄 MLflow Workflow

### Training with MLflow

```bash
# Train all models and auto-promote best
python -m src.train_with_mlflow --epochs 20 --auto-promote
```

### View Experiments

```bash
# Start MLflow UI
mlflow ui

# Access at http://localhost:5000
```

### Model Management

- **View Models**: MLflow UI → Models tab
- **Promote to Production**: Add "production" alias to desired version
- **Compare Models**: Select multiple runs in MLflow UI
- **Download Models**: Export models for deployment

### API Integration

The API automatically uses the Production model when MLflow is enabled:

```bash
USE_MLFLOW=true uvicorn src.api.main:app
```

📚 **Complete Guide**: See [MLflow Guide](docs/MLFLOW_GUIDE.md) for detailed MLflow workflows.

---

## 🚀 CI/CD & Automated Deployment

### Automated Deployment Pipeline

The project includes a complete CI/CD pipeline using GitHub Actions:

**Workflow:**
1. 🔄 Push code to `main` branch
2. ✅ GitHub Actions runs automated tests
3. 🐳 Build Docker images
4. 🚀 Deploy to production server via SSH
5. 🔄 Rebuild containers with new code
6. ✅ Health check verifies deployment

### Quick Setup

#### 1. Generate SSH Key

```powershell
# Windows PowerShell
ssh-keygen -t ed25519 -C "github-actions-deploy" -f $env:USERPROFILE\.ssh\github_actions_deploy
```

#### 2. Add Public Key to Server

```powershell
# Display public key
type $env:USERPROFILE\.ssh\github_actions_deploy.pub

# Copy output, then SSH to server and add:
ssh ubuntu@your-server-ip
echo "PASTE_PUBLIC_KEY_HERE" >> ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys
```

#### 3. Configure GitHub Secrets

Go to: **Repository → Settings → Secrets → Actions**

Add the following secrets:
- `SSH_PRIVATE_KEY`: Your private SSH key (entire key including BEGIN/END lines)
- `SERVER_HOST`: Your server IP address
- `SERVER_USER`: Server username (typically `ubuntu`)

#### 4. Test Deployment

```bash
git push origin main
# Watch deployment in GitHub Actions tab
```

📚 **Setup Guides**:
- [Quick Setup Guide](DEPLOYMENT_SETUP.md)
- [Complete CI/CD Guide](docs/CI_CD_DEPLOYMENT.md)
- [SSH Key Setup](docs/SSH_KEY_SETUP.md)

---

## 🧪 Testing

### Run Tests

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_api.py -v

# Run with verbose output
pytest -v

# Run in Docker
docker-compose exec api pytest
```

### Test Coverage

View coverage report:
```bash
# Generate HTML coverage report
pytest --cov=src --cov-report=html

# Open htmlcov/index.html in browser
```

📚 **Testing Guide**: See [Testing Guide](docs/TESTING.md) for comprehensive testing documentation.

---

## 📚 Documentation

Comprehensive documentation is available in the `docs/` directory:

| Document | Description |
|----------|-------------|
| [Usage Guide](docs/USAGE.md) | Complete usage instructions and workflows |
| [Training Guide](docs/TRAINING.md) | Training workflows and best practices |
| [Model Registry Guide](docs/MODEL_REGISTRY.md) | Model versioning system documentation |
| [API Documentation](docs/API.md) | Complete API reference with examples |
| [MLflow Guide](docs/MLFLOW_GUIDE.md) | MLflow integration and workflows |
| [Docker Guide](docs/DOCKER.md) | Docker deployment and management |
| [Deployment Guide](docs/DEPLOYMENT.md) | Production deployment strategies |
| [CI/CD Guide](docs/CI_CD.md) | Continuous Integration setup |
| [CI/CD Deployment](docs/CI_CD_DEPLOYMENT.md) | Automated deployment setup |
| [Testing Guide](docs/TESTING.md) | Testing framework and practices |
| [SSH Key Setup](docs/SSH_KEY_SETUP.md) | SSH key configuration for CI/CD |

---

## 🏗️ Model Architectures

The project supports four state-of-the-art deep learning architectures:

### LSTM (Long Short-Term Memory)
- **Type**: Standard LSTM network
- **Use Case**: General purpose sequence modeling
- **Advantages**: Proven architecture, good for sequential data

### BiLSTM (Bidirectional LSTM)
- **Type**: Bidirectional LSTM
- **Use Case**: Better context understanding
- **Advantages**: Processes sequences in both directions

### GRU (Gated Recurrent Unit)
- **Type**: Gated Recurrent Unit
- **Use Case**: Faster training with similar performance
- **Advantages**: Fewer parameters, faster convergence

### Transformer
- **Type**: Transformer encoder with attention mechanism
- **Use Case**: State-of-the-art performance
- **Advantages**: Attention mechanism, parallel processing

### Architecture Configuration

All models share the following configuration:
- **Sequence Length**: 30 cycles
- **Input Features**: 24 (3 operational settings + 21 sensors)
- **Hidden Size**: 64
- **Number of Layers**: 2
- **Dropout Rate**: 0.2

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/amazing-feature
   ```
3. **Make your changes**
4. **Run tests**
   ```bash
   pytest
   ```
5. **Commit your changes**
   ```bash
   git commit -m 'Add amazing feature'
   ```
6. **Push to the branch**
   ```bash
   git push origin feature/amazing-feature
   ```
7. **Open a Pull Request**

📋 **Guidelines**: See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed contribution guidelines.

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **NASA CMAPSS Dataset**: For providing the aircraft engine degradation simulation data
- **PyTorch**: Deep learning framework
- **MLflow**: MLOps capabilities and experiment tracking
- **FastAPI**: Modern, fast web framework for building APIs
- **Docker**: Containerization platform

---

## 📞 Support & Contact

### Getting Help

- 📖 **Documentation**: Check the [docs/](docs/) directory
- 🐛 **Issues**: Open an issue on [GitHub Issues](https://github.com/yourusername/predictive-maintenance-mlops/issues)
- 💬 **Questions**: Review the [FAQ](docs/USAGE.md#troubleshooting) section
- 📧 **Contact**: Reach out via GitHub Discussions

### Resources

- [API Documentation](docs/API.md)
- [Training Guide](docs/TRAINING.md)
- [Deployment Guide](docs/DEPLOYMENT.md)
- [Troubleshooting](docs/USAGE.md#troubleshooting)

---

<div align="center">

**⭐ Star this repo if you find it helpful!**

Made with ❤️ for the MLOps community

[⬆ Back to Top](#-predictive-maintenance-mlops)

</div>
