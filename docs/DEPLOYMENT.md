# Deployment Guide

This guide covers deploying the Predictive Maintenance API to various platforms.

## 🐳 Docker Deployment

### Build and Run Locally

```bash
# Build image
docker build -t predictive-maintenance-api .

# Run container
docker run -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/data:/app/data \
  predictive-maintenance-api
```

### Using Docker Compose

```bash
docker-compose up -d
```

## ☁️ Cloud Deployment

### Railway

1. **Connect Repository**
   - Go to [Railway](https://railway.app)
   - Click "New Project" → "Deploy from GitHub repo"
   - Select your repository

2. **Configure**
   - Railway auto-detects Dockerfile
   - Set port: `8000`
   - Add environment variables if needed

3. **Deploy**
   - Railway automatically deploys on push
   - Get your API URL from Railway dashboard

### Render

1. **Create Web Service**
   - Go to [Render](https://render.com)
   - Click "New" → "Web Service"
   - Connect your GitHub repository

2. **Configure**
   - **Build Command**: `docker build -t api .`
   - **Start Command**: `docker run -p $PORT:8000 api`
   - **Environment**: Python 3.10

3. **Deploy**
   - Render builds and deploys automatically
   - Your API will be available at `https://your-app.onrender.com`

### AWS EC2

1. **Launch EC2 Instance**
   ```bash
   # SSH into instance
   ssh -i your-key.pem ubuntu@your-ec2-ip
   ```

2. **Install Docker**
   ```bash
   sudo apt-get update
   sudo apt-get install docker.io docker-compose -y
   sudo usermod -aG docker ubuntu
   ```

3. **Clone and Deploy**
   ```bash
   git clone your-repo-url
   cd predictive-maintenance-mlops
   docker-compose up -d
   ```

4. **Configure Security Group**
   - Open port 8000 in EC2 Security Group
   - Access API at `http://your-ec2-ip:8000`

### Azure Container Instances

1. **Build and Push to Azure Container Registry**
   ```bash
   az acr build --registry your-registry --image predictive-maintenance-api .
   ```

2. **Deploy Container**
   ```bash
   az container create \
     --resource-group your-rg \
     --name predictive-maintenance-api \
     --image your-registry.azurecr.io/predictive-maintenance-api \
     --ports 8000 \
     --ip-address Public
   ```

## 🔧 Environment Variables

Create `.env` file or set in platform:

```env
# Optional: Override defaults
MODEL_VERSION=1
MODEL_TYPE=lstm
API_HOST=0.0.0.0
API_PORT=8000
LOG_LEVEL=INFO
```

## 📊 Health Checks

All platforms can use:
- **Health endpoint**: `GET /health`
- **Detailed status**: `GET /`

## 🚀 Production Checklist

- [ ] Set up proper CORS origins
- [ ] Configure logging
- [ ] Set up monitoring (e.g., Sentry)
- [ ] Use HTTPS (via reverse proxy or platform)
- [ ] Set resource limits
- [ ] Configure auto-scaling
- [ ] Set up backup for models
- [ ] Monitor API performance

## 📝 Example Deployment Scripts

### Railway/Render (No Dockerfile needed)

Create `Procfile`:
```
web: uvicorn src.api.main:app --host 0.0.0.0 --port $PORT
```

### Heroku

Create `Procfile`:
```
web: uvicorn src.api.main:app --host 0.0.0.0 --port $PORT
```

Add `runtime.txt`:
```
python-3.10.12
```

## 🔍 Testing Deployment

```bash
# Health check
curl https://your-api-url.com/health

# List models
curl https://your-api-url.com/models

# Make prediction
curl -X POST "https://your-api-url.com/predict?version=1&model_type=lstm" \
  -H "Content-Type: application/json" \
  -d @example_request.json
```

