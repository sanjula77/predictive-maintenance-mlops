# Docker Deployment Guide

This guide explains how to containerize and deploy the Predictive Maintenance API using Docker.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Docker Commands](#docker-commands)
- [Docker Compose](#docker-compose)
- [MLflow Integration](#mlflow-integration)
- [Production Deployment](#production-deployment)
- [Troubleshooting](#troubleshooting)

## Prerequisites

- Docker installed ([Install Docker](https://docs.docker.com/get-docker/))
- Docker Compose (usually included with Docker Desktop)
- MLflow models trained and registered (if using MLflow mode)

## Quick Start

### 1. Build the Docker Image

```bash
docker build -t predictive-maintenance-api:latest .
```

### 2. Run the Container

```bash
docker run -d \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/mlruns:/app/mlruns \
  -e USE_MLFLOW=true \
  --name predictive-maintenance-api \
  predictive-maintenance-api:latest
```

### 3. Verify It's Running

```bash
# Check container status
docker ps

# Check logs
docker logs predictive-maintenance-api

# Test API
curl http://localhost:8000/health
```

## Docker Commands

### Build Image

```bash
# Build with default tag
docker build -t predictive-maintenance-api:latest .

# Build with specific tag/version
docker build -t predictive-maintenance-api:v1.0.0 .
```

### Run Container

```bash
# Run in foreground (see logs)
docker run -p 8000:8000 predictive-maintenance-api:latest

# Run in background (detached)
docker run -d -p 8000:8000 --name api predictive-maintenance-api:latest

# Run with environment variables
docker run -d \
  -p 8000:8000 \
  -e USE_MLFLOW=true \
  -e MLFLOW_TRACKING_URI=file:///app/mlruns \
  predictive-maintenance-api:latest
```

### Container Management

```bash
# List running containers
docker ps

# List all containers (including stopped)
docker ps -a

# Stop container
docker stop predictive-maintenance-api

# Start container
docker start predictive-maintenance-api

# Restart container
docker restart predictive-maintenance-api

# Remove container
docker rm predictive-maintenance-api

# View logs
docker logs predictive-maintenance-api

# Follow logs (real-time)
docker logs -f predictive-maintenance-api

# Execute command in running container
docker exec -it predictive-maintenance-api bash
```

### Image Management

```bash
# List images
docker images

# Remove image
docker rmi predictive-maintenance-api:latest

# Tag image for registry
docker tag predictive-maintenance-api:latest your-registry/predictive-maintenance-api:v1.0.0

# Push to registry
docker push your-registry/predictive-maintenance-api:v1.0.0
```

## Docker Compose

Docker Compose makes it easier to manage the application with volumes and environment variables.

### Start Services

```bash
# Start in background
docker-compose up -d

# Start and see logs
docker-compose up

# Rebuild and start
docker-compose up --build
```

### Stop Services

```bash
# Stop services
docker-compose stop

# Stop and remove containers
docker-compose down

# Stop, remove containers, and volumes
docker-compose down -v
```

### View Logs

```bash
# View logs
docker-compose logs

# Follow logs
docker-compose logs -f

# View logs for specific service
docker-compose logs api
```

### Update Services

```bash
# Rebuild and restart
docker-compose up -d --build

# Restart without rebuild
docker-compose restart
```

## MLflow Integration

### Local MLflow (Default)

The Docker setup mounts the `mlruns` directory to persist MLflow data:

```yaml
volumes:
  - ./mlruns:/app/mlruns
```

This allows:
- Models to persist between container restarts
- Access to MLflow UI from host machine
- Model registry to work correctly

### Remote MLflow Server

If using a remote MLflow server, set the tracking URI:

```bash
docker run -d \
  -p 8000:8000 \
  -e USE_MLFLOW=true \
  -e MLFLOW_TRACKING_URI=http://mlflow-server:5000 \
  predictive-maintenance-api:latest
```

Or in `docker-compose.yml`:

```yaml
environment:
  - USE_MLFLOW=true
  - MLFLOW_TRACKING_URI=http://mlflow-server:5000
```

### MLflow UI Access

Run MLflow UI separately (on host or in another container):

```bash
# On host machine
mlflow ui --backend-store-uri file:///path/to/mlruns

# Or in Docker
docker run -d \
  -p 5000:5000 \
  -v $(pwd)/mlruns:/mlruns \
  python:3.10-slim \
  sh -c "pip install mlflow && mlflow ui --backend-store-uri file:///mlruns --host 0.0.0.0"
```

## Production Deployment

### Multi-Stage Build (Optional)

For smaller images, use multi-stage builds:

```dockerfile
# Build stage
FROM python:3.10-slim as builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.10-slim
WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY . .
ENV PATH=/root/.local/bin:$PATH
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Environment Variables

Set production environment variables:

```bash
docker run -d \
  -p 8000:8000 \
  -e USE_MLFLOW=true \
  -e PYTHONUNBUFFERED=1 \
  -e LOG_LEVEL=INFO \
  predictive-maintenance-api:latest
```

### Resource Limits

Set CPU and memory limits:

```yaml
# docker-compose.yml
services:
  api:
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
        reservations:
          cpus: '1'
          memory: 2G
```

### Health Checks

The Dockerfile includes a health check. Monitor it:

```bash
# Check health status
docker inspect --format='{{.State.Health.Status}}' predictive-maintenance-api
```

## Troubleshooting

### Container Won't Start

```bash
# Check logs
docker logs predictive-maintenance-api

# Check if port is already in use
lsof -i :8000

# Run interactively to debug
docker run -it --rm predictive-maintenance-api:latest bash
```

### MLflow Models Not Found

1. Ensure `mlruns` directory is mounted:
   ```bash
   docker run -v $(pwd)/mlruns:/app/mlruns ...
   ```

2. Check MLflow tracking URI:
   ```bash
   docker exec predictive-maintenance-api env | grep MLFLOW
   ```

3. Verify models exist:
   ```bash
   docker exec predictive-maintenance-api ls -la /app/mlruns/models/
   ```

### Permission Issues

If you encounter permission issues with volumes:

```bash
# Fix permissions (Linux/Mac)
sudo chown -R $USER:$USER mlruns models

# Or run container with specific user
docker run -u $(id -u):$(id -g) ...
```

### Out of Memory

If container runs out of memory:

```bash
# Increase memory limit
docker run --memory="4g" ...
```

Or in `docker-compose.yml`:
```yaml
services:
  api:
    deploy:
      resources:
        limits:
          memory: 4G
```

### Network Issues

```bash
# Check network connectivity
docker exec predictive-maintenance-api ping google.com

# Inspect network
docker network inspect bridge
```

## Best Practices

1. **Use specific tags**: Don't use `latest` in production
   ```bash
   docker build -t predictive-maintenance-api:v1.0.0 .
   ```

2. **Use .dockerignore**: Already configured to exclude unnecessary files

3. **Mount volumes for persistence**: Models and MLflow data should be mounted

4. **Set resource limits**: Prevent containers from consuming all resources

5. **Use health checks**: Monitor container health

6. **Log management**: Consider log rotation for production

7. **Security**: Don't run as root user in production (add user to Dockerfile)

## Example: Complete Production Setup

```bash
# 1. Build image
docker build -t predictive-maintenance-api:v1.0.0 .

# 2. Tag for registry
docker tag predictive-maintenance-api:v1.0.0 \
  your-registry/predictive-maintenance-api:v1.0.0

# 3. Push to registry
docker push your-registry/predictive-maintenance-api:v1.0.0

# 4. Deploy on server
docker pull your-registry/predictive-maintenance-api:v1.0.0
docker run -d \
  --name api \
  -p 8000:8000 \
  -v /data/models:/app/models \
  -v /data/mlruns:/app/mlruns \
  -e USE_MLFLOW=true \
  --restart unless-stopped \
  your-registry/predictive-maintenance-api:v1.0.0
```

## Additional Resources

- [Docker Documentation](https://docs.docker.com/)
- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [MLflow Docker Guide](https://www.mlflow.org/docs/latest/tracking.html#docker)

