# Deployment Optimization Guide

## Issue: Docker Build Timeout

If you're experiencing timeouts during Docker builds in CI/CD, here are optimization strategies.

## Quick Fixes Applied

1. **Removed `--no-cache` flag** - Uses Docker layer caching for faster rebuilds
2. **Added job timeout** - 30 minutes timeout for the deployment job
3. **Optimized Dockerfile** - Better pip caching and error handling

## Why Builds Are Slow

The main bottleneck is installing PyTorch and MLflow dependencies:
- PyTorch CPU build: ~200-300MB download
- MLflow with dependencies: ~100MB
- Total build time: 5-10 minutes on first build
- Subsequent builds: 2-3 minutes (with cache)

## Optimization Strategies

### Option 1: Use Docker Layer Caching (Current)

**Benefits:**
- Faster subsequent builds
- Only rebuilds changed layers
- Works automatically

**How it works:**
- First build: Downloads all packages (slow)
- Next builds: Reuses cached layers (fast)

### Option 2: Pre-build on Server

**Manual approach:**
```bash
# SSH into server
ssh ubuntu@80.225.215.211

# Build image once
cd ~/predictive-maintenance-mlops
docker-compose build

# Future deployments will be faster
```

### Option 3: Use Multi-stage Build

**Optimize Dockerfile:**
```dockerfile
# Stage 1: Dependencies
FROM python:3.10-slim as deps
WORKDIR /app
COPY requirements.txt .
RUN pip install --user -r requirements.txt

# Stage 2: Runtime
FROM python:3.10-slim
WORKDIR /app
COPY --from=deps /root/.local /root/.local
COPY . .
ENV PATH=/root/.local/bin:$PATH
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Option 4: Use Docker Registry

**Build once, deploy many times:**
1. Build image locally or in CI
2. Push to Docker Hub/OCI Registry
3. Server pulls pre-built image

**Workflow:**
```yaml
# Build and push
docker build -t your-registry/predictive-api:latest .
docker push your-registry/predictive-api:latest

# On server, just pull
docker pull your-registry/predictive-api:latest
docker-compose up -d
```

### Option 5: Reduce Dependencies

**Split requirements:**
- `requirements-base.txt` - Core dependencies
- `requirements-dev.txt` - Development only
- `requirements-ml.txt` - ML libraries (large)

**Install only what's needed:**
```dockerfile
COPY requirements-base.txt .
RUN pip install -r requirements-base.txt
```

## Current Setup

The current setup uses:
- ✅ Docker layer caching
- ✅ 30-minute job timeout
- ✅ Optimized pip installation
- ✅ Error handling

## Monitoring Build Time

**Check build logs:**
```bash
# On server
docker-compose logs api

# Check build time
time docker-compose build
```

**Expected times:**
- First build: 5-10 minutes
- Cached build: 2-3 minutes
- Code-only change: 30 seconds

## Troubleshooting

### Build Still Timing Out

1. **Increase timeout:**
   ```yaml
   timeout-minutes: 45  # In deploy.yml
   ```

2. **Build on server first:**
   ```bash
   ssh ubuntu@80.225.215.211
   cd ~/predictive-maintenance-mlops
   docker-compose build
   ```

3. **Use pre-built image:**
   - Build locally
   - Push to registry
   - Pull on server

### Build Fails with Memory Error

**Solution:**
```bash
# Increase Docker memory limit
# Or use smaller base image
FROM python:3.10-slim
```

### Slow Network on Server

**Solution:**
- Use local package mirrors
- Pre-download packages
- Use Docker registry cache

## Best Practices

1. **Use layer caching** - Don't use `--no-cache` unless necessary
2. **Order matters** - Copy requirements before code
3. **Minimize layers** - Combine RUN commands
4. **Use .dockerignore** - Exclude unnecessary files
5. **Monitor build times** - Track improvements

## Next Steps

If builds are still too slow:
1. Consider using Docker Registry approach
2. Pre-build images on schedule
3. Use smaller base images
4. Split dependencies

