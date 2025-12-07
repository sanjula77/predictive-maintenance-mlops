@echo off
REM Quick start script for Docker deployment (Windows)

echo 🐳 Starting Predictive Maintenance API with Docker...

REM Check if Docker is installed
docker --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Docker is not installed. Please install Docker Desktop first.
    exit /b 1
)

REM Build and start
echo 📦 Building Docker image...
docker-compose build

echo 🚀 Starting container...
docker-compose up -d

echo ⏳ Waiting for API to be ready...
timeout /t 5 /nobreak >nul

REM Check health (requires curl or PowerShell)
echo ✅ Container started!
echo 📍 API available at: http://localhost:8000
echo 📊 API docs at: http://localhost:8000/docs
echo.
echo To view logs: docker-compose logs -f
echo To stop: docker-compose down

pause

