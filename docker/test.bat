@echo off
REM Test Docker deployment (Windows)

echo 🧪 Testing Docker deployment...

REM Test if container is running
docker ps | findstr predictive-maintenance-api >nul
if errorlevel 1 (
    echo ❌ Container is not running. Start it first with: docker-compose up -d
    exit /b 1
)

echo ✅ Container is running

REM Test health endpoint
echo Testing health endpoint...
curl -f http://localhost:8000/health >nul 2>&1
if errorlevel 1 (
    echo ❌ Health check failed
    exit /b 1
)
echo ✅ Health check passed

REM Test models endpoint
echo Testing models endpoint...
curl -f http://localhost:8000/models >nul 2>&1
if errorlevel 1 (
    echo ❌ Models endpoint failed
    exit /b 1
)
echo ✅ Models endpoint working

echo.
echo 🎉 All tests passed! Docker deployment is working correctly.
echo 📍 API: http://localhost:8000
echo 📊 Docs: http://localhost:8000/docs

pause

