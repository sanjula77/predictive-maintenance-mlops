@echo off
REM Stop Docker containers (Windows)

echo 🛑 Stopping Predictive Maintenance API...

docker-compose down

echo ✅ Containers stopped

pause

