#!/bin/bash
# Rebuild and restart Docker containers

set -e

echo "🔄 Rebuilding and restarting Predictive Maintenance API..."

docker-compose down
docker-compose build --no-cache
docker-compose up -d

echo "⏳ Waiting for API to be ready..."
sleep 5

if curl -f http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ API rebuilt and running!"
    echo "📍 API available at: http://localhost:8000"
else
    echo "⚠️  API might still be starting. Check logs with: docker-compose logs"
fi

