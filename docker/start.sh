#!/bin/bash
# Quick start script for Docker deployment

set -e

echo "🐳 Starting Predictive Maintenance API with Docker..."

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if docker-compose is installed
if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Build and start
echo "📦 Building Docker image..."
docker-compose build

echo "🚀 Starting container..."
docker-compose up -d

echo "⏳ Waiting for API to be ready..."
sleep 5

# Check health
if curl -f http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ API is running and healthy!"
    echo "📍 API available at: http://localhost:8000"
    echo "📊 API docs at: http://localhost:8000/docs"
    echo ""
    echo "To view logs: docker-compose logs -f"
    echo "To stop: docker-compose down"
else
    echo "⚠️  API might still be starting. Check logs with: docker-compose logs"
fi

