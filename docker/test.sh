#!/bin/bash
# Test Docker deployment

set -e

echo "🧪 Testing Docker deployment..."

# Test if container is running
if ! docker ps | grep -q predictive-maintenance-api; then
    echo "❌ Container is not running. Start it first with: docker-compose up -d"
    exit 1
fi

echo "✅ Container is running"

# Test health endpoint
echo "Testing health endpoint..."
if curl -f http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ Health check passed"
else
    echo "❌ Health check failed"
    exit 1
fi

# Test models endpoint
echo "Testing models endpoint..."
if curl -f http://localhost:8000/models > /dev/null 2>&1; then
    echo "✅ Models endpoint working"
else
    echo "❌ Models endpoint failed"
    exit 1
fi

# Test API docs
echo "Testing API documentation..."
if curl -f http://localhost:8000/docs > /dev/null 2>&1; then
    echo "✅ API docs accessible"
else
    echo "❌ API docs not accessible"
    exit 1
fi

echo ""
echo "🎉 All tests passed! Docker deployment is working correctly."
echo "📍 API: http://localhost:8000"
echo "📊 Docs: http://localhost:8000/docs"

