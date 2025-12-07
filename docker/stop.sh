#!/bin/bash
# Stop Docker containers

set -e

echo "🛑 Stopping Predictive Maintenance API..."

docker-compose down

echo "✅ Containers stopped"

