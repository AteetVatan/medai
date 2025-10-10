#!/bin/bash

# medAI MVP Docker Build and Run Script

set -e

# Configuration
IMAGE_NAME="medai-mvp"
CONTAINER_NAME="medai-mvp-container"
PORT="8000"

echo "🏗️  Building medAI MVP Docker image..."

# Build the Docker image
docker build -t $IMAGE_NAME .

echo "✅ Docker image built successfully!"

echo "🚀 Starting medAI MVP container..."

# Stop and remove existing container if it exists
docker stop $CONTAINER_NAME 2>/dev/null || true
docker rm $CONTAINER_NAME 2>/dev/null || true

# Run the container
docker run -d \
    --name $CONTAINER_NAME \
    -p $PORT:8000 \
    --restart unless-stopped \
    $IMAGE_NAME

echo "✅ Container started successfully!"
echo "🌐 Application is available at: http://localhost:$PORT"
echo "📊 Container logs: docker logs $CONTAINER_NAME"
echo "🛑 Stop container: docker stop $CONTAINER_NAME"
echo "🗑️  Remove container: docker rm $CONTAINER_NAME"
