@echo off
REM medAI MVP Docker Build and Run Script for Windows

set IMAGE_NAME=medai-mvp
set CONTAINER_NAME=medai-mvp-container
set PORT=8000

echo 🏗️  Building medAI MVP Docker image...

REM Build the Docker image
docker build -t %IMAGE_NAME% .

if %ERRORLEVEL% neq 0 (
    echo ❌ Docker build failed!
    exit /b 1
)

echo ✅ Docker image built successfully!

echo 🚀 Starting medAI MVP container...

REM Stop and remove existing container if it exists
docker stop %CONTAINER_NAME% 2>nul
docker rm %CONTAINER_NAME% 2>nul

REM Run the container
docker run -d ^
    --name %CONTAINER_NAME% ^
    -p %PORT%:8000 ^
    --restart unless-stopped ^
    %IMAGE_NAME%

if %ERRORLEVEL% neq 0 (
    echo ❌ Docker run failed!
    exit /b 1
)

echo ✅ Container started successfully!
echo 🌐 Application is available at: http://localhost:%PORT%
echo 📊 Container logs: docker logs %CONTAINER_NAME%
echo 🛑 Stop container: docker stop %CONTAINER_NAME%
echo 🗑️  Remove container: docker rm %CONTAINER_NAME%
