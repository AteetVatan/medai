# medAI MVP Docker Deployment

This document explains how to build and run the medAI MVP application using Docker in production mode.

## Quick Start

### Option 1: Using the build script (Recommended)

**Windows:**
```bash
docker-build.bat
```

**Linux/Mac:**
```bash
./docker-build.sh
```

### Option 2: Manual Docker commands

1. **Build the Docker image:**
```bash
docker build -t medai-mvp .
```

2. **Run the container:**
```bash
docker run -d \
    --name medai-mvp-container \
    -p 8000:8000 \
    --restart unless-stopped \
    medai-mvp
```

## Configuration

The Docker container runs the application in production mode with the following settings:
- **Host:** 0.0.0.0 (accessible from all interfaces)
- **Port:** 8000
- **Workers:** 4 (for better performance)
- **Log Level:** info
- **Reload:** Disabled (production mode)

## Environment Variables

Make sure to create a `.env` file in your project root with the required configuration:

```bash
# Copy the example environment file
cp env.example .env

# Edit the .env file with your API keys and settings
```

## Health Check

The container includes a health check that verifies the application is responding:
- **Interval:** 30 seconds
- **Timeout:** 30 seconds
- **Retries:** 3

## Container Management

### View logs:
```bash
docker logs medai-mvp-container
```

### Stop the container:
```bash
docker stop medai-mvp-container
```

### Remove the container:
```bash
docker rm medai-mvp-container
```

### Access the application:
Open your browser and navigate to: http://localhost:8000

## Production Considerations

1. **Environment Variables:** Ensure all required environment variables are set in your `.env` file
2. **API Keys:** Configure all necessary API keys for external services
3. **Database:** If using a database, ensure it's accessible from the container
4. **Networking:** Configure proper networking for external service access
5. **Monitoring:** Consider adding monitoring and logging solutions
6. **SSL/TLS:** Use a reverse proxy (nginx, traefik) for SSL termination in production

## Troubleshooting

### Container won't start:
- Check if port 8000 is already in use
- Verify the `.env` file exists and is properly configured
- Check container logs: `docker logs medai-mvp-container`

### Application not accessible:
- Verify the container is running: `docker ps`
- Check if the port mapping is correct: `docker port medai-mvp-container`
- Ensure firewall allows traffic on port 8000

### Performance issues:
- Increase the number of workers in the Dockerfile CMD
- Monitor resource usage: `docker stats medai-mvp-container`
- Consider using a multi-stage build for smaller image size
