# medAI MVP Production Dockerfile
FROM python:3.12-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app

# Set work directory
WORKDIR /app

# Install system dependencies for WeasyPrint and other requirements
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libcairo2-dev \
    libpango1.0-dev \
    libgdk-pixbuf-xlib-2.0-dev \
    libffi-dev \
    shared-mime-info \
    libxml2-dev \
    libxslt1-dev \
    libjpeg-dev \
    libpng-dev \
    libfreetype6-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Verify critical files exist
RUN ls -la /app/start_dev.py && \
    ls -la /app/.env || echo "Warning: .env file not found"

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash app && \
    chown -R app:app /app
USER app

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Start the application in production mode
CMD ["python", "start_dev.py", "--production", "--host", "0.0.0.0", "--port", "8000", "--workers", "4", "--log-level", "info"]
