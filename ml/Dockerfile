# Multi-stage Dockerfile for ML Pipeline
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p models logs artifacts data

# Development stage
FROM base as development

# Install development dependencies
RUN pip install --no-cache-dir \
    jupyter \
    ipython \
    pytest-xdist

# Expose ports
EXPOSE 8000 5000

# Default command for development
CMD ["python", "-m", "src.cli.main", "serve", "--host", "0.0.0.0", "--port", "8000"]

# Production stage
FROM base as production

# Create non-root user
RUN groupadd -r mluser && useradd -r -g mluser mluser

# Change ownership of app directory
RUN chown -R mluser:mluser /app

# Switch to non-root user
USER mluser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command for production
CMD ["python", "-m", "src.cli.main", "serve", "--host", "0.0.0.0", "--port", "8000"]

