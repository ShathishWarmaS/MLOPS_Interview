# Multi-stage Docker build for MLOps Platform
FROM python:3.9-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r mlops && useradd -r -g mlops mlops

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt requirements-dev.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

#--------------------------------------------------
# Development stage
#--------------------------------------------------
FROM base as development

# Install development dependencies
RUN pip install --no-cache-dir -r requirements-dev.txt

# Copy source code
COPY . .

# Change ownership to non-root user
RUN chown -R mlops:mlops /app

USER mlops

# Expose port
EXPOSE 8000

# Default command for development
CMD ["python", "-m", "uvicorn", "serving.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

#--------------------------------------------------
# Production stage
#--------------------------------------------------
FROM base as production

# Copy only necessary files
COPY serving/ ./serving/
COPY training/ ./training/
COPY monitoring/ ./monitoring/
COPY config/ ./config/

# Create necessary directories
RUN mkdir -p logs models data

# Change ownership to non-root user
RUN chown -R mlops:mlops /app

USER mlops

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Production command
CMD ["python", "-m", "uvicorn", "serving.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]

#--------------------------------------------------
# Training stage (for model training jobs)
#--------------------------------------------------
FROM base as training

# Install additional training dependencies
RUN pip install --no-cache-dir jupyter optuna hyperopt

# Copy training-specific code
COPY training/ ./training/
COPY monitoring/ ./monitoring/
COPY config/ ./config/

# Create directories for training artifacts
RUN mkdir -p models experiments logs data

# Change ownership
RUN chown -R mlops:mlops /app

USER mlops

# Default command for training
CMD ["python", "-m", "training.pipelines.training_pipeline"]

#--------------------------------------------------
# Monitoring stage (for monitoring services)
#--------------------------------------------------
FROM base as monitoring

# Install monitoring-specific dependencies
RUN pip install --no-cache-dir prometheus_client grafana-api

# Copy monitoring code
COPY monitoring/ ./monitoring/
COPY config/ ./config/

# Create directories
RUN mkdir -p logs dashboards

# Change ownership
RUN chown -R mlops:mlops /app

USER mlops

# Expose monitoring port
EXPOSE 9090

# Default command for monitoring
CMD ["python", "-m", "monitoring.model_monitor"]