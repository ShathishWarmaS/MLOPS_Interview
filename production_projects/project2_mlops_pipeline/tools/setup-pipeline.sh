#!/bin/bash

# ==============================================================================
# MLOps Pipeline Setup Script
# Complete automation for production-grade MLOps pipeline setup
# ==============================================================================

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="mlops-pipeline"
PYTHON_VERSION="3.9"
VENV_NAME="mlops_pipeline_env"

# Functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $*"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $*"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $*"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $*"
}

log_header() {
    echo -e "${PURPLE}$*${NC}"
}

show_banner() {
    log_header "🚀 MLOps Pipeline Setup"
    log_header "=========================="
    echo ""
    log_info "This script will set up a complete production-grade MLOps pipeline"
    log_info "including training pipelines, model serving, CI/CD, and monitoring."
    echo ""
}

check_dependencies() {
    log_header "🔍 Checking Dependencies..."
    
    # Check Python
    if command -v python3 &> /dev/null; then
        PYTHON_VER=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
        log_info "Python version: $PYTHON_VER"
        if [[ "$PYTHON_VER" < "3.8" ]]; then
            log_error "Python 3.8+ required. Found: $PYTHON_VER"
            exit 1
        fi
    else
        log_error "Python 3 not found. Please install Python 3.8+"
        exit 1
    fi
    
    # Check Git
    if command -v git &> /dev/null; then
        GIT_VER=$(git --version | cut -d' ' -f3)
        log_info "Git version: $GIT_VER"
    else
        log_error "Git not found. Please install Git"
        exit 1
    fi
    
    # Check Docker
    if command -v docker &> /dev/null; then
        DOCKER_VER=$(docker --version | cut -d' ' -f3 | cut -d',' -f1)
        log_info "Docker version: $DOCKER_VER"
        
        # Check if Docker is running
        if ! docker info >/dev/null 2>&1; then
            log_warning "Docker is not running. Please start Docker."
        fi
    else
        log_warning "Docker not found. Docker features will not be available."
    fi
    
    # Check Docker Compose
    if command -v docker-compose &> /dev/null; then
        COMPOSE_VER=$(docker-compose --version | cut -d' ' -f3 | cut -d',' -f1)
        log_info "Docker Compose version: $COMPOSE_VER"
    else
        log_warning "Docker Compose not found. Infrastructure setup will be limited."
    fi
    
    # Check Kubernetes tools
    if command -v kubectl &> /dev/null; then
        KUBECTL_VER=$(kubectl version --client --short 2>/dev/null | cut -d' ' -f3 || echo "unknown")
        log_info "kubectl version: $KUBECTL_VER"
    else
        log_warning "kubectl not found. Kubernetes features will not be available."
    fi
    
    log_success "Dependency check completed"
}

setup_project_structure() {
    log_header "📁 Setting up Project Structure..."
    
    # Create directory structure
    directories=(
        "config/environments"
        "config/models"
        "config/pipelines"
        "data/raw"
        "data/processed"
        "data/models"
        "data/schemas"
        "data/samples"
        "data/validation"
        "logs"
        "reports"
        "tests/unit"
        "tests/integration"
        "tests/e2e"
        "tests/performance"
        "tests/security"
        "docs/architecture"
        "docs/deployment"
        "docs/monitoring"
        "docs/api"
        "infrastructure/terraform/environments"
        "infrastructure/terraform/modules"
        "infrastructure/kubernetes/base"
        "infrastructure/kubernetes/overlays"
        "infrastructure/helm/mlops-platform"
        "infrastructure/helm/monitoring"
        "serving/batch"
        "serving/streaming"
        "training/experiments/config"
        "training/experiments/hyperparams"
        "training/models"
        "training/utils"
        "monitoring/alerting/notifications"
        "monitoring/dashboards/grafana"
        "monitoring/dashboards/custom"
        ".github/workflows"
    )
    
    for dir in "${directories[@]}"; do
        mkdir -p "$dir"
        log_info "Created directory: $dir"
    done
    
    # Create __init__.py files for Python packages
    find . -type d \( -name training -o -name serving -o -name monitoring \) -exec find {} -type d -exec touch {}/__init__.py \;
    
    log_success "Project structure created"
}

setup_python_environment() {
    log_header "🐍 Setting up Python Environment..."
    
    # Create virtual environment
    if [[ ! -d "$VENV_NAME" ]]; then
        log_info "Creating virtual environment: $VENV_NAME"
        python3 -m venv "$VENV_NAME"
    else
        log_info "Virtual environment already exists: $VENV_NAME"
    fi
    
    # Activate virtual environment
    source "$VENV_NAME/bin/activate"
    
    # Upgrade pip
    log_info "Upgrading pip..."
    pip install --upgrade pip setuptools wheel
    
    # Install requirements
    if [[ -f "requirements.txt" ]]; then
        log_info "Installing Python dependencies..."
        pip install -r requirements.txt
        log_success "Python dependencies installed"
    else
        log_warning "requirements.txt not found"
    fi
    
    # Install development dependencies
    if [[ -f "requirements-dev.txt" ]]; then
        log_info "Installing development dependencies..."
        pip install -r requirements-dev.txt
        log_success "Development dependencies installed"
    fi
    
    log_success "Python environment setup completed"
}

setup_configuration() {
    log_header "⚙️ Setting up Configuration..."
    
    # Create environment configuration
    cat > config/environments/development.yaml << 'EOF'
# Development Environment Configuration
environment: development
debug: true

# Data sources
data_sources:
  synthetic:
    type: synthetic
    n_samples: 10000
    n_features: 10
    required: true

# Model training
training:
  baseline:
    model_type: logistic_regression
  advanced:
    models:
      random_forest:
        n_estimators: 100
        max_depth: 10
      gradient_boosting:
        n_estimators: 100
        learning_rate: 0.1

# Model serving
serving:
  host: "0.0.0.0"
  port: 8000
  workers: 1
  reload: true

# Monitoring
monitoring:
  enabled: true
  drift_detection: true
  performance_tracking: true
EOF

    # Create production configuration
    cat > config/environments/production.yaml << 'EOF'
# Production Environment Configuration
environment: production
debug: false

# Data sources
data_sources:
  database:
    type: database
    connection_string: "${DATABASE_URL}"
    required: true
  s3:
    type: s3
    bucket: "${S3_BUCKET}"
    prefix: "ml-data/"
    required: false

# Model training
training:
  baseline:
    model_type: random_forest
  advanced:
    models:
      random_forest:
        n_estimators: 200
        max_depth: 15
      gradient_boosting:
        n_estimators: 200
        learning_rate: 0.05
      neural_network:
        hidden_layers: [128, 64, 32]
        dropout: 0.2

# Model serving
serving:
  host: "0.0.0.0"
  port: 8000
  workers: 4
  reload: false

# Monitoring
monitoring:
  enabled: true
  drift_detection: true
  performance_tracking: true
  alerting: true
EOF

    # Create pipeline configuration
    cat > config/pipelines/training_pipeline.yaml << 'EOF'
# Training Pipeline Configuration
data_validation:
  quality_rules:
    completeness_check:
      type: completeness
      column: target
      threshold: 0.99
      critical: true
    range_check:
      type: range
      column: feature_0
      min: -5.0
      max: 5.0
      critical: false

feature_engineering:
  numerical_features:
    feature_squared:
      type: polynomial
      column: feature_0
      degree: 2
    feature_interaction:
      type: interaction
      column1: feature_0
      column2: feature_1

model_selection:
  primary_metric: f1_score
  secondary_metrics: [accuracy, roc_auc]
  minimize: false

model_registry:
  model_name: mlops_pipeline_model
  stage: Staging
EOF

    # Create .env template
    cat > .env.example << 'EOF'
# Environment Configuration Template
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=INFO

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=1

# Database
DATABASE_URL=postgresql://user:password@localhost:5432/mlops
REDIS_URL=redis://localhost:6379

# MLflow
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_EXPERIMENT_NAME=mlops_pipeline

# Model Configuration
MODEL_NAME=mlops_pipeline_model
MODEL_VERSION=latest

# Cloud Storage
S3_BUCKET=mlops-data-bucket
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key

# Monitoring
PROMETHEUS_URL=http://localhost:9090
GRAFANA_URL=http://localhost:3000

# Security
SECRET_KEY=your-secret-key-change-in-production
JWT_SECRET=your-jwt-secret

# Features
METRICS_ENABLED=true
MONITORING_ENABLED=true
ALERTING_ENABLED=false
EOF

    # Copy to actual .env if it doesn't exist
    if [[ ! -f ".env" ]]; then
        cp .env.example .env
        log_info "Created .env file from template"
    fi
    
    log_success "Configuration setup completed"
}

setup_docker() {
    log_header "🐳 Setting up Docker Environment..."
    
    if ! command -v docker &> /dev/null; then
        log_warning "Docker not found. Skipping Docker setup."
        return
    fi
    
    # Create Dockerfile for training
    cat > Dockerfile.training << 'EOF'
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY training/ ./training/
COPY config/ ./config/
COPY data/ ./data/

# Create non-root user
RUN useradd -m -u 1000 mlops && chown -R mlops:mlops /app
USER mlops

# Default command
CMD ["python", "-m", "training.pipelines.training_pipeline"]
EOF

    # Create Dockerfile for serving
    cat > Dockerfile.serving << 'EOF'
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY serving/ ./serving/
COPY monitoring/ ./monitoring/
COPY config/ ./config/

# Create non-root user
RUN useradd -m -u 1000 mlops && chown -R mlops:mlops /app
USER mlops

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

CMD ["python", "-m", "serving.api.main"]
EOF

    # Create docker-compose.yml
    cat > docker-compose.yml << 'EOF'
version: '3.8'

services:
  # MLOps API
  mlops-api:
    build:
      context: .
      dockerfile: Dockerfile.serving
    ports:
      - "8000:8000"
    environment:
      - ENVIRONMENT=development
      - MLFLOW_TRACKING_URI=http://mlflow:5000
      - REDIS_URL=redis://redis:6379
      - DATABASE_URL=postgresql://mlops:mlops@postgres:5432/mlops
    depends_on:
      - redis
      - postgres
      - mlflow
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  # MLflow Tracking Server
  mlflow:
    image: ghcr.io/mlflow/mlflow:latest
    ports:
      - "5000:5000"
    environment:
      - MLFLOW_BACKEND_STORE_URI=postgresql://mlflow:mlflow@postgres:5432/mlflow
      - MLFLOW_DEFAULT_ARTIFACT_ROOT=s3://mlflow-artifacts
    depends_on:
      - postgres
    command: >
      mlflow server
      --backend-store-uri postgresql://mlflow:mlflow@postgres:5432/mlflow
      --default-artifact-root ./artifacts
      --host 0.0.0.0
      --port 5000
    restart: unless-stopped

  # PostgreSQL Database
  postgres:
    image: postgres:15
    environment:
      - POSTGRES_USER=mlops
      - POSTGRES_PASSWORD=mlops
      - POSTGRES_DB=mlops
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./infrastructure/postgres/init.sql:/docker-entrypoint-initdb.d/init.sql
    restart: unless-stopped

  # Redis Cache
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped

  # Prometheus Monitoring
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
    restart: unless-stopped

  # Grafana Dashboards
  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana/dashboards:/var/lib/grafana/dashboards
      - ./monitoring/grafana/provisioning:/etc/grafana/provisioning
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:
  prometheus_data:
  grafana_data:
EOF

    # Create monitoring configuration
    mkdir -p monitoring
    cat > monitoring/prometheus.yml << 'EOF'
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'mlops-api'
    static_configs:
      - targets: ['mlops-api:8000']
    metrics_path: /metrics
    scrape_interval: 30s

  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']
EOF

    log_success "Docker environment setup completed"
}

setup_github_actions() {
    log_header "🔄 Setting up GitHub Actions..."
    
    # The workflow file already exists, just ensure the directory is created
    mkdir -p .github/workflows
    
    # Create additional workflow for data pipeline
    cat > .github/workflows/data-pipeline.yml << 'EOF'
name: Data Pipeline

on:
  schedule:
    - cron: '0 0 * * *'  # Daily at midnight
  workflow_dispatch:

jobs:
  data-pipeline:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
      
      - name: Run data pipeline
        run: |
          python -m training.pipelines.training_pipeline --pipeline data
EOF

    log_success "GitHub Actions setup completed"
}

setup_infrastructure() {
    log_header "☸️  Setting up Infrastructure as Code..."
    
    # Create Kubernetes manifests
    mkdir -p infrastructure/kubernetes/base
    
    cat > infrastructure/kubernetes/base/deployment.yaml << 'EOF'
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mlops-api
  labels:
    app: mlops-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: mlops-api
  template:
    metadata:
      labels:
        app: mlops-api
    spec:
      containers:
      - name: mlops-api
        image: mlops-api:latest
        ports:
        - containerPort: 8000
        env:
        - name: ENVIRONMENT
          value: "production"
        - name: MLFLOW_TRACKING_URI
          value: "http://mlflow:5000"
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 10
EOF

    cat > infrastructure/kubernetes/base/service.yaml << 'EOF'
apiVersion: v1
kind: Service
metadata:
  name: mlops-api-service
spec:
  selector:
    app: mlops-api
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
EOF

    # Create Helm chart
    mkdir -p infrastructure/helm/mlops-platform/templates
    
    cat > infrastructure/helm/mlops-platform/Chart.yaml << 'EOF'
apiVersion: v2
name: mlops-platform
description: A Helm chart for MLOps Platform
type: application
version: 0.1.0
appVersion: "1.0"
EOF

    cat > infrastructure/helm/mlops-platform/values.yaml << 'EOF'
replicaCount: 3

image:
  repository: mlops-api
  pullPolicy: IfNotPresent
  tag: "latest"

service:
  type: LoadBalancer
  port: 80

ingress:
  enabled: true
  className: "nginx"
  annotations: {}
  hosts:
    - host: api.mlops.local
      paths:
        - path: /
          pathType: Prefix
  tls: []

resources:
  limits:
    cpu: 500m
    memory: 1Gi
  requests:
    cpu: 250m
    memory: 512Mi

autoscaling:
  enabled: true
  minReplicas: 3
  maxReplicas: 10
  targetCPUUtilizationPercentage: 80

nodeSelector: {}

tolerations: []

affinity: {}
EOF

    log_success "Infrastructure setup completed"
}

setup_monitoring() {
    log_header "📊 Setting up Monitoring and Observability..."
    
    # Create Grafana dashboard
    mkdir -p monitoring/grafana/dashboards
    
    cat > monitoring/grafana/dashboards/mlops-dashboard.json << 'EOF'
{
  "dashboard": {
    "id": null,
    "title": "MLOps Pipeline Dashboard",
    "tags": ["mlops", "monitoring"],
    "style": "dark",
    "timezone": "browser",
    "panels": [
      {
        "id": 1,
        "title": "Prediction Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(http_requests_total{endpoint=\"/predict\"}[5m])",
            "legendFormat": "Predictions/sec"
          }
        ]
      },
      {
        "id": 2,
        "title": "Model Accuracy",
        "type": "stat",
        "targets": [
          {
            "expr": "model_accuracy",
            "legendFormat": "Accuracy"
          }
        ]
      },
      {
        "id": 3,
        "title": "Response Time",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))",
            "legendFormat": "95th percentile"
          }
        ]
      }
    ],
    "time": {
      "from": "now-1h",
      "to": "now"
    },
    "refresh": "5s"
  }
}
EOF

    # Create alerting rules
    mkdir -p monitoring/alerting
    
    cat > monitoring/alerting/rules.yml << 'EOF'
groups:
  - name: mlops_alerts
    rules:
      - alert: HighErrorRate
        expr: rate(http_errors_total[5m]) > 0.1
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "High error rate detected"
          description: "Error rate is {{ $value }} errors per second"

      - alert: HighLatency
        expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High latency detected"
          description: "95th percentile latency is {{ $value }}s"

      - alert: ModelAccuracyDrop
        expr: model_accuracy < 0.8
        for: 10m
        labels:
          severity: critical
        annotations:
          summary: "Model accuracy has dropped"
          description: "Model accuracy is {{ $value }}"
EOF

    log_success "Monitoring setup completed"
}

run_initial_tests() {
    log_header "🧪 Running Initial Tests..."
    
    # Activate virtual environment
    source "$VENV_NAME/bin/activate"
    
    # Run pipeline configuration validation
    log_info "Validating pipeline configuration..."
    python -c "
import sys
sys.path.append('.')
from training.pipelines.training_pipeline import PipelineRunner

runner = PipelineRunner()
if runner.validate_config():
    print('✅ Configuration validation passed')
else:
    print('❌ Configuration validation failed')
    sys.exit(1)
"
    
    # Test data pipeline
    log_info "Testing data pipeline..."
    python -c "
import sys
sys.path.append('.')
from training.pipelines.training_pipeline import PipelineRunner

runner = PipelineRunner()
runner.config['data_sources']['synthetic']['n_samples'] = 1000

try:
    results = runner.run_data_pipeline()
    print(f'✅ Data pipeline test passed. Generated {len(results[\"features\"])} samples')
except Exception as e:
    print(f'❌ Data pipeline test failed: {e}')
    sys.exit(1)
"
    
    log_success "Initial tests completed"
}

show_next_steps() {
    log_header "🎉 Setup Complete!"
    echo ""
    log_info "Your MLOps pipeline is ready! Here's what you can do next:"
    echo ""
    
    echo "📖 Getting Started:"
    echo "  1. Activate the environment: source $VENV_NAME/bin/activate"
    echo "  2. Run the training pipeline: python -m training.pipelines.training_pipeline"
    echo "  3. Start the API server: python -m serving.api.main"
    echo "  4. Test the API: curl http://localhost:8000/health"
    echo ""
    
    echo "🐳 Docker Commands:"
    echo "  • Start infrastructure: docker-compose up -d"
    echo "  • View logs: docker-compose logs -f"
    echo "  • Stop services: docker-compose down"
    echo ""
    
    echo "🔄 CI/CD:"
    echo "  • Commit changes to trigger GitHub Actions"
    echo "  • View workflows at: https://github.com/your-repo/actions"
    echo ""
    
    echo "📊 Monitoring:"
    echo "  • MLflow UI: http://localhost:5000"
    echo "  • Prometheus: http://localhost:9090"
    echo "  • Grafana: http://localhost:3000 (admin/admin)"
    echo ""
    
    echo "📚 Documentation:"
    echo "  • Project README: README.md"
    echo "  • API docs: http://localhost:8000/docs"
    echo "  • Architecture docs: docs/architecture.md"
    echo ""
    
    log_success "Happy MLOps engineering! 🚀"
}

main() {
    # Check if we're in the right directory
    if [[ ! -f "requirements.txt" ]]; then
        log_error "Please run this script from the project root directory"
        exit 1
    fi
    
    show_banner
    
    # Parse arguments
    INCLUDE_DOCKER=false
    INCLUDE_INFRA=false
    RUN_TESTS=false
    
    for arg in "$@"; do
        case $arg in
            --with-docker)
                INCLUDE_DOCKER=true
                ;;
            --with-infrastructure)
                INCLUDE_INFRA=true
                ;;
            --with-tests)
                RUN_TESTS=true
                ;;
            --full)
                INCLUDE_DOCKER=true
                INCLUDE_INFRA=true
                RUN_TESTS=true
                ;;
            --help)
                echo "Usage: $0 [options]"
                echo "Options:"
                echo "  --with-docker         Include Docker setup"
                echo "  --with-infrastructure Include Kubernetes/Helm setup"
                echo "  --with-tests          Run initial tests"
                echo "  --full                Include everything"
                echo "  --help                Show this help"
                exit 0
                ;;
        esac
    done
    
    # Run setup steps
    check_dependencies
    setup_project_structure
    setup_python_environment
    setup_configuration
    setup_github_actions
    setup_monitoring
    
    if [[ "$INCLUDE_DOCKER" == "true" ]]; then
        setup_docker
    fi
    
    if [[ "$INCLUDE_INFRA" == "true" ]]; then
        setup_infrastructure
    fi
    
    if [[ "$RUN_TESTS" == "true" ]]; then
        run_initial_tests
    fi
    
    show_next_steps
}

# Run main function with all arguments
main "$@"