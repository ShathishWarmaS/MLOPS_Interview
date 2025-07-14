#!/bin/bash

# ==============================================================================
# Fraud Detection System Setup Script
# Production-grade MLOps project setup automation
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
PROJECT_NAME="fraud-detection"
PYTHON_VERSION="3.9"
VENV_NAME="fraud_detection_env"

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
    
    # Check Docker
    if command -v docker &> /dev/null; then
        DOCKER_VER=$(docker --version | cut -d' ' -f3 | cut -d',' -f1)
        log_info "Docker version: $DOCKER_VER"
    else
        log_warning "Docker not found. Docker features will not be available."
    fi
    
    # Check Docker Compose
    if command -v docker-compose &> /dev/null; then
        COMPOSE_VER=$(docker-compose --version | cut -d' ' -f3 | cut -d',' -f1)
        log_info "Docker Compose version: $COMPOSE_VER"
    else
        log_warning "Docker Compose not found. Local infrastructure setup will not be available."
    fi
    
    log_success "Dependency check completed"
}

setup_project_structure() {
    log_header "📁 Setting up Project Structure..."
    
    # Create necessary directories
    directories=(
        "data/raw"
        "data/processed" 
        "data/models"
        "logs"
        "tests/unit"
        "tests/integration"
        "tests/load"
        "monitoring/grafana/dashboards"
        "monitoring/grafana/datasources"
        "scripts"
    )
    
    for dir in "${directories[@]}"; do
        mkdir -p "$dir"
        log_info "Created directory: $dir"
    done
    
    # Create empty __init__.py files
    find src -type d -exec touch {}/__init__.py \;
    
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
    pip install --upgrade pip
    
    # Install requirements
    if [[ -f "requirements.txt" ]]; then
        log_info "Installing Python dependencies..."
        pip install -r requirements.txt
        log_success "Python dependencies installed"
    else
        log_warning "requirements.txt not found"
    fi
    
    # Install development dependencies
    log_info "Installing development tools..."
    pip install pytest pytest-asyncio black isort flake8 mypy jupyter
    
    log_success "Python environment setup completed"
}

create_missing_files() {
    log_header "📄 Creating Missing Implementation Files..."
    
    # Create middleware.py
    cat > src/api/middleware.py << 'EOF'
"""
Custom middleware for the Fraud Detection API
"""

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
import time
import uuid
import logging

logger = logging.getLogger(__name__)

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for request logging"""
    
    async def dispatch(self, request: Request, call_next):
        # Generate request ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        # Add to headers
        start_time = time.time()
        
        # Process request
        response = await call_next(request)
        
        # Calculate duration
        duration_ms = (time.time() - start_time) * 1000
        
        # Log request
        logger.info(
            f"{request.method} {request.url.path} - "
            f"Status: {response.status_code} - "
            f"Duration: {duration_ms:.2f}ms - "
            f"Request ID: {request_id}"
        )
        
        # Add headers to response
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time"] = f"{duration_ms:.2f}ms"
        
        return response

class MetricsMiddleware(BaseHTTPMiddleware):
    """Middleware for metrics collection"""
    
    def __init__(self, app):
        super().__init__(app)
        self.request_count = 0
        self.total_latency = 0.0
    
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        # Process request
        response = await call_next(request)
        
        # Update metrics
        self.request_count += 1
        latency = (time.time() - start_time) * 1000
        self.total_latency += latency
        
        return response
EOF
    
    # Create predictor.py
    cat > src/inference/predictor.py << 'EOF'
"""
Fraud Prediction Service
Production-ready model inference
"""

import joblib
import numpy as np
import pandas as pd
import asyncio
import logging
from typing import Dict, Any
import time
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from api.models import TransactionRequest, RiskLevel
from shared.config import config

logger = logging.getLogger(__name__)

class FraudPredictor:
    """Production fraud detection predictor"""
    
    def __init__(self):
        self.model = None
        self.data_processor = None
        self.model_version = "1.0.0"
        self.load_time = None
        self.prediction_count = 0
        
    async def load_model(self):
        """Load trained model and preprocessor"""
        try:
            logger.info("Loading fraud detection model...")
            
            model_path = Path("data/models/best_fraud_model.pkl")
            processor_path = Path("data/models/data_processor.pkl")
            
            if not model_path.exists():
                # Create a dummy model for demo purposes
                logger.warning("Model files not found. Creating dummy model for demo...")
                await self._create_dummy_model()
                return
            
            # Load model and processor
            self.model = joblib.load(model_path)
            self.data_processor = joblib.load(processor_path)
            
            self.load_time = time.time()
            logger.info(f"Model loaded successfully. Version: {self.model_version}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            # Create dummy model as fallback
            await self._create_dummy_model()
    
    async def _create_dummy_model(self):
        """Create a dummy model for demo purposes"""
        from sklearn.ensemble import RandomForestClassifier
        from training.data_processor import FraudDataProcessor
        
        logger.info("Creating dummy model for demonstration...")
        
        # Create dummy processor
        self.data_processor = FraudDataProcessor()
        
        # Create dummy model
        self.model = RandomForestClassifier(n_estimators=10, random_state=42)
        
        # Train on dummy data
        dummy_data = self.data_processor.generate_synthetic_data(n_samples=1000)
        processed_data = self.data_processor.preprocess_features(dummy_data, is_training=True)
        
        feature_columns = [
            'amount_log', 'hour', 'day_of_week', 'user_age', 'account_age_days',
            'merchant_category_encoded', 'is_weekend', 'is_night', 
            'is_business_hours', 'account_age_weeks', 'is_new_account',
            'amount_bin_encoded', 'age_group_encoded', 'risk_score'
        ]
        
        # Only use features that exist
        available_features = [f for f in feature_columns if f in processed_data.columns]
        
        X = processed_data[available_features]
        y = processed_data['is_fraud']
        
        self.model.fit(X, y)
        self.load_time = time.time()
        
        logger.info("Dummy model created and trained")
    
    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        return self.model is not None and self.data_processor is not None
    
    async def predict(self, request: TransactionRequest) -> Dict[str, Any]:
        """Make fraud prediction"""
        if not self.is_loaded():
            raise RuntimeError("Model not loaded")
        
        try:
            # Convert request to DataFrame
            transaction_data = self._request_to_dataframe(request)
            
            # Preprocess features
            processed_data = self.data_processor.preprocess_features(
                transaction_data, 
                is_training=False
            )
            
            # Get available features
            feature_columns = [
                'amount_log', 'hour', 'day_of_week', 'user_age', 'account_age_days',
                'merchant_category_encoded', 'is_weekend', 'is_night', 
                'is_business_hours', 'account_age_weeks', 'is_new_account',
                'amount_bin_encoded', 'age_group_encoded', 'risk_score'
            ]
            
            available_features = [f for f in feature_columns if f in processed_data.columns]
            X = processed_data[available_features].values
            
            # Make prediction
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(X)[0]
                fraud_probability = probabilities[1] if len(probabilities) > 1 else probabilities[0]
                confidence = max(probabilities) - min(probabilities)
            else:
                # Fallback for models without predict_proba
                fraud_probability = float(np.random.random() * 0.3)  # Demo values
                confidence = 0.8
            
            # Determine risk level
            risk_level = self._calculate_risk_level(fraud_probability)
            
            # Update prediction count
            self.prediction_count += 1
            
            return {
                "fraud_probability": float(fraud_probability),
                "risk_level": risk_level,
                "confidence": float(confidence),
                "model_version": self.model_version
            }
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise
    
    def _request_to_dataframe(self, request: TransactionRequest) -> pd.DataFrame:
        """Convert request to DataFrame"""
        data = {
            'amount': [request.amount],
            'hour': [request.hour],
            'day_of_week': [request.day_of_week],
            'merchant_category': [request.merchant_category],
            'user_age': [request.user_age],
            'account_age_days': [request.account_age_days]
        }
        
        return pd.DataFrame(data)
    
    def _calculate_risk_level(self, probability: float) -> RiskLevel:
        """Calculate risk level based on probability"""
        if probability >= config.HIGH_RISK_THRESHOLD:
            return RiskLevel.HIGH
        elif probability >= config.LOW_RISK_THRESHOLD:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "model_version": self.model_version,
            "load_time": self.load_time,
            "prediction_count": self.prediction_count,
            "is_loaded": self.is_loaded()
        }
    
    async def cleanup(self):
        """Cleanup resources"""
        logger.info("Cleaning up predictor resources...")
EOF

    # Create metrics collector
    cat > src/monitoring/metrics_collector.py << 'EOF'
"""
Metrics Collection for Fraud Detection System
"""

import asyncio
import logging
import time
from typing import Dict, Any
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

class MetricsCollector:
    """Collect and store application metrics"""
    
    def __init__(self):
        self.metrics = defaultdict(list)
        self.counters = defaultdict(int)
        self.latencies = deque(maxlen=1000)  # Keep last 1000 latencies
        self.start_time = time.time()
        
    async def initialize(self):
        """Initialize metrics collector"""
        logger.info("Metrics collector initialized")
        
    async def log_prediction(self, request, prediction, latency: float, request_id: str):
        """Log prediction metrics"""
        self.counters['total_predictions'] += 1
        self.latencies.append(latency)
        
        # Log fraud rate
        if prediction['fraud_probability'] > 0.5:
            self.counters['fraud_predictions'] += 1
        
        if prediction['risk_level'] == 'high':
            self.counters['high_risk_predictions'] += 1
    
    async def get_current_metrics(self) -> Dict[str, Any]:
        """Get current metrics"""
        total_predictions = self.counters['total_predictions']
        uptime = time.time() - self.start_time
        
        # Calculate rates
        predictions_per_second = total_predictions / max(uptime, 1)
        fraud_rate = (self.counters['fraud_predictions'] / max(total_predictions, 1)) * 100
        high_risk_rate = (self.counters['high_risk_predictions'] / max(total_predictions, 1)) * 100
        
        # Calculate latency stats
        if self.latencies:
            latencies_sorted = sorted(self.latencies)
            avg_latency = sum(self.latencies) / len(self.latencies)
            p95_latency = latencies_sorted[int(0.95 * len(latencies_sorted))]
            p99_latency = latencies_sorted[int(0.99 * len(latencies_sorted))]
        else:
            avg_latency = p95_latency = p99_latency = 0.0
        
        return {
            'total_predictions': total_predictions,
            'predictions_per_second': predictions_per_second,
            'average_latency_ms': avg_latency,
            'p95_latency_ms': p95_latency,
            'p99_latency_ms': p99_latency,
            'fraud_rate': fraud_rate,
            'high_risk_rate': high_risk_rate,
            'uptime_seconds': uptime,
            'error_rate': 0.0  # TODO: Implement error tracking
        }
    
    async def health_check(self):
        """Health check for metrics collector"""
        return True
    
    async def close(self):
        """Close metrics collector"""
        logger.info("Metrics collector closed")
EOF

    log_success "Missing implementation files created"
}

create_configuration_files() {
    log_header "⚙️ Creating Configuration Files..."
    
    # Create .env file
    cat > .env << 'EOF'
# Environment Configuration
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=INFO

# API Configuration  
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=1

# Database Configuration
DATABASE_URL=postgresql://fraud_user:fraud_password@localhost:5432/fraud_detection

# Redis Configuration
REDIS_URL=redis://localhost:6379

# MLflow Configuration
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_EXPERIMENT_NAME=fraud_detection

# Model Configuration
MODEL_NAME=fraud_detection_production
MODEL_VERSION=latest
DEFAULT_FRAUD_THRESHOLD=0.5
HIGH_RISK_THRESHOLD=0.7
LOW_RISK_THRESHOLD=0.3

# Security
SECRET_KEY=your-secret-key-change-in-production

# Features
METRICS_ENABLED=true
FEATURE_STORE_ENABLED=true
ALERT_ENABLED=false
EOF

    # Create Makefile
    cat > Makefile << 'EOF'
.PHONY: help install train serve test clean docker-build docker-up

help:
	@echo "Available commands:"
	@echo "  install     - Install dependencies"
	@echo "  train       - Train the fraud detection model"
	@echo "  serve       - Start the API server"
	@echo "  test        - Run tests"
	@echo "  clean       - Clean up generated files"
	@echo "  docker-build - Build Docker images"
	@echo "  docker-up   - Start services with Docker Compose"

install:
	pip install -r requirements.txt

train:
	python src/training/train_model.py

serve:
	python src/api/main.py

test:
	pytest tests/ -v

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf .pytest_cache
	rm -rf logs/*

docker-build:
	docker build -t fraud-detection-api -f docker/Dockerfile.api .

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down
EOF

    # Create prometheus config
    mkdir -p monitoring
    cat > monitoring/prometheus.yml << 'EOF'
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'fraud-detection-api'
    static_configs:
      - targets: ['fraud-api:8000']
    metrics_path: /metrics
    scrape_interval: 30s
EOF

    log_success "Configuration files created"
}

setup_docker() {
    log_header "🐳 Setting up Docker Environment..."
    
    if ! command -v docker &> /dev/null; then
        log_warning "Docker not found. Skipping Docker setup."
        return
    fi
    
    # Build images
    log_info "Building Docker images..."
    if [[ -f "docker/Dockerfile.api" ]]; then
        docker build -t fraud-detection-api -f docker/Dockerfile.api .
        log_success "API Docker image built"
    fi
    
    # Start services
    if command -v docker-compose &> /dev/null; then
        log_info "Starting infrastructure services..."
        docker-compose up -d postgres redis mlflow
        
        # Wait for services to be ready
        log_info "Waiting for services to be ready..."
        sleep 10
        
        log_success "Infrastructure services started"
    fi
}

run_training() {
    log_header "🧠 Training Fraud Detection Model..."
    
    # Activate virtual environment
    source "$VENV_NAME/bin/activate"
    
    # Run training
    log_info "Starting model training..."
    python src/training/train_model.py --samples 10000
    
    log_success "Model training completed"
}

test_api() {
    log_header "🧪 Testing API..."
    
    # Start API in background
    log_info "Starting API server..."
    source "$VENV_NAME/bin/activate"
    python src/api/main.py &
    API_PID=$!
    
    # Wait for API to start
    sleep 5
    
    # Test health endpoint
    log_info "Testing health endpoint..."
    if curl -f http://localhost:8000/health; then
        log_success "Health check passed"
    else
        log_error "Health check failed"
    fi
    
    # Test prediction endpoint
    log_info "Testing prediction endpoint..."
    curl -X POST "http://localhost:8000/predict" \
         -H "Content-Type: application/json" \
         -d '{
           "amount": 150.0,
           "merchant": "Amazon",
           "merchant_category": "online",
           "hour": 14,
           "day_of_week": 2,
           "user_age": 35,
           "account_age_days": 456
         }'
    
    # Stop API
    kill $API_PID
    
    log_success "API testing completed"
}

main() {
    log_header "🚀 Fraud Detection System Setup"
    echo
    
    # Check if we're in the right directory
    if [[ ! -f "requirements.txt" ]]; then
        log_error "Please run this script from the project root directory"
        exit 1
    fi
    
    # Run setup steps
    check_dependencies
    setup_project_structure
    setup_python_environment
    create_missing_files
    create_configuration_files
    
    if [[ "${1:-}" == "--with-docker" ]]; then
        setup_docker
    fi
    
    if [[ "${1:-}" == "--full" ]] || [[ "${2:-}" == "--full" ]]; then
        run_training
        test_api
    fi
    
    echo
    log_header "🎉 Setup Complete!"
    echo
    log_info "Next steps:"
    echo "  1. Activate virtual environment: source $VENV_NAME/bin/activate"
    echo "  2. Train model: python src/training/train_model.py"
    echo "  3. Start API: python src/api/main.py"
    echo "  4. Test API: curl http://localhost:8000/health"
    echo
    log_info "For Docker setup: ./scripts/setup.sh --with-docker"
    log_info "For full setup with training: ./scripts/setup.sh --full"
    echo
    log_success "Happy coding! 🚀"
}

# Run main function
main "$@"