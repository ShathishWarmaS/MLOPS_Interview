# Fraud Detection System Makefile
# Production-grade MLOps project automation

.PHONY: help install setup train serve test clean lint format docker-build docker-up docker-down k8s-deploy

# Default target
help:
	@echo "🛡️  Fraud Detection System - MLOps Project"
	@echo "============================================="
	@echo ""
	@echo "🚀 Quick Start:"
	@echo "  make setup      - Complete project setup"
	@echo "  make train      - Train fraud detection model"
	@echo "  make serve      - Start API server"
	@echo "  make test       - Run all tests"
	@echo ""
	@echo "📦 Installation:"
	@echo "  make install    - Install Python dependencies"
	@echo "  make setup      - Full project setup with venv"
	@echo ""
	@echo "🧠 Model Training:"
	@echo "  make train      - Train model with default settings"
	@echo "  make train-large - Train with larger dataset"
	@echo "  make train-quick - Quick training for testing"
	@echo ""
	@echo "🌐 API Server:"
	@echo "  make serve      - Start development server"
	@echo "  make serve-prod - Start production server"
	@echo ""
	@echo "🧪 Testing:"
	@echo "  make test       - Run all tests"
	@echo "  make test-unit  - Run unit tests only"
	@echo "  make test-api   - Test API endpoints"
	@echo "  make test-load  - Run load tests"
	@echo ""
	@echo "🐳 Docker:"
	@echo "  make docker-build - Build Docker images"
	@echo "  make docker-up    - Start with Docker Compose"
	@echo "  make docker-down  - Stop Docker services"
	@echo ""
	@echo "☸️  Kubernetes:"
	@echo "  make k8s-deploy   - Deploy to Kubernetes"
	@echo "  make k8s-delete   - Remove from Kubernetes"
	@echo ""
	@echo "🛠️  Development:"
	@echo "  make lint       - Run code linting"
	@echo "  make format     - Format code"
	@echo "  make clean      - Clean generated files"
	@echo ""

# Setup and Installation
setup:
	@echo "🚀 Setting up fraud detection system..."
	./scripts/setup.sh --full

install:
	@echo "📦 Installing dependencies..."
	pip install -r requirements.txt

install-dev:
	@echo "📦 Installing development dependencies..."
	pip install -r requirements.txt
	pip install pytest pytest-asyncio pytest-cov black isort flake8 mypy jupyter

# Model Training
train:
	@echo "🧠 Training fraud detection model..."
	python src/training/train_model.py --samples 50000

train-large:
	@echo "🧠 Training with large dataset..."
	python src/training/train_model.py --samples 100000

train-quick:
	@echo "🧠 Quick training for testing..."
	python src/training/train_model.py --samples 5000

# API Server
serve:
	@echo "🌐 Starting development server..."
	python src/api/main.py

serve-prod:
	@echo "🌐 Starting production server..."
	gunicorn src.api.main:app -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000

# Testing
test:
	@echo "🧪 Running all tests..."
	pytest tests/ -v --cov=src --cov-report=html

test-unit:
	@echo "🧪 Running unit tests..."
	pytest tests/unit/ -v

test-integration:
	@echo "🧪 Running integration tests..."
	pytest tests/integration/ -v

test-api:
	@echo "🧪 Testing API endpoints..."
	@echo "Starting API server for testing..."
	@python src/api/main.py &
	@API_PID=$$!; \
	sleep 5; \
	echo "Testing health endpoint..."; \
	curl -f http://localhost:8000/health || echo "Health check failed"; \
	echo "Testing prediction endpoint..."; \
	curl -X POST "http://localhost:8000/predict" \
		-H "Content-Type: application/json" \
		-d '{"amount": 150.0, "merchant": "Amazon", "merchant_category": "online", "hour": 14, "day_of_week": 2, "user_age": 35, "account_age_days": 456}' || echo "Prediction test failed"; \
	kill $$API_PID

test-load:
	@echo "🧪 Running load tests..."
	@if command -v locust &> /dev/null; then \
		locust -f tests/load/locustfile.py --host=http://localhost:8000; \
	else \
		echo "Locust not installed. Install with: pip install locust"; \
	fi

# Docker
docker-build:
	@echo "🐳 Building Docker images..."
	docker build -t fraud-detection-api:latest -f docker/Dockerfile.api .
	@echo "✅ Docker images built successfully"

docker-up:
	@echo "🐳 Starting services with Docker Compose..."
	docker-compose up -d
	@echo "✅ Services started. Check status with: docker-compose ps"

docker-down:
	@echo "🐳 Stopping Docker services..."
	docker-compose down
	@echo "✅ Services stopped"

docker-logs:
	@echo "📋 Showing Docker logs..."
	docker-compose logs -f

# Kubernetes
k8s-deploy:
	@echo "☸️  Deploying to Kubernetes..."
	kubectl apply -f k8s/
	@echo "✅ Deployed to Kubernetes"

k8s-delete:
	@echo "☸️  Removing from Kubernetes..."
	kubectl delete -f k8s/
	@echo "✅ Removed from Kubernetes"

k8s-status:
	@echo "☸️  Kubernetes status..."
	kubectl get pods,services,deployments -n fraud-detection

# Development Tools
lint:
	@echo "🔍 Running code linting..."
	flake8 src/ tests/ --max-line-length=100 --ignore=E203,W503
	mypy src/ --ignore-missing-imports

format:
	@echo "🎨 Formatting code..."
	black src/ tests/ --line-length=100
	isort src/ tests/ --profile black

clean:
	@echo "🧹 Cleaning up..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf htmlcov/
	rm -rf dist/
	rm -rf build/
	rm -rf logs/*.log

# Data and Models
generate-data:
	@echo "📊 Generating synthetic data..."
	python scripts/generate_data.py

download-models:
	@echo "📥 Downloading pre-trained models..."
	@echo "No pre-trained models available yet. Run 'make train' to create models."

# Monitoring
start-monitoring:
	@echo "📊 Starting monitoring stack..."
	docker-compose up -d prometheus grafana
	@echo "✅ Monitoring available at:"
	@echo "  📈 Prometheus: http://localhost:9090"
	@echo "  📊 Grafana: http://localhost:3000 (admin/admin)"

logs:
	@echo "📋 Showing application logs..."
	tail -f logs/fraud_detection.log

# Database
db-migrate:
	@echo "🗄️  Running database migrations..."
	alembic upgrade head

db-reset:
	@echo "🗄️  Resetting database..."
	docker-compose down postgres
	docker volume rm fraud-detection_postgres_data
	docker-compose up -d postgres

# Performance
benchmark:
	@echo "⚡ Running performance benchmarks..."
	python scripts/benchmark.py

profile:
	@echo "📊 Profiling application..."
	python -m cProfile -o profile.stats src/api/main.py

# Security
security-scan:
	@echo "🔒 Running security scans..."
	@if command -v bandit &> /dev/null; then \
		bandit -r src/; \
	else \
		echo "Bandit not installed. Install with: pip install bandit"; \
	fi

# Documentation
docs:
	@echo "📚 Generating documentation..."
	@if command -v sphinx-build &> /dev/null; then \
		sphinx-build -b html docs/ docs/_build/html; \
		echo "Documentation available at docs/_build/html/index.html"; \
	else \
		echo "Sphinx not installed. Install with: pip install sphinx"; \
	fi

# Environment Management
env-create:
	@echo "🐍 Creating Python virtual environment..."
	python3 -m venv fraud_detection_env
	@echo "✅ Virtual environment created. Activate with:"
	@echo "source fraud_detection_env/bin/activate"

env-activate:
	@echo "🐍 To activate virtual environment, run:"
	@echo "source fraud_detection_env/bin/activate"

# MLflow
mlflow-ui:
	@echo "🔬 Starting MLflow UI..."
	mlflow ui --port 5000 --host 0.0.0.0

# Full Workflow
full-pipeline:
	@echo "🚀 Running full ML pipeline..."
	make clean
	make train
	make test
	make serve &
	sleep 10
	make test-api
	@echo "✅ Full pipeline completed successfully"

# Production Deployment
deploy-staging:
	@echo "🚀 Deploying to staging..."
	@echo "Set ENVIRONMENT=staging"
	@export ENVIRONMENT=staging && make docker-build && make docker-up

deploy-production:
	@echo "🚀 Deploying to production..."
	@echo "⚠️  Make sure you've tested in staging first!"
	@read -p "Are you sure you want to deploy to production? (y/N): " confirm && \
	if [ "$$confirm" = "y" ] || [ "$$confirm" = "Y" ]; then \
		export ENVIRONMENT=production && make docker-build && make k8s-deploy; \
	else \
		echo "Deployment cancelled"; \
	fi

# Status and Health
status:
	@echo "📊 System Status"
	@echo "==============="
	@echo ""
	@echo "🐳 Docker Services:"
	@docker-compose ps 2>/dev/null || echo "Docker Compose not running"
	@echo ""
	@echo "☸️  Kubernetes Pods:"
	@kubectl get pods -n fraud-detection 2>/dev/null || echo "Kubernetes not available"
	@echo ""
	@echo "🌐 API Health:"
	@curl -s http://localhost:8000/health | jq . 2>/dev/null || echo "API not responding"

# Version Information
version:
	@echo "📋 Version Information"
	@echo "====================="
	@echo "Project: Fraud Detection System v1.0.0"
	@echo "Python: $(shell python --version 2>&1)"
	@echo "Docker: $(shell docker --version 2>&1 || echo 'Not installed')"
	@echo "Kubernetes: $(shell kubectl version --client --short 2>&1 || echo 'Not installed')"