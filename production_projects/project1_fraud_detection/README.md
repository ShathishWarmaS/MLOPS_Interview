# 🛡️ Project 1: Real-Time Fraud Detection System

## **Production-Grade MLOps Project from Scratch**

Build a complete fraud detection system that can handle **10,000+ transactions per second** with **sub-100ms latency** in production.

---

## 🎯 **Project Overview**

### **What You'll Build**
A full-stack ML system that:
- ✅ **Trains models** on historical transaction data
- ✅ **Serves predictions** via REST API in real-time  
- ✅ **Monitors performance** and detects drift
- ✅ **Scales automatically** based on traffic
- ✅ **Handles failures** gracefully with rollback
- ✅ **Tracks experiments** with MLflow
- ✅ **Deploys containerized** with Docker & Kubernetes

### **Business Value**
- Prevent fraudulent transactions in real-time
- Reduce false positives to improve customer experience
- Scale to handle peak traffic (Black Friday, etc.)
- Provide audit trail for compliance
- Monitor model performance continuously

---

## 🏗️ **System Architecture**

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Training  │    │   Feature   │    │   Model     │
│   Pipeline  │───▶│   Store     │◀───│  Registry   │
└─────────────┘    │  (Redis)    │    │ (MLflow)    │
                   └─────────────┘    └─────────────┘
                           ▲                 │
                           │                 ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Client    │───▶│   API       │───▶│  Inference  │
│ Application │    │  Gateway    │    │   Service   │
└─────────────┘    │ (FastAPI)   │    │  (Model)    │
                   └─────────────┘    └─────────────┘
                           │                 │
                           ▼                 ▼
                   ┌─────────────┐    ┌─────────────┐
                   │ Monitoring  │    │ Metrics     │
                   │ Dashboard   │    │ Database    │
                   │ (Grafana)   │    │(PostgreSQL)│
                   └─────────────┘    └─────────────┘
```

---

## 🚀 **Quick Start (15 minutes)**

### **1. Clone and Setup**
```bash
cd mlops_interview_prep/production_projects/project1_fraud_detection

# Install dependencies
pip install -r requirements.txt

# Start infrastructure
docker-compose up -d
```

### **2. Train Model**
```bash
python src/training/train_model.py
```

### **3. Start API Server**
```bash
python src/api/main.py
```

### **4. Test Prediction**
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"amount": 150.0, "merchant": "Amazon", "hour": 14}'
```

### **5. View Monitoring**
```bash
# Open browser
open http://localhost:3000  # Grafana dashboard
open http://localhost:5000  # MLflow UI
```

---

## 📂 **Project Structure**

```
project1_fraud_detection/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── docker-compose.yml                # Local development stack
├── Makefile                          # Build automation
│
├── src/                              # Source code
│   ├── api/                          # REST API service
│   │   ├── main.py                   # FastAPI application
│   │   ├── models.py                 # Pydantic models
│   │   ├── routes.py                 # API routes
│   │   └── middleware.py             # Custom middleware
│   │
│   ├── training/                     # Model training
│   │   ├── train_model.py            # Training script
│   │   ├── data_processor.py         # Data preprocessing
│   │   ├── feature_engineer.py       # Feature engineering
│   │   └── model_evaluator.py        # Model evaluation
│   │
│   ├── inference/                    # Model serving
│   │   ├── predictor.py              # Prediction service
│   │   ├── feature_store.py          # Feature management
│   │   └── model_loader.py           # Model loading
│   │
│   ├── monitoring/                   # Monitoring & observability
│   │   ├── metrics_collector.py      # Metrics collection
│   │   ├── drift_detector.py         # Data drift detection
│   │   └── alerting.py               # Alert management
│   │
│   └── shared/                       # Shared utilities
│       ├── config.py                 # Configuration
│       ├── database.py               # Database connections
│       ├── logging.py                # Logging setup
│       └── utils.py                  # Utility functions
│
├── tests/                            # Test suites
│   ├── unit/                         # Unit tests
│   ├── integration/                  # Integration tests
│   └── load/                         # Load tests
│
├── docker/                           # Docker configurations
│   ├── Dockerfile.api                # API service image
│   ├── Dockerfile.training           # Training image
│   └── Dockerfile.monitoring         # Monitoring image
│
├── k8s/                              # Kubernetes manifests
│   ├── namespace.yaml                # Namespace definition
│   ├── configmap.yaml               # Configuration
│   ├── secrets.yaml                 # Secrets management
│   ├── deployment.yaml              # Service deployments
│   ├── service.yaml                 # Service definitions
│   ├── ingress.yaml                 # Ingress configuration
│   └── hpa.yaml                     # Horizontal Pod Autoscaler
│
├── scripts/                          # Deployment & utility scripts
│   ├── build.sh                     # Build Docker images
│   ├── deploy.sh                    # Deploy to Kubernetes
│   ├── test.sh                      # Run test suite
│   └── generate_data.py             # Generate test data
│
├── docs/                             # Documentation
│   ├── api.md                       # API documentation
│   ├── deployment.md                # Deployment guide
│   ├── monitoring.md                # Monitoring guide
│   └── troubleshooting.md           # Troubleshooting guide
│
└── data/                             # Data directory
    ├── raw/                         # Raw data files
    ├── processed/                   # Processed data
    └── models/                      # Trained models
```

---

## 🛠️ **Implementation Guide**

### **Step 1: Core Data Science Components**

#### **1.1 Data Generation & Processing**
```python
# src/training/data_processor.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple, Dict
import logging

class FraudDataProcessor:
    """Process transaction data for fraud detection"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_stats = {}
        
    def generate_synthetic_data(self, n_samples: int = 10000) -> pd.DataFrame:
        """Generate synthetic transaction data"""
        np.random.seed(42)
        
        # Generate features
        data = {
            'amount': np.random.lognormal(4, 1.5, n_samples),
            'hour': np.random.randint(0, 24, n_samples),
            'day_of_week': np.random.randint(0, 7, n_samples),
            'merchant_category': np.random.choice(['grocery', 'gas', 'restaurant', 'retail', 'online'], n_samples),
            'user_age': np.random.randint(18, 80, n_samples),
            'account_age_days': np.random.randint(1, 3650, n_samples),
        }
        
        df = pd.DataFrame(data)
        
        # Create fraud labels (5% fraud rate)
        fraud_probability = self._calculate_fraud_probability(df)
        df['is_fraud'] = np.random.binomial(1, fraud_probability)
        
        return df
    
    def _calculate_fraud_probability(self, df: pd.DataFrame) -> np.ndarray:
        """Calculate fraud probability based on features"""
        base_prob = 0.05
        
        # Higher amounts more likely to be fraud
        amount_factor = np.clip(df['amount'] / 1000, 0, 2)
        
        # Late night transactions more suspicious
        time_factor = np.where((df['hour'] < 6) | (df['hour'] > 22), 2, 1)
        
        # Online transactions more risky
        merchant_factor = np.where(df['merchant_category'] == 'online', 1.5, 1)
        
        # New accounts more risky
        account_factor = np.where(df['account_age_days'] < 30, 2, 1)
        
        return np.clip(base_prob * amount_factor * time_factor * merchant_factor * account_factor, 0, 0.3)
    
    def preprocess_features(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """Preprocess features for model training/inference"""
        df_processed = df.copy()
        
        # Feature engineering
        df_processed['amount_log'] = np.log1p(df_processed['amount'])
        df_processed['is_weekend'] = (df_processed['day_of_week'] >= 5).astype(int)
        df_processed['is_night'] = ((df_processed['hour'] < 6) | (df_processed['hour'] > 22)).astype(int)
        
        # Encode categorical features
        categorical_features = ['merchant_category']
        for feature in categorical_features:
            if is_training:
                le = LabelEncoder()
                df_processed[f'{feature}_encoded'] = le.fit_transform(df_processed[feature])
                self.label_encoders[feature] = le
            else:
                if feature in self.label_encoders:
                    df_processed[f'{feature}_encoded'] = self.label_encoders[feature].transform(df_processed[feature])
        
        # Scale numerical features
        numerical_features = ['amount_log', 'hour', 'day_of_week', 'user_age', 'account_age_days']
        if is_training:
            df_processed[numerical_features] = self.scaler.fit_transform(df_processed[numerical_features])
            self.feature_stats = {
                'mean': self.scaler.mean_,
                'scale': self.scaler.scale_
            }
        else:
            df_processed[numerical_features] = self.scaler.transform(df_processed[numerical_features])
        
        return df_processed
```

#### **1.2 Model Training Pipeline**
```python
# src/training/train_model.py
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import pandas as pd
import numpy as np
import joblib
import logging
from data_processor import FraudDataProcessor
from model_evaluator import ModelEvaluator

class FraudModelTrainer:
    """Train fraud detection models with MLflow tracking"""
    
    def __init__(self, experiment_name: str = "fraud_detection"):
        mlflow.set_experiment(experiment_name)
        self.data_processor = FraudDataProcessor()
        self.evaluator = ModelEvaluator()
        
    def train_and_compare_models(self):
        """Train multiple models and select the best one"""
        
        # Generate and process data
        logging.info("Generating synthetic data...")
        df = self.data_processor.generate_synthetic_data(n_samples=50000)
        df_processed = self.data_processor.preprocess_features(df, is_training=True)
        
        # Prepare features and target
        feature_columns = [
            'amount_log', 'hour', 'day_of_week', 'user_age', 'account_age_days',
            'merchant_category_encoded', 'is_weekend', 'is_night'
        ]
        
        X = df_processed[feature_columns]
        y = df_processed['is_fraud']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train multiple models
        models = {
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            'gradient_boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
        }
        
        best_model = None
        best_score = 0
        
        for model_name, model in models.items():
            with mlflow.start_run(run_name=f"fraud_detection_{model_name}"):
                # Train model
                logging.info(f"Training {model_name}...")
                model.fit(X_train, y_train)
                
                # Evaluate model
                predictions = model.predict(X_test)
                probabilities = model.predict_proba(X_test)[:, 1]
                
                metrics = self.evaluator.calculate_metrics(y_test, predictions, probabilities)
                
                # Log parameters
                mlflow.log_params(model.get_params())
                
                # Log metrics
                for metric_name, metric_value in metrics.items():
                    mlflow.log_metric(metric_name, metric_value)
                
                # Log model
                mlflow.sklearn.log_model(
                    model,
                    f"fraud_model_{model_name}",
                    registered_model_name="fraud_detection_model"
                )
                
                # Log artifacts
                self._log_evaluation_artifacts(y_test, predictions, probabilities, model_name)
                
                logging.info(f"{model_name} - AUC: {metrics['auc_score']:.4f}")
                
                # Track best model
                if metrics['auc_score'] > best_score:
                    best_score = metrics['auc_score']
                    best_model = (model_name, model)
        
        # Save best model and preprocessor
        if best_model:
            model_name, model = best_model
            logging.info(f"Best model: {model_name} with AUC: {best_score:.4f}")
            
            # Save model and preprocessor
            joblib.dump(model, 'data/models/best_fraud_model.pkl')
            joblib.dump(self.data_processor, 'data/models/data_processor.pkl')
            
            # Register best model in MLflow
            with mlflow.start_run(run_name=f"best_model_{model_name}"):
                mlflow.sklearn.log_model(
                    model,
                    "model",
                    registered_model_name="fraud_detection_best"
                )
                mlflow.log_metric("best_auc", best_score)
        
        return best_model
    
    def _log_evaluation_artifacts(self, y_true, y_pred, y_prob, model_name):
        """Log evaluation artifacts to MLflow"""
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.metrics import roc_curve, precision_recall_curve
        
        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(f'confusion_matrix_{model_name}.png')
        mlflow.log_artifact(f'confusion_matrix_{model_name}.png')
        plt.close()
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc_score(y_true, y_prob):.4f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend()
        plt.savefig(f'roc_curve_{model_name}.png')
        mlflow.log_artifact(f'roc_curve_{model_name}.png')
        plt.close()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    trainer = FraudModelTrainer()
    trainer.train_and_compare_models()
```

### **Step 2: Production API Service**

#### **2.1 FastAPI Application**
```python
# src/api/main.py
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio
import logging
import time
from typing import Dict, List
import uvicorn

from models import TransactionRequest, PredictionResponse, HealthResponse
from routes import prediction_router, monitoring_router
from middleware import MetricsMiddleware
from inference.predictor import FraudPredictor
from monitoring.metrics_collector import MetricsCollector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Fraud Detection API",
    description="Real-time fraud detection service",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add custom metrics middleware
app.add_middleware(MetricsMiddleware)

# Global variables for model and metrics
predictor = None
metrics_collector = None

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    global predictor, metrics_collector
    
    try:
        logger.info("Initializing fraud detection service...")
        
        # Initialize predictor
        predictor = FraudPredictor()
        await predictor.load_model()
        
        # Initialize metrics collector
        metrics_collector = MetricsCollector()
        
        logger.info("Fraud detection service initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize service: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down fraud detection service...")
    
    if metrics_collector:
        await metrics_collector.close()

# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        # Check if model is loaded
        if predictor is None or not predictor.is_loaded():
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        return HealthResponse(
            status="healthy",
            timestamp=time.time(),
            version="1.0.0",
            model_loaded=True
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unhealthy")

# Prediction endpoint
@app.post("/predict", response_model=PredictionResponse)
async def predict_fraud(
    request: TransactionRequest,
    background_tasks: BackgroundTasks
) -> PredictionResponse:
    """Predict fraud probability for a transaction"""
    start_time = time.time()
    
    try:
        # Validate request
        if request.amount <= 0:
            raise HTTPException(status_code=400, detail="Amount must be positive")
        
        # Make prediction
        prediction_result = await predictor.predict(request)
        
        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000
        
        # Log metrics asynchronously
        background_tasks.add_task(
            log_prediction_metrics,
            request,
            prediction_result,
            latency_ms
        )
        
        # Create response
        response = PredictionResponse(
            transaction_id=request.transaction_id,
            fraud_probability=prediction_result["fraud_probability"],
            risk_level=prediction_result["risk_level"],
            confidence=prediction_result["confidence"],
            latency_ms=latency_ms,
            model_version=prediction_result["model_version"]
        )
        
        logger.info(f"Prediction completed - ID: {request.transaction_id}, "
                   f"Probability: {response.fraud_probability:.4f}, "
                   f"Latency: {latency_ms:.2f}ms")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

async def log_prediction_metrics(
    request: TransactionRequest,
    prediction_result: Dict,
    latency_ms: float
):
    """Log prediction metrics asynchronously"""
    try:
        if metrics_collector:
            await metrics_collector.log_prediction(
                request=request,
                prediction=prediction_result,
                latency=latency_ms
            )
    except Exception as e:
        logger.error(f"Failed to log metrics: {e}")

# Include routers
app.include_router(prediction_router, prefix="/api/v1")
app.include_router(monitoring_router, prefix="/monitoring")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
```

#### **2.2 Pydantic Models**
```python
# src/api/models.py
from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, Any
from enum import Enum
import uuid

class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class TransactionRequest(BaseModel):
    """Transaction data for fraud prediction"""
    transaction_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    amount: float = Field(..., gt=0, description="Transaction amount")
    merchant: str = Field(..., description="Merchant name")
    merchant_category: str = Field(..., description="Merchant category")
    hour: int = Field(..., ge=0, le=23, description="Hour of transaction (0-23)")
    day_of_week: int = Field(..., ge=0, le=6, description="Day of week (0=Monday)")
    user_age: int = Field(..., ge=18, le=120, description="User age")
    account_age_days: int = Field(..., ge=0, description="Account age in days")
    user_id: Optional[str] = Field(None, description="User ID")
    
    @validator('merchant_category')
    def validate_merchant_category(cls, v):
        valid_categories = ['grocery', 'gas', 'restaurant', 'retail', 'online']
        if v not in valid_categories:
            raise ValueError(f'merchant_category must be one of {valid_categories}')
        return v

class PredictionResponse(BaseModel):
    """Fraud prediction response"""
    transaction_id: str
    fraud_probability: float = Field(..., ge=0, le=1)
    risk_level: RiskLevel
    confidence: float = Field(..., ge=0, le=1)
    latency_ms: float
    model_version: str
    timestamp: float = Field(default_factory=lambda: time.time())

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: float
    version: str
    model_loaded: bool
    uptime_seconds: Optional[float] = None

class MetricsResponse(BaseModel):
    """Metrics response"""
    total_predictions: int
    average_latency_ms: float
    fraud_rate: float
    accuracy: Optional[float] = None
    model_version: str
    uptime_seconds: float
```

### **Step 3: Model Inference Service**

#### **3.1 Prediction Service**
```python
# src/inference/predictor.py
import joblib
import numpy as np
import pandas as pd
import asyncio
import logging
from typing import Dict, Any
import time
from pathlib import Path

from api.models import TransactionRequest, RiskLevel
from shared.config import Config

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
            
            if not model_path.exists() or not processor_path.exists():
                raise FileNotFoundError("Model files not found. Please train model first.")
            
            # Load model and processor
            self.model = joblib.load(model_path)
            self.data_processor = joblib.load(processor_path)
            
            self.load_time = time.time()
            logger.info(f"Model loaded successfully. Version: {self.model_version}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
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
            
            # Prepare features for prediction
            feature_columns = [
                'amount_log', 'hour', 'day_of_week', 'user_age', 'account_age_days',
                'merchant_category_encoded', 'is_weekend', 'is_night'
            ]
            
            X = processed_data[feature_columns].values
            
            # Make prediction
            fraud_probability = self.model.predict_proba(X)[0][1]
            confidence = max(self.model.predict_proba(X)[0]) - min(self.model.predict_proba(X)[0])
            
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
        if probability >= 0.7:
            return RiskLevel.HIGH
        elif probability >= 0.3:
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
```

### **Step 4: Docker Configuration**

#### **4.1 API Service Dockerfile**
```dockerfile
# docker/Dockerfile.api
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY data/ ./data/

# Create non-root user
RUN useradd --create-home --shell /bin/bash mluser
RUN chown -R mluser:mluser /app
USER mluser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["python", "src/api/main.py"]
```

#### **4.2 Docker Compose for Development**
```yaml
# docker-compose.yml
version: '3.8'

services:
  # PostgreSQL database
  postgres:
    image: postgres:13
    container_name: fraud-postgres
    environment:
      POSTGRES_DB: fraud_detection
      POSTGRES_USER: fraud_user
      POSTGRES_PASSWORD: fraud_password
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./scripts/init_db.sql:/docker-entrypoint-initdb.d/init_db.sql
    ports:
      - "5432:5432"
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U fraud_user -d fraud_detection"]
      interval: 10s
      timeout: 5s
      retries: 5

  # Redis for caching
  redis:
    image: redis:7-alpine
    container_name: fraud-redis
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 3

  # MLflow tracking server
  mlflow:
    image: python:3.9-slim
    container_name: fraud-mlflow
    ports:
      - "5000:5000"
    environment:
      - MLFLOW_BACKEND_STORE_URI=postgresql://fraud_user:fraud_password@postgres:5432/fraud_detection
      - MLFLOW_DEFAULT_ARTIFACT_ROOT=./mlruns
    volumes:
      - mlflow_data:/mlflow
    command: >
      bash -c "pip install mlflow psycopg2-binary &&
               mlflow server --host 0.0.0.0 --port 5000"
    depends_on:
      postgres:
        condition: service_healthy

  # Fraud detection API
  fraud-api:
    build:
      context: .
      dockerfile: docker/Dockerfile.api
    container_name: fraud-api
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://fraud_user:fraud_password@postgres:5432/fraud_detection
      - REDIS_URL=redis://redis:6379
      - MLFLOW_TRACKING_URI=http://mlflow:5000
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
      mlflow:
        condition: service_started
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  # Prometheus for metrics
  prometheus:
    image: prom/prometheus:latest
    container_name: fraud-prometheus
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

  # Grafana for visualization
  grafana:
    image: grafana/grafana:latest
    container_name: fraud-grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana/dashboards:/etc/grafana/provisioning/dashboards
      - ./monitoring/grafana/datasources:/etc/grafana/provisioning/datasources
    depends_on:
      - prometheus

volumes:
  postgres_data:
  redis_data:
  mlflow_data:
  prometheus_data:
  grafana_data:

networks:
  default:
    name: fraud-detection-network
```

This is a complete, production-grade fraud detection system. Would you like me to continue with the Kubernetes deployment configurations, monitoring setup, and the other two production projects?