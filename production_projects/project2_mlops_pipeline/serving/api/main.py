"""
MLOps Model Serving API
Production-ready FastAPI service for model inference
"""

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import Dict, Any, List
import uvicorn
import mlflow
import mlflow.sklearn
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from prometheus_client.exposition import CONTENT_TYPE_LATEST
import joblib
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from serving.api.models import (
    PredictionRequest, PredictionResponse, BatchPredictionRequest, 
    BatchPredictionResponse, HealthResponse, MetricsResponse
)
from serving.api.predictor import ModelPredictor
from serving.api.middleware import MetricsMiddleware, LoggingMiddleware
from monitoring.model_monitor import ModelMonitor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Prometheus metrics
REQUEST_COUNT = Counter('api_requests_total', 'Total API requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('api_request_duration_seconds', 'API request duration')
PREDICTION_COUNT = Counter('predictions_total', 'Total predictions made', ['model_version'])
PREDICTION_LATENCY = Histogram('prediction_latency_seconds', 'Prediction latency')
MODEL_ACCURACY = Gauge('model_accuracy', 'Current model accuracy')
ACTIVE_CONNECTIONS = Gauge('active_connections', 'Number of active connections')

# Global variables
predictor: ModelPredictor = None
monitor: ModelMonitor = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    global predictor, monitor
    
    logger.info("Starting MLOps API service...")
    
    try:
        # Initialize model predictor
        predictor = ModelPredictor()
        await predictor.load_model()
        
        # Initialize model monitor
        monitor = ModelMonitor()
        await monitor.initialize()
        
        logger.info("✅ Service initialization completed")
        yield
        
    except Exception as e:
        logger.error(f"Failed to initialize service: {e}")
        raise
    
    finally:
        # Shutdown
        logger.info("Shutting down MLOps API service...")
        
        if predictor:
            await predictor.cleanup()
        
        if monitor:
            await monitor.cleanup()
        
        logger.info("✅ Service shutdown completed")

# Create FastAPI app
app = FastAPI(
    title="MLOps Model Serving API",
    description="Production-ready ML model serving with monitoring and observability",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(MetricsMiddleware)
app.add_middleware(LoggingMiddleware)

# Dependency injection
async def get_predictor() -> ModelPredictor:
    """Get model predictor instance"""
    if predictor is None or not predictor.is_ready():
        raise HTTPException(status_code=503, detail="Model not ready")
    return predictor

async def get_monitor() -> ModelMonitor:
    """Get model monitor instance"""
    if monitor is None:
        raise HTTPException(status_code=503, detail="Monitor not ready")
    return monitor

# Health check endpoints
@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Comprehensive health check
    
    Returns service health status including model readiness and dependencies
    """
    start_time = time.time()
    
    try:
        # Check model predictor
        model_ready = predictor is not None and predictor.is_ready()
        model_info = predictor.get_model_info() if model_ready else {}
        
        # Check monitor
        monitor_ready = monitor is not None
        
        # Check dependencies (mock for now)
        dependencies = {
            "mlflow": "healthy",
            "database": "healthy",
            "cache": "healthy"
        }
        
        # Determine overall status
        overall_status = "healthy" if model_ready and monitor_ready else "unhealthy"
        
        response = HealthResponse(
            status=overall_status,
            timestamp=time.time(),
            version="2.0.0",
            model_loaded=model_ready,
            model_version=model_info.get('version', 'unknown'),
            uptime_seconds=time.time() - start_time,
            dependencies=dependencies,
            system_info={
                "memory_usage": "25%",  # Mock system info
                "cpu_usage": "15%",
                "disk_usage": "60%"
            }
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

@app.get("/ready", tags=["Health"])
async def readiness_check():
    """
    Kubernetes readiness probe
    
    Returns 200 if service is ready to accept traffic
    """
    if predictor is None or not predictor.is_ready():
        raise HTTPException(status_code=503, detail="Service not ready")
    
    return {"status": "ready", "timestamp": time.time()}

@app.get("/live", tags=["Health"])
async def liveness_check():
    """
    Kubernetes liveness probe
    
    Returns 200 if service is alive
    """
    return {"status": "alive", "timestamp": time.time()}

# Prediction endpoints
@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(
    request: PredictionRequest,
    background_tasks: BackgroundTasks,
    predictor: ModelPredictor = Depends(get_predictor),
    monitor: ModelMonitor = Depends(get_monitor)
):
    """
    Make single prediction
    
    Performs ML inference on a single data sample with monitoring and logging
    """
    start_time = time.time()
    request_id = f"pred_{int(time.time() * 1000)}"
    
    try:
        # Make prediction
        with REQUEST_DURATION.time():
            prediction = await predictor.predict(request.features)
        
        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000
        
        # Create response
        response = PredictionResponse(
            request_id=request_id,
            prediction=prediction['prediction'],
            confidence=prediction['confidence'],
            model_version=prediction['model_version'],
            latency_ms=latency_ms,
            timestamp=time.time(),
            features_used=list(request.features.keys())
        )
        
        # Update metrics
        PREDICTION_COUNT.labels(model_version=prediction['model_version']).inc()
        PREDICTION_LATENCY.observe(latency_ms / 1000)
        
        # Background monitoring
        background_tasks.add_task(
            monitor.log_prediction,
            request=request,
            response=response,
            latency_ms=latency_ms
        )
        
        logger.info(f"Prediction completed: {request_id} in {latency_ms:.2f}ms")
        return response
        
    except Exception as e:
        logger.error(f"Prediction failed for {request_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch(
    request: BatchPredictionRequest,
    background_tasks: BackgroundTasks,
    predictor: ModelPredictor = Depends(get_predictor),
    monitor: ModelMonitor = Depends(get_monitor)
):
    """
    Make batch predictions
    
    Performs ML inference on multiple data samples efficiently
    """
    start_time = time.time()
    batch_id = f"batch_{int(time.time() * 1000)}"
    
    try:
        logger.info(f"Processing batch prediction: {batch_id} with {len(request.samples)} samples")
        
        # Process batch
        predictions = await predictor.predict_batch(request.samples)
        
        # Calculate metrics
        total_latency_ms = (time.time() - start_time) * 1000
        avg_latency_ms = total_latency_ms / len(request.samples)
        
        # Create response
        response = BatchPredictionResponse(
            batch_id=batch_id,
            predictions=predictions,
            total_samples=len(request.samples),
            successful_predictions=len(predictions),
            failed_predictions=0,
            total_latency_ms=total_latency_ms,
            average_latency_ms=avg_latency_ms,
            timestamp=time.time()
        )
        
        # Update metrics
        for pred in predictions:
            PREDICTION_COUNT.labels(model_version=pred.model_version).inc()
        
        PREDICTION_LATENCY.observe(avg_latency_ms / 1000)
        
        # Background monitoring
        background_tasks.add_task(
            monitor.log_batch_prediction,
            request=request,
            response=response
        )
        
        logger.info(f"Batch prediction completed: {batch_id} in {total_latency_ms:.2f}ms")
        return response
        
    except Exception as e:
        logger.error(f"Batch prediction failed for {batch_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

# Monitoring endpoints
@app.get("/metrics", tags=["Monitoring"])
async def get_metrics():
    """
    Prometheus metrics endpoint
    
    Returns metrics in Prometheus format for monitoring and alerting
    """
    try:
        # Generate Prometheus metrics
        metrics_data = generate_latest()
        return Response(content=metrics_data, media_type=CONTENT_TYPE_LATEST)
        
    except Exception as e:
        logger.error(f"Failed to generate metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate metrics")

@app.get("/metrics/detailed", response_model=MetricsResponse, tags=["Monitoring"])
async def get_detailed_metrics(
    monitor: ModelMonitor = Depends(get_monitor)
):
    """
    Detailed application metrics
    
    Returns comprehensive metrics for monitoring and debugging
    """
    try:
        # Get current metrics from monitor
        metrics = await monitor.get_current_metrics()
        
        # Add system metrics
        response = MetricsResponse(
            total_predictions=metrics.get('total_predictions', 0),
            predictions_per_second=metrics.get('predictions_per_second', 0.0),
            average_latency_ms=metrics.get('average_latency_ms', 0.0),
            p95_latency_ms=metrics.get('p95_latency_ms', 0.0),
            p99_latency_ms=metrics.get('p99_latency_ms', 0.0),
            model_accuracy=metrics.get('model_accuracy', 0.0),
            model_version=predictor.get_model_info().get('version', 'unknown'),
            uptime_seconds=metrics.get('uptime_seconds', 0.0),
            error_rate=metrics.get('error_rate', 0.0),
            memory_usage_mb=metrics.get('memory_usage_mb', 0.0),
            cpu_usage_percent=metrics.get('cpu_usage_percent', 0.0)
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Failed to get detailed metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to get detailed metrics")

# Model management endpoints
@app.get("/model/info", tags=["Model"])
async def get_model_info(predictor: ModelPredictor = Depends(get_predictor)):
    """
    Get current model information
    
    Returns metadata about the currently loaded model
    """
    try:
        model_info = predictor.get_model_info()
        return {
            "model_name": model_info.get('name', 'unknown'),
            "model_version": model_info.get('version', 'unknown'),
            "model_stage": model_info.get('stage', 'unknown'),
            "loaded_at": model_info.get('loaded_at'),
            "features": model_info.get('features', []),
            "performance_metrics": model_info.get('metrics', {}),
            "model_size_mb": model_info.get('size_mb', 0)
        }
        
    except Exception as e:
        logger.error(f"Failed to get model info: {e}")
        raise HTTPException(status_code=500, detail="Failed to get model info")

@app.post("/model/reload", tags=["Model"])
async def reload_model(
    background_tasks: BackgroundTasks,
    predictor: ModelPredictor = Depends(get_predictor)
):
    """
    Reload the ML model
    
    Reloads the model from the registry (admin endpoint)
    """
    try:
        logger.info("Reloading model...")
        
        # Reload model in background
        background_tasks.add_task(predictor.reload_model)
        
        return {
            "status": "model_reload_initiated",
            "message": "Model reload started in background",
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Failed to reload model: {e}")
        raise HTTPException(status_code=500, detail="Failed to reload model")

# Debug endpoints (only in development)
@app.get("/debug/features", tags=["Debug"])
async def debug_features(predictor: ModelPredictor = Depends(get_predictor)):
    """
    Get expected feature schema for debugging
    """
    try:
        feature_schema = predictor.get_feature_schema()
        return {
            "feature_names": feature_schema.get('names', []),
            "feature_types": feature_schema.get('types', {}),
            "required_features": feature_schema.get('required', []),
            "optional_features": feature_schema.get('optional', []),
            "example_input": feature_schema.get('example', {})
        }
        
    except Exception as e:
        logger.error(f"Failed to get feature schema: {e}")
        raise HTTPException(status_code=500, detail="Failed to get feature schema")

# Error handlers
@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    """Handle validation errors"""
    logger.warning(f"Validation error: {exc}")
    return JSONResponse(
        status_code=400,
        content={
            "error": "validation_error",
            "message": str(exc),
            "timestamp": time.time()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_server_error",
            "message": "An unexpected error occurred",
            "timestamp": time.time()
        }
    )

# Startup event for metrics initialization
@app.on_event("startup")
async def startup_event():
    """Initialize metrics on startup"""
    MODEL_ACCURACY.set(0.85)  # Set initial model accuracy
    logger.info("Metrics initialized")

def create_app() -> FastAPI:
    """Factory function to create FastAPI app"""
    return app

if __name__ == "__main__":
    import os
    
    # Configuration
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    workers = int(os.getenv("API_WORKERS", "1"))
    reload = os.getenv("DEBUG", "false").lower() == "true"
    
    logger.info(f"Starting MLOps API server on {host}:{port}")
    
    # Run with uvicorn
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        workers=workers,
        reload=reload,
        access_log=True,
        log_level="info"
    )