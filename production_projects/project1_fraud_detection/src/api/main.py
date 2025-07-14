#!/usr/bin/env python3
"""
Fraud Detection API - Main Application
Production-grade FastAPI service for real-time fraud detection
"""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import asyncio
import logging
import time
import traceback
from typing import Dict, List
import uvicorn
import sys
from pathlib import Path
import os

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from api.models import TransactionRequest, PredictionResponse, HealthResponse
from api.middleware import MetricsMiddleware, RequestLoggingMiddleware
from inference.predictor import FraudPredictor
from monitoring.metrics_collector import MetricsCollector
from shared.config import config
from shared.logging import setup_application_logging, get_performance_logger, get_request_logger

# Setup logging
logger = setup_application_logging()
performance_logger = get_performance_logger()
request_logger = get_request_logger()

# Initialize FastAPI app
app = FastAPI(
    title="Fraud Detection API",
    description="Real-time fraud detection service with MLOps best practices",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add custom middleware
app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(MetricsMiddleware)

# Global variables for services
predictor: FraudPredictor = None
metrics_collector: MetricsCollector = None
startup_time = time.time()

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred",
            "request_id": getattr(request.state, "request_id", None)
        }
    )

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    global predictor, metrics_collector
    
    try:
        logger.info("🚀 Starting Fraud Detection API...")
        logger.info(f"Environment: {config.ENVIRONMENT}")
        logger.info(f"Debug mode: {config.DEBUG}")
        
        # Validate configuration
        config.validate_config()
        logger.info("✅ Configuration validated")
        
        # Initialize predictor
        logger.info("🧠 Initializing fraud predictor...")
        predictor = FraudPredictor()
        await predictor.load_model()
        logger.info("✅ Fraud predictor initialized")
        
        # Initialize metrics collector
        if config.METRICS_ENABLED:
            logger.info("📊 Initializing metrics collector...")
            metrics_collector = MetricsCollector()
            await metrics_collector.initialize()
            logger.info("✅ Metrics collector initialized")
        
        logger.info("🎉 Fraud Detection API started successfully")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize service: {e}")
        logger.error(traceback.format_exc())
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("🛑 Shutting down Fraud Detection API...")
    
    try:
        if metrics_collector:
            await metrics_collector.close()
            logger.info("✅ Metrics collector closed")
        
        if predictor:
            await predictor.cleanup()
            logger.info("✅ Predictor cleaned up")
        
        logger.info("👋 Fraud Detection API shutdown complete")
        
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")

# Health check endpoint
@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint for load balancers and monitoring systems
    
    Returns comprehensive health status including:
    - Service availability
    - Model loading status
    - Dependencies health
    - System metrics
    """
    try:
        current_time = time.time()
        uptime_seconds = current_time - startup_time
        
        # Check if model is loaded
        model_loaded = predictor is not None and predictor.is_loaded()
        
        # Check dependencies
        dependencies_healthy = True
        dependency_status = {}
        
        if metrics_collector:
            try:
                await metrics_collector.health_check()
                dependency_status["metrics_collector"] = "healthy"
            except Exception as e:
                dependency_status["metrics_collector"] = f"unhealthy: {str(e)}"
                dependencies_healthy = False
        
        # Determine overall status
        if model_loaded and dependencies_healthy:
            status = "healthy"
            status_code = 200
        else:
            status = "unhealthy"
            status_code = 503
        
        response = HealthResponse(
            status=status,
            timestamp=current_time,
            version="1.0.0",
            model_loaded=model_loaded,
            uptime_seconds=uptime_seconds,
            dependencies=dependency_status
        )
        
        if status_code != 200:
            raise HTTPException(status_code=status_code, detail=response.dict())
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Health check failed")

# Readiness probe
@app.get("/ready", tags=["Health"])
async def readiness_check():
    """
    Readiness probe for Kubernetes
    
    Indicates if the service is ready to receive traffic
    """
    if predictor is None or not predictor.is_loaded():
        raise HTTPException(status_code=503, detail="Service not ready")
    
    return {"status": "ready", "timestamp": time.time()}

# Liveness probe
@app.get("/live", tags=["Health"])
async def liveness_check():
    """
    Liveness probe for Kubernetes
    
    Indicates if the service is alive (but may not be ready)
    """
    return {"status": "alive", "timestamp": time.time()}

# Main prediction endpoint
@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_fraud(
    request: TransactionRequest,
    background_tasks: BackgroundTasks
) -> PredictionResponse:
    """
    Predict fraud probability for a transaction
    
    This endpoint performs real-time fraud detection with:
    - Input validation
    - Feature engineering
    - Model inference
    - Response formatting
    - Async metrics logging
    
    **Business Logic:**
    - Low risk: probability < 0.3
    - Medium risk: 0.3 ≤ probability < 0.7  
    - High risk: probability ≥ 0.7
    """
    start_time = time.time()
    request_id = getattr(request, 'transaction_id', 'unknown')
    
    try:
        # Validate service readiness
        if predictor is None or not predictor.is_loaded():
            raise HTTPException(
                status_code=503, 
                detail="Model not ready. Please try again later."
            )
        
        # Validate request data
        if request.amount <= 0:
            raise HTTPException(
                status_code=400, 
                detail="Transaction amount must be positive"
            )
        
        if not (0 <= request.hour <= 23):
            raise HTTPException(
                status_code=400,
                detail="Hour must be between 0 and 23"
            )
        
        # Make prediction
        logger.info(f"Processing prediction request: {request_id}")
        prediction_result = await predictor.predict(request)
        
        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000
        
        # Create response
        response = PredictionResponse(
            transaction_id=request.transaction_id,
            fraud_probability=prediction_result["fraud_probability"],
            risk_level=prediction_result["risk_level"],
            confidence=prediction_result["confidence"],
            latency_ms=latency_ms,
            model_version=prediction_result["model_version"],
            timestamp=time.time()
        )
        
        # Log prediction metrics asynchronously
        background_tasks.add_task(
            log_prediction_metrics,
            request,
            prediction_result,
            latency_ms,
            request_id
        )
        
        # Log performance
        performance_logger.log_prediction_performance(
            duration_ms=latency_ms,
            model_version=prediction_result["model_version"],
            request_id=request_id
        )
        
        logger.info(
            f"Prediction completed - ID: {request_id}, "
            f"Probability: {response.fraud_probability:.4f}, "
            f"Risk: {response.risk_level}, "
            f"Latency: {latency_ms:.2f}ms"
        )
        
        return response
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Prediction failed for request {request_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail=f"Prediction failed: {str(e)}"
        )

async def log_prediction_metrics(
    request: TransactionRequest,
    prediction_result: Dict,
    latency_ms: float,
    request_id: str
):
    """Log prediction metrics asynchronously"""
    try:
        if metrics_collector:
            await metrics_collector.log_prediction(
                request=request,
                prediction=prediction_result,
                latency=latency_ms,
                request_id=request_id
            )
    except Exception as e:
        logger.error(f"Failed to log metrics for request {request_id}: {e}")

# Batch prediction endpoint
@app.post("/predict/batch", tags=["Prediction"])
async def predict_batch(
    requests: List[TransactionRequest],
    background_tasks: BackgroundTasks
) -> List[PredictionResponse]:
    """
    Process multiple transactions in batch
    
    Useful for:
    - Bulk processing
    - Historical analysis
    - Model validation
    
    Limited to prevent resource exhaustion.
    """
    start_time = time.time()
    
    # Validate batch size
    if len(requests) > config.MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"Batch size too large. Maximum: {config.MAX_BATCH_SIZE}"
        )
    
    if not requests:
        raise HTTPException(
            status_code=400,
            detail="Request list cannot be empty"
        )
    
    try:
        logger.info(f"Processing batch prediction with {len(requests)} transactions")
        
        # Process all requests
        responses = []
        for request in requests:
            response = await predict_fraud(request, background_tasks)
            responses.append(response)
        
        batch_latency_ms = (time.time() - start_time) * 1000
        
        logger.info(
            f"Batch prediction completed - "
            f"Count: {len(responses)}, "
            f"Total latency: {batch_latency_ms:.2f}ms, "
            f"Avg latency: {batch_latency_ms / len(responses):.2f}ms"
        )
        
        return responses
        
    except Exception as e:
        logger.error(f"Batch prediction failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Batch prediction failed: {str(e)}"
        )

# Model information endpoint
@app.get("/model/info", tags=["Model"])
async def get_model_info():
    """
    Get information about the currently loaded model
    
    Returns:
    - Model version
    - Loading timestamp
    - Prediction count
    - Model metadata
    """
    try:
        if predictor is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        model_info = predictor.get_model_info()
        return {
            "model_info": model_info,
            "service_uptime_seconds": time.time() - startup_time,
            "environment": config.ENVIRONMENT
        }
        
    except Exception as e:
        logger.error(f"Failed to get model info: {e}")
        raise HTTPException(status_code=500, detail="Failed to get model info")

# Metrics endpoint
@app.get("/metrics", tags=["Monitoring"])
async def get_metrics():
    """
    Get service metrics for monitoring
    
    Returns current operational metrics including:
    - Request counts
    - Latency statistics
    - Error rates
    - Model performance
    """
    try:
        if metrics_collector is None:
            return {"message": "Metrics collection disabled"}
        
        metrics = await metrics_collector.get_current_metrics()
        return {
            "metrics": metrics,
            "timestamp": time.time(),
            "uptime_seconds": time.time() - startup_time
        }
        
    except Exception as e:
        logger.error(f"Failed to get metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to get metrics")

# Configuration endpoint (for debugging)
@app.get("/config", tags=["Debug"])
async def get_config():
    """
    Get current configuration (non-sensitive data only)
    
    Useful for debugging and verification
    """
    if not config.DEBUG:
        raise HTTPException(status_code=404, detail="Endpoint not available")
    
    return {
        "environment": config.get_environment_info(),
        "model_config": config.get_model_config(),
        "features": {
            "metrics_enabled": config.METRICS_ENABLED,
            "feature_store_enabled": config.FEATURE_STORE_ENABLED,
            "alert_enabled": config.ALERT_ENABLED
        }
    }

def create_app() -> FastAPI:
    """Application factory for testing"""
    return app

if __name__ == "__main__":
    # Run with uvicorn
    uvicorn.run(
        "main:app",
        host=config.API_HOST,
        port=config.API_PORT,
        workers=config.API_WORKERS,
        reload=config.DEBUG,
        log_level=config.LOG_LEVEL.lower(),
        access_log=True
    )