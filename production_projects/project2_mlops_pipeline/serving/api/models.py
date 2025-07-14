"""
Pydantic Models for MLOps Serving API
Request/response models with comprehensive validation
"""

from pydantic import BaseModel, Field, validator
from typing import Dict, List, Any, Optional, Union
from enum import Enum
import time
import uuid

class PredictionRequest(BaseModel):
    """Single prediction request model"""
    
    features: Dict[str, Union[float, int, str, bool]] = Field(
        ...,
        description="Feature values for prediction",
        example={
            "feature_0": 1.5,
            "feature_1": -0.5,
            "feature_2": 2.1,
            "categorical_feature": "category_a"
        }
    )
    
    request_id: Optional[str] = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Optional request identifier"
    )
    
    model_version: Optional[str] = Field(
        None,
        description="Specific model version to use (optional)"
    )
    
    explain: bool = Field(
        False,
        description="Whether to include model explanation"
    )
    
    @validator('features')
    def validate_features(cls, v):
        """Validate feature dictionary"""
        if not v:
            raise ValueError("Features cannot be empty")
        
        # Check for reasonable number of features
        if len(v) > 1000:
            raise ValueError("Too many features (max 1000)")
        
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "features": {
                    "feature_0": 1.5,
                    "feature_1": -0.5,
                    "feature_2": 2.1,
                    "categorical_feature": "category_a"
                },
                "request_id": "req_123456",
                "explain": False
            }
        }

class PredictionResponse(BaseModel):
    """Single prediction response model"""
    
    request_id: str = Field(
        ...,
        description="Request identifier"
    )
    
    prediction: Union[float, int, str, List[float]] = Field(
        ...,
        description="Model prediction result"
    )
    
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Prediction confidence score"
    )
    
    model_version: str = Field(
        ...,
        description="Model version used for prediction"
    )
    
    latency_ms: float = Field(
        ...,
        description="Prediction latency in milliseconds"
    )
    
    timestamp: float = Field(
        default_factory=time.time,
        description="Response timestamp"
    )
    
    features_used: Optional[List[str]] = Field(
        None,
        description="List of features used in prediction"
    )
    
    explanation: Optional[Dict[str, Any]] = Field(
        None,
        description="Model explanation data (if requested)"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "request_id": "req_123456",
                "prediction": 0.85,
                "confidence": 0.92,
                "model_version": "v2.1.0",
                "latency_ms": 45.2,
                "timestamp": 1634567890.123,
                "features_used": ["feature_0", "feature_1", "feature_2"]
            }
        }

class BatchPredictionRequest(BaseModel):
    """Batch prediction request model"""
    
    samples: List[Dict[str, Union[float, int, str, bool]]] = Field(
        ...,
        min_items=1,
        max_items=1000,
        description="List of feature dictionaries for batch prediction"
    )
    
    batch_id: Optional[str] = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Batch identifier"
    )
    
    model_version: Optional[str] = Field(
        None,
        description="Specific model version to use"
    )
    
    parallel: bool = Field(
        True,
        description="Whether to process samples in parallel"
    )
    
    @validator('samples')
    def validate_samples(cls, v):
        """Validate batch samples"""
        if len(v) > 1000:
            raise ValueError("Batch size cannot exceed 1000 samples")
        
        # Check that all samples have the same feature keys
        if len(v) > 1:
            first_keys = set(v[0].keys())
            for i, sample in enumerate(v[1:], 1):
                if set(sample.keys()) != first_keys:
                    raise ValueError(f"Sample {i} has different features than sample 0")
        
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "samples": [
                    {"feature_0": 1.5, "feature_1": -0.5},
                    {"feature_0": 2.1, "feature_1": 0.3},
                    {"feature_0": -1.2, "feature_1": 1.8}
                ],
                "batch_id": "batch_123456",
                "parallel": True
            }
        }

class BatchPredictionResponse(BaseModel):
    """Batch prediction response model"""
    
    batch_id: str = Field(
        ...,
        description="Batch identifier"
    )
    
    predictions: List[PredictionResponse] = Field(
        ...,
        description="Individual prediction results"
    )
    
    total_samples: int = Field(
        ...,
        description="Total number of samples in batch"
    )
    
    successful_predictions: int = Field(
        ...,
        description="Number of successful predictions"
    )
    
    failed_predictions: int = Field(
        ...,
        description="Number of failed predictions"
    )
    
    total_latency_ms: float = Field(
        ...,
        description="Total batch processing time"
    )
    
    average_latency_ms: float = Field(
        ...,
        description="Average per-sample latency"
    )
    
    timestamp: float = Field(
        default_factory=time.time,
        description="Response timestamp"
    )
    
    errors: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="Details of any prediction errors"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "batch_id": "batch_123456",
                "predictions": [
                    {
                        "request_id": "req_1",
                        "prediction": 0.85,
                        "confidence": 0.92,
                        "model_version": "v2.1.0",
                        "latency_ms": 45.2,
                        "timestamp": 1634567890.123
                    }
                ],
                "total_samples": 3,
                "successful_predictions": 3,
                "failed_predictions": 0,
                "total_latency_ms": 135.6,
                "average_latency_ms": 45.2,
                "timestamp": 1634567890.123
            }
        }

class HealthResponse(BaseModel):
    """Health check response model"""
    
    status: str = Field(
        ...,
        description="Overall health status"
    )
    
    timestamp: float = Field(
        ...,
        description="Health check timestamp"
    )
    
    version: str = Field(
        ...,
        description="Service version"
    )
    
    model_loaded: bool = Field(
        ...,
        description="Whether ML model is loaded"
    )
    
    model_version: Optional[str] = Field(
        None,
        description="Current model version"
    )
    
    uptime_seconds: Optional[float] = Field(
        None,
        description="Service uptime in seconds"
    )
    
    dependencies: Optional[Dict[str, str]] = Field(
        None,
        description="Status of external dependencies"
    )
    
    system_info: Optional[Dict[str, Any]] = Field(
        None,
        description="System resource information"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "timestamp": 1634567890.123,
                "version": "2.0.0",
                "model_loaded": True,
                "model_version": "v2.1.0",
                "uptime_seconds": 3600.5,
                "dependencies": {
                    "mlflow": "healthy",
                    "database": "healthy",
                    "cache": "healthy"
                },
                "system_info": {
                    "memory_usage": "25%",
                    "cpu_usage": "15%"
                }
            }
        }

class MetricsResponse(BaseModel):
    """Detailed metrics response model"""
    
    # Request metrics
    total_predictions: int = Field(
        ...,
        description="Total number of predictions made"
    )
    
    predictions_per_second: float = Field(
        ...,
        description="Current predictions per second"
    )
    
    # Latency metrics
    average_latency_ms: float = Field(
        ...,
        description="Average prediction latency"
    )
    
    p95_latency_ms: float = Field(
        ...,
        description="95th percentile latency"
    )
    
    p99_latency_ms: float = Field(
        ...,
        description="99th percentile latency"
    )
    
    # Model performance
    model_accuracy: float = Field(
        ...,
        description="Current model accuracy"
    )
    
    model_version: str = Field(
        ...,
        description="Current model version"
    )
    
    # System metrics
    uptime_seconds: float = Field(
        ...,
        description="Service uptime"
    )
    
    error_rate: float = Field(
        ...,
        description="Error rate (errors per total requests)"
    )
    
    memory_usage_mb: float = Field(
        ...,
        description="Memory usage in MB"
    )
    
    cpu_usage_percent: float = Field(
        ...,
        description="CPU usage percentage"
    )
    
    # Additional metrics
    cache_hit_rate: Optional[float] = Field(
        None,
        description="Cache hit rate percentage"
    )
    
    queue_depth: Optional[int] = Field(
        None,
        description="Current request queue depth"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "total_predictions": 10000,
                "predictions_per_second": 50.5,
                "average_latency_ms": 45.2,
                "p95_latency_ms": 89.1,
                "p99_latency_ms": 156.7,
                "model_accuracy": 0.85,
                "model_version": "v2.1.0",
                "uptime_seconds": 7200.5,
                "error_rate": 0.01,
                "memory_usage_mb": 512.3,
                "cpu_usage_percent": 25.7
            }
        }

class ModelInfo(BaseModel):
    """Model information response model"""
    
    model_name: str = Field(
        ...,
        description="Model name"
    )
    
    model_version: str = Field(
        ...,
        description="Model version"
    )
    
    model_stage: str = Field(
        ...,
        description="Model stage (e.g., Production, Staging)"
    )
    
    loaded_at: Optional[float] = Field(
        None,
        description="Model load timestamp"
    )
    
    features: List[str] = Field(
        ...,
        description="List of expected feature names"
    )
    
    performance_metrics: Dict[str, float] = Field(
        ...,
        description="Model performance metrics"
    )
    
    model_size_mb: float = Field(
        ...,
        description="Model size in megabytes"
    )
    
    framework: Optional[str] = Field(
        None,
        description="ML framework used (e.g., sklearn, tensorflow)"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "model_name": "mlops_pipeline_model",
                "model_version": "v2.1.0",
                "model_stage": "Production",
                "loaded_at": 1634567890.123,
                "features": ["feature_0", "feature_1", "feature_2"],
                "performance_metrics": {
                    "accuracy": 0.85,
                    "f1_score": 0.83,
                    "precision": 0.87,
                    "recall": 0.79
                },
                "model_size_mb": 15.3,
                "framework": "sklearn"
            }
        }

class FeatureSchema(BaseModel):
    """Feature schema model for debugging"""
    
    feature_names: List[str] = Field(
        ...,
        description="List of feature names"
    )
    
    feature_types: Dict[str, str] = Field(
        ...,
        description="Feature types mapping"
    )
    
    required_features: List[str] = Field(
        ...,
        description="List of required features"
    )
    
    optional_features: List[str] = Field(
        ...,
        description="List of optional features"
    )
    
    example_input: Dict[str, Union[float, int, str, bool]] = Field(
        ...,
        description="Example valid input"
    )
    
    feature_ranges: Optional[Dict[str, Dict[str, float]]] = Field(
        None,
        description="Valid ranges for numerical features"
    )
    
    categorical_values: Optional[Dict[str, List[str]]] = Field(
        None,
        description="Valid values for categorical features"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "feature_names": ["feature_0", "feature_1", "category"],
                "feature_types": {
                    "feature_0": "float",
                    "feature_1": "float",
                    "category": "string"
                },
                "required_features": ["feature_0", "feature_1"],
                "optional_features": ["category"],
                "example_input": {
                    "feature_0": 1.5,
                    "feature_1": -0.5,
                    "category": "A"
                },
                "feature_ranges": {
                    "feature_0": {"min": -5.0, "max": 5.0},
                    "feature_1": {"min": -3.0, "max": 3.0}
                },
                "categorical_values": {
                    "category": ["A", "B", "C"]
                }
            }
        }

class ErrorResponse(BaseModel):
    """Error response model"""
    
    error: str = Field(
        ...,
        description="Error type"
    )
    
    message: str = Field(
        ...,
        description="Human-readable error message"
    )
    
    details: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional error details"
    )
    
    request_id: Optional[str] = Field(
        None,
        description="Request identifier"
    )
    
    timestamp: float = Field(
        default_factory=time.time,
        description="Error timestamp"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "error": "validation_error",
                "message": "Invalid feature values provided",
                "details": {
                    "invalid_features": ["feature_0"],
                    "expected_type": "float",
                    "received_type": "string"
                },
                "request_id": "req_123456",
                "timestamp": 1634567890.123
            }
        }