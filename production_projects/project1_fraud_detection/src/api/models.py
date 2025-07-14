"""
Pydantic Models for Fraud Detection API
Data validation and serialization models
"""

from pydantic import BaseModel, Field, validator, root_validator
from typing import Optional, Dict, Any, List
from enum import Enum
import uuid
import time
import re

class RiskLevel(str, Enum):
    """Risk level enumeration"""
    LOW = "low"
    MEDIUM = "medium" 
    HIGH = "high"

class TransactionRequest(BaseModel):
    """
    Transaction data for fraud prediction
    
    Represents a financial transaction with all relevant features
    for fraud detection analysis.
    """
    
    # Transaction identifiers
    transaction_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique transaction identifier",
        example="txn_1234567890"
    )
    
    # Core transaction data
    amount: float = Field(
        ...,
        gt=0,
        le=50000,  # Reasonable upper limit
        description="Transaction amount in USD",
        example=150.75
    )
    
    merchant: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Merchant name",
        example="Amazon.com"
    )
    
    merchant_category: str = Field(
        ...,
        description="Merchant category code",
        example="online"
    )
    
    # Time-related features
    hour: int = Field(
        ...,
        ge=0,
        le=23,
        description="Hour of transaction (0-23)",
        example=14
    )
    
    day_of_week: int = Field(
        ...,
        ge=0,
        le=6,
        description="Day of week (0=Monday, 6=Sunday)",
        example=2
    )
    
    # User features
    user_age: int = Field(
        ...,
        ge=18,
        le=100,
        description="User age in years",
        example=35
    )
    
    account_age_days: int = Field(
        ...,
        ge=0,
        le=7300,  # ~20 years max
        description="Account age in days",
        example=456
    )
    
    # Optional identifiers
    user_id: Optional[str] = Field(
        None,
        description="User identifier",
        example="user_abc123"
    )
    
    # Additional context (optional)
    device_id: Optional[str] = Field(
        None,
        description="Device identifier",
        example="device_xyz789"
    )
    
    ip_address: Optional[str] = Field(
        None,
        description="IP address",
        example="192.168.1.1"
    )
    
    currency: str = Field(
        default="USD",
        description="Transaction currency",
        example="USD"
    )
    
    @validator('merchant_category')
    def validate_merchant_category(cls, v):
        """Validate merchant category"""
        valid_categories = ['grocery', 'gas', 'restaurant', 'retail', 'online', 'other']
        if v.lower() not in valid_categories:
            raise ValueError(f'merchant_category must be one of {valid_categories}')
        return v.lower()
    
    @validator('merchant')
    def validate_merchant(cls, v):
        """Validate merchant name"""
        # Remove excessive whitespace and validate length
        cleaned = re.sub(r'\s+', ' ', v.strip())
        if len(cleaned) < 1:
            raise ValueError('merchant name cannot be empty')
        return cleaned
    
    @validator('currency')
    def validate_currency(cls, v):
        """Validate currency code"""
        valid_currencies = ['USD', 'EUR', 'GBP', 'CAD', 'AUD']
        if v.upper() not in valid_currencies:
            raise ValueError(f'currency must be one of {valid_currencies}')
        return v.upper()
    
    @validator('ip_address')
    def validate_ip_address(cls, v):
        """Basic IP address validation"""
        if v is None:
            return v
        
        # Simple IPv4 validation
        import ipaddress
        try:
            ipaddress.ip_address(v)
            return v
        except ValueError:
            raise ValueError('Invalid IP address format')
    
    @root_validator
    def validate_transaction_logic(cls, values):
        """Validate business logic constraints"""
        amount = values.get('amount')
        merchant_category = values.get('merchant_category')
        hour = values.get('hour')
        
        # Business rule validations
        if amount and merchant_category:
            # Very high amounts for certain categories might be suspicious
            if merchant_category in ['gas', 'grocery'] and amount > 1000:
                pass  # Log warning but don't reject
        
        return values
    
    class Config:
        """Pydantic config"""
        schema_extra = {
            "example": {
                "transaction_id": "txn_1234567890",
                "amount": 150.75,
                "merchant": "Amazon.com",
                "merchant_category": "online",
                "hour": 14,
                "day_of_week": 2,
                "user_age": 35,
                "account_age_days": 456,
                "user_id": "user_abc123",
                "currency": "USD"
            }
        }

class PredictionResponse(BaseModel):
    """
    Fraud prediction response
    
    Contains the fraud detection results and metadata
    """
    
    transaction_id: str = Field(
        ...,
        description="Transaction identifier from request"
    )
    
    fraud_probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Probability of fraud (0.0 = not fraud, 1.0 = fraud)"
    )
    
    risk_level: RiskLevel = Field(
        ...,
        description="Risk level categorization"
    )
    
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Model confidence in prediction"
    )
    
    latency_ms: float = Field(
        ...,
        description="Prediction latency in milliseconds"
    )
    
    model_version: str = Field(
        ...,
        description="Version of model used for prediction"
    )
    
    timestamp: float = Field(
        default_factory=time.time,
        description="Prediction timestamp (Unix epoch)"
    )
    
    # Additional metadata
    features_used: Optional[List[str]] = Field(
        None,
        description="List of features used in prediction"
    )
    
    explanation: Optional[Dict[str, Any]] = Field(
        None,
        description="Model explanation/interpretability data"
    )
    
    class Config:
        """Pydantic config"""
        schema_extra = {
            "example": {
                "transaction_id": "txn_1234567890",
                "fraud_probability": 0.15,
                "risk_level": "low",
                "confidence": 0.87,
                "latency_ms": 45.2,
                "model_version": "1.0.0",
                "timestamp": 1634567890.123
            }
        }

class HealthResponse(BaseModel):
    """
    Health check response
    
    Provides comprehensive service health information
    """
    
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
        description="Whether ML model is loaded and ready"
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
        """Pydantic config"""
        schema_extra = {
            "example": {
                "status": "healthy",
                "timestamp": 1634567890.123,
                "version": "1.0.0",
                "model_loaded": True,
                "uptime_seconds": 3600.5,
                "dependencies": {
                    "database": "healthy",
                    "redis": "healthy",
                    "mlflow": "healthy"
                }
            }
        }

class MetricsResponse(BaseModel):
    """
    Metrics response for monitoring
    
    Operational metrics for system monitoring and alerting
    """
    
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
    fraud_rate: float = Field(
        ...,
        description="Percentage of transactions flagged as fraud"
    )
    
    high_risk_rate: float = Field(
        ...,
        description="Percentage of high-risk transactions"
    )
    
    # System metrics
    model_version: str = Field(
        ...,
        description="Current model version"
    )
    
    uptime_seconds: float = Field(
        ...,
        description="Service uptime"
    )
    
    # Error metrics
    error_rate: float = Field(
        ...,
        description="Error rate (errors per total requests)"
    )
    
    last_error: Optional[str] = Field(
        None,
        description="Last error message"
    )
    
    class Config:
        """Pydantic config"""
        schema_extra = {
            "example": {
                "total_predictions": 10000,
                "predictions_per_second": 50.5,
                "average_latency_ms": 45.2,
                "p95_latency_ms": 89.1,
                "p99_latency_ms": 156.7,
                "fraud_rate": 5.2,
                "high_risk_rate": 2.1,
                "model_version": "1.0.0",
                "uptime_seconds": 7200.5,
                "error_rate": 0.01
            }
        }

class BatchPredictionRequest(BaseModel):
    """
    Batch prediction request
    
    Container for multiple transaction predictions
    """
    
    transactions: List[TransactionRequest] = Field(
        ...,
        min_items=1,
        max_items=1000,  # Prevent abuse
        description="List of transactions to predict"
    )
    
    batch_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Batch identifier"
    )
    
    @validator('transactions')
    def validate_batch_size(cls, v):
        """Validate batch size limits"""
        if len(v) > 1000:
            raise ValueError('Batch size cannot exceed 1000 transactions')
        return v

class BatchPredictionResponse(BaseModel):
    """
    Batch prediction response
    
    Results for batch processing
    """
    
    batch_id: str = Field(
        ...,
        description="Batch identifier"
    )
    
    predictions: List[PredictionResponse] = Field(
        ...,
        description="Individual prediction results"
    )
    
    batch_summary: Dict[str, Any] = Field(
        ...,
        description="Batch processing summary statistics"
    )
    
    total_latency_ms: float = Field(
        ...,
        description="Total batch processing time"
    )

class ErrorResponse(BaseModel):
    """
    Error response model
    
    Standardized error format
    """
    
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
        description="Request identifier for tracking"
    )
    
    timestamp: float = Field(
        default_factory=time.time,
        description="Error timestamp"
    )
    
    class Config:
        """Pydantic config"""
        schema_extra = {
            "example": {
                "error": "validation_error",
                "message": "Transaction amount must be positive",
                "details": {
                    "field": "amount",
                    "value": -100.0
                },
                "request_id": "req_1234567890",
                "timestamp": 1634567890.123
            }
        }