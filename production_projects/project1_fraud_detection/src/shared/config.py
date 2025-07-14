"""
Configuration Management for Fraud Detection System
Centralized configuration with environment-specific settings
"""

import os
from pathlib import Path
from typing import Dict, Any
import logging

class Config:
    """Application configuration settings"""
    
    # Environment
    ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
    DEBUG = os.getenv("DEBUG", "false").lower() == "true"
    
    # API Configuration
    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", "8000"))
    API_WORKERS = int(os.getenv("API_WORKERS", "1"))
    
    # Database Configuration
    DATABASE_URL = os.getenv(
        "DATABASE_URL", 
        "postgresql://fraud_user:fraud_password@localhost:5432/fraud_detection"
    )
    
    # Redis Configuration
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
    REDIS_TTL = int(os.getenv("REDIS_TTL", "3600"))  # 1 hour default
    
    # MLflow Configuration
    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "fraud_detection")
    
    # Model Configuration
    MODEL_NAME = os.getenv("MODEL_NAME", "fraud_detection_production")
    MODEL_VERSION = os.getenv("MODEL_VERSION", "latest")
    MODEL_PATH = os.getenv("MODEL_PATH", "data/models/")
    
    # Feature Store Configuration
    FEATURE_STORE_ENABLED = os.getenv("FEATURE_STORE_ENABLED", "true").lower() == "true"
    FEATURE_TTL_SECONDS = int(os.getenv("FEATURE_TTL_SECONDS", "300"))  # 5 minutes
    
    # Monitoring Configuration
    METRICS_ENABLED = os.getenv("METRICS_ENABLED", "true").lower() == "true"
    PROMETHEUS_PORT = int(os.getenv("PROMETHEUS_PORT", "8080"))
    
    # Logging Configuration
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT = os.getenv(
        "LOG_FORMAT", 
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Security Configuration
    SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-here")
    JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
    JWT_EXPIRATION_HOURS = int(os.getenv("JWT_EXPIRATION_HOURS", "24"))
    
    # Performance Configuration
    MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT_REQUESTS", "100"))
    REQUEST_TIMEOUT_SECONDS = int(os.getenv("REQUEST_TIMEOUT_SECONDS", "30"))
    
    # Business Rules
    DEFAULT_FRAUD_THRESHOLD = float(os.getenv("DEFAULT_FRAUD_THRESHOLD", "0.5"))
    HIGH_RISK_THRESHOLD = float(os.getenv("HIGH_RISK_THRESHOLD", "0.7"))
    LOW_RISK_THRESHOLD = float(os.getenv("LOW_RISK_THRESHOLD", "0.3"))
    
    # Alert Configuration
    ALERT_ENABLED = os.getenv("ALERT_ENABLED", "true").lower() == "true"
    SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL", "")
    EMAIL_SMTP_HOST = os.getenv("EMAIL_SMTP_HOST", "localhost")
    EMAIL_SMTP_PORT = int(os.getenv("EMAIL_SMTP_PORT", "587"))
    
    # Data Configuration
    MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", "1000"))
    DATA_RETENTION_DAYS = int(os.getenv("DATA_RETENTION_DAYS", "90"))
    
    @classmethod
    def get_database_config(cls) -> Dict[str, Any]:
        """Get database configuration"""
        return {
            "url": cls.DATABASE_URL,
            "pool_size": 10,
            "max_overflow": 20,
            "pool_timeout": 30,
            "pool_recycle": 3600
        }
    
    @classmethod
    def get_redis_config(cls) -> Dict[str, Any]:
        """Get Redis configuration"""
        return {
            "url": cls.REDIS_URL,
            "decode_responses": True,
            "socket_timeout": 5,
            "socket_connect_timeout": 5,
            "retry_on_timeout": True
        }
    
    @classmethod
    def get_model_config(cls) -> Dict[str, Any]:
        """Get model configuration"""
        return {
            "name": cls.MODEL_NAME,
            "version": cls.MODEL_VERSION,
            "path": cls.MODEL_PATH,
            "fraud_threshold": cls.DEFAULT_FRAUD_THRESHOLD,
            "high_risk_threshold": cls.HIGH_RISK_THRESHOLD,
            "low_risk_threshold": cls.LOW_RISK_THRESHOLD
        }
    
    @classmethod
    def validate_config(cls) -> bool:
        """Validate configuration settings"""
        required_settings = [
            "DATABASE_URL",
            "REDIS_URL",
            "SECRET_KEY"
        ]
        
        missing_settings = []
        for setting in required_settings:
            if not getattr(cls, setting, None):
                missing_settings.append(setting)
        
        if missing_settings:
            raise ValueError(f"Missing required configuration: {missing_settings}")
        
        return True
    
    @classmethod
    def setup_logging(cls):
        """Setup logging configuration"""
        logging.basicConfig(
            level=getattr(logging, cls.LOG_LEVEL.upper()),
            format=cls.LOG_FORMAT,
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler("logs/fraud_detection.log") if os.path.exists("logs") else logging.NullHandler()
            ]
        )
    
    @classmethod
    def get_environment_info(cls) -> Dict[str, Any]:
        """Get environment information"""
        return {
            "environment": cls.ENVIRONMENT,
            "debug": cls.DEBUG,
            "api_host": cls.API_HOST,
            "api_port": cls.API_PORT,
            "log_level": cls.LOG_LEVEL,
            "metrics_enabled": cls.METRICS_ENABLED,
            "feature_store_enabled": cls.FEATURE_STORE_ENABLED
        }

# Environment-specific configurations
class DevelopmentConfig(Config):
    """Development environment configuration"""
    DEBUG = True
    LOG_LEVEL = "DEBUG"
    API_WORKERS = 1

class ProductionConfig(Config):
    """Production environment configuration"""
    DEBUG = False
    LOG_LEVEL = "INFO"
    API_WORKERS = 4
    
    # Production-specific overrides
    REQUEST_TIMEOUT_SECONDS = 10
    MAX_CONCURRENT_REQUESTS = 1000

class TestingConfig(Config):
    """Testing environment configuration"""
    DEBUG = True
    LOG_LEVEL = "DEBUG"
    
    # Use in-memory/test databases
    DATABASE_URL = "sqlite:///test.db"
    REDIS_URL = "redis://localhost:6379/1"  # Different Redis DB
    
    # Disable external services
    METRICS_ENABLED = False
    ALERT_ENABLED = False

# Configuration factory
def get_config() -> Config:
    """Get configuration based on environment"""
    env = os.getenv("ENVIRONMENT", "development").lower()
    
    if env == "production":
        return ProductionConfig()
    elif env == "testing":
        return TestingConfig()
    else:
        return DevelopmentConfig()

# Global config instance
config = get_config()