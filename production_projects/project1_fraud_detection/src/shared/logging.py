"""
Logging Configuration for Fraud Detection System
Structured logging with performance monitoring
"""

import logging
import logging.handlers
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any
import os

class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging"""
    
    def format(self, record):
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add extra fields if present
        if hasattr(record, "request_id"):
            log_entry["request_id"] = record.request_id
        
        if hasattr(record, "user_id"):
            log_entry["user_id"] = record.user_id
        
        if hasattr(record, "duration_ms"):
            log_entry["duration_ms"] = record.duration_ms
        
        if hasattr(record, "model_version"):
            log_entry["model_version"] = record.model_version
        
        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        return json.dumps(log_entry)

class PerformanceLogger:
    """Logger for performance metrics"""
    
    def __init__(self, logger_name: str):
        self.logger = logging.getLogger(f"{logger_name}.performance")
    
    def log_prediction_performance(self, duration_ms: float, model_version: str, 
                                  request_id: str = None):
        """Log prediction performance metrics"""
        extra = {
            "duration_ms": duration_ms,
            "model_version": model_version,
            "metric_type": "prediction_latency"
        }
        
        if request_id:
            extra["request_id"] = request_id
        
        self.logger.info("Prediction completed", extra=extra)
    
    def log_model_load_performance(self, duration_ms: float, model_version: str):
        """Log model loading performance"""
        extra = {
            "duration_ms": duration_ms,
            "model_version": model_version,
            "metric_type": "model_load_time"
        }
        
        self.logger.info("Model loaded", extra=extra)

class RequestLogger:
    """Logger for API requests"""
    
    def __init__(self, logger_name: str):
        self.logger = logging.getLogger(f"{logger_name}.requests")
    
    def log_request(self, method: str, path: str, status_code: int, 
                   duration_ms: float, request_id: str = None, user_id: str = None):
        """Log API request details"""
        extra = {
            "method": method,
            "path": path,
            "status_code": status_code,
            "duration_ms": duration_ms,
            "metric_type": "api_request"
        }
        
        if request_id:
            extra["request_id"] = request_id
        
        if user_id:
            extra["user_id"] = user_id
        
        level = logging.INFO
        if status_code >= 500:
            level = logging.ERROR
        elif status_code >= 400:
            level = logging.WARNING
        
        self.logger.log(level, f"{method} {path} - {status_code}", extra=extra)

class SecurityLogger:
    """Logger for security events"""
    
    def __init__(self, logger_name: str):
        self.logger = logging.getLogger(f"{logger_name}.security")
    
    def log_suspicious_activity(self, activity_type: str, details: Dict[str, Any], 
                               severity: str = "medium"):
        """Log suspicious activity"""
        extra = {
            "activity_type": activity_type,
            "severity": severity,
            "details": details,
            "metric_type": "security_event"
        }
        
        level = logging.WARNING
        if severity == "high":
            level = logging.ERROR
        elif severity == "low":
            level = logging.INFO
        
        self.logger.log(level, f"Suspicious activity: {activity_type}", extra=extra)

def setup_logging(name: str, log_level: str = "INFO", 
                 log_file: str = None, use_json: bool = False) -> logging.Logger:
    """
    Setup logging configuration
    
    Args:
        name: Logger name
        log_level: Logging level
        log_file: Optional log file path
        use_json: Whether to use JSON formatting
    
    Returns:
        Configured logger
    """
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Choose formatter
    if use_json:
        formatter = JSONFormatter()
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        # Ensure log directory exists
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Rotating file handler
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def setup_application_logging():
    """Setup logging for the entire application"""
    
    # Get configuration
    from shared.config import config
    
    # Ensure logs directory exists
    os.makedirs("logs", exist_ok=True)
    
    # Main application logger
    app_logger = setup_logging(
        "fraud_detection",
        log_level=config.LOG_LEVEL,
        log_file="logs/fraud_detection.log",
        use_json=True
    )
    
    # API logger
    api_logger = setup_logging(
        "fraud_detection.api",
        log_level=config.LOG_LEVEL,
        log_file="logs/api.log",
        use_json=True
    )
    
    # Performance logger
    perf_logger = setup_logging(
        "fraud_detection.performance",
        log_level="INFO",
        log_file="logs/performance.log",
        use_json=True
    )
    
    # Security logger
    security_logger = setup_logging(
        "fraud_detection.security",
        log_level="INFO",
        log_file="logs/security.log",
        use_json=True
    )
    
    # Model logger
    model_logger = setup_logging(
        "fraud_detection.model",
        log_level=config.LOG_LEVEL,
        log_file="logs/model.log",
        use_json=True
    )
    
    # Suppress noisy third-party loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    
    return app_logger

class LoggerMixin:
    """Mixin class to add logging capabilities to any class"""
    
    @property
    def logger(self):
        if not hasattr(self, '_logger'):
            self._logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
        return self._logger

class TimingLogger:
    """Context manager for timing operations"""
    
    def __init__(self, logger: logging.Logger, operation: str, 
                 level: int = logging.INFO, extra: Dict[str, Any] = None):
        self.logger = logger
        self.operation = operation
        self.level = level
        self.extra = extra or {}
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        self.logger.log(self.level, f"Starting {self.operation}", extra=self.extra)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration_ms = (time.time() - self.start_time) * 1000
        
        extra = self.extra.copy()
        extra["duration_ms"] = duration_ms
        
        if exc_type is None:
            self.logger.log(self.level, f"Completed {self.operation}", extra=extra)
        else:
            extra["exception_type"] = exc_type.__name__
            self.logger.error(f"Failed {self.operation}", extra=extra)

# Global logger instances
app_logger = None
performance_logger = None
request_logger = None
security_logger = None

def get_logger(name: str = None) -> logging.Logger:
    """Get a logger instance"""
    if name:
        return logging.getLogger(name)
    
    global app_logger
    if app_logger is None:
        app_logger = setup_application_logging()
    return app_logger

def get_performance_logger() -> PerformanceLogger:
    """Get performance logger instance"""
    global performance_logger
    if performance_logger is None:
        performance_logger = PerformanceLogger("fraud_detection")
    return performance_logger

def get_request_logger() -> RequestLogger:
    """Get request logger instance"""
    global request_logger
    if request_logger is None:
        request_logger = RequestLogger("fraud_detection")
    return request_logger

def get_security_logger() -> SecurityLogger:
    """Get security logger instance"""
    global security_logger
    if security_logger is None:
        security_logger = SecurityLogger("fraud_detection")
    return security_logger