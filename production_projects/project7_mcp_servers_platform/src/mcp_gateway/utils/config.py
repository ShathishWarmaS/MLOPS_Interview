"""
Configuration management for MCP Gateway
Handles environment variables and configuration settings
"""

import os
import time
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class DatabaseConfig:
    """Database configuration"""
    url: str = "postgresql://mcp:mcp@localhost:5432/mcp"
    pool_size: int = 10
    max_overflow: int = 20
    echo: bool = False
    ssl_mode: str = "prefer"

@dataclass
class RedisConfig:
    """Redis configuration"""
    host: str = "localhost"
    port: int = 6379
    password: Optional[str] = None
    db: int = 0
    max_connections: int = 50
    socket_timeout: int = 30
    cluster_mode: bool = False

@dataclass
class JWTConfig:
    """JWT authentication configuration"""
    secret_key: str = "your-secret-key-change-in-production"
    algorithm: str = "HS256"
    expires_in: int = 3600  # 1 hour
    refresh_expires_in: int = 86400  # 24 hours

@dataclass
class SecurityConfig:
    """Security configuration"""
    rate_limit_enabled: bool = True
    requests_per_minute: int = 100
    burst_limit: int = 150
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    cors_methods: List[str] = field(default_factory=lambda: ["GET", "POST", "PUT", "DELETE"])
    cors_headers: List[str] = field(default_factory=lambda: ["*"])

@dataclass
class MonitoringConfig:
    """Monitoring and observability configuration"""
    metrics_enabled: bool = True
    metrics_port: int = 9090
    tracing_enabled: bool = True
    jaeger_endpoint: Optional[str] = None
    log_level: str = "INFO"
    structured_logging: bool = True

@dataclass
class ServerConfig:
    """Server configuration"""
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    reload: bool = False
    debug: bool = False
    max_connections: int = 1000
    timeout_seconds: int = 30

@dataclass
class RoutingConfig:
    """Request routing configuration"""
    load_balancer_strategy: str = "round_robin"  # round_robin, least_connections, weighted
    health_check_interval: int = 30
    circuit_breaker_enabled: bool = True
    circuit_breaker_failure_threshold: int = 5
    circuit_breaker_timeout: int = 60

class Config:
    """Main configuration class"""
    
    def __init__(self):
        self.start_time = time.time()
        
        # Load environment variables
        self._load_env_vars()
        
        # Initialize configuration sections
        self.database = DatabaseConfig(
            url=os.getenv("DATABASE_URL", "postgresql://mcp:mcp@localhost:5432/mcp"),
            pool_size=int(os.getenv("DB_POOL_SIZE", "10")),
            max_overflow=int(os.getenv("DB_MAX_OVERFLOW", "20")),
            echo=os.getenv("DB_ECHO", "false").lower() == "true",
            ssl_mode=os.getenv("DB_SSL_MODE", "prefer")
        )
        
        self.redis = RedisConfig(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", "6379")),
            password=os.getenv("REDIS_PASSWORD"),
            db=int(os.getenv("REDIS_DB", "0")),
            max_connections=int(os.getenv("REDIS_MAX_CONNECTIONS", "50")),
            socket_timeout=int(os.getenv("REDIS_SOCKET_TIMEOUT", "30")),
            cluster_mode=os.getenv("REDIS_CLUSTER_MODE", "false").lower() == "true"
        )
        
        self.jwt = JWTConfig(
            secret_key=os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production"),
            algorithm=os.getenv("JWT_ALGORITHM", "HS256"),
            expires_in=int(os.getenv("JWT_EXPIRES_IN", "3600")),
            refresh_expires_in=int(os.getenv("JWT_REFRESH_EXPIRES_IN", "86400"))
        )
        
        self.security = SecurityConfig(
            rate_limit_enabled=os.getenv("RATE_LIMIT_ENABLED", "true").lower() == "true",
            requests_per_minute=int(os.getenv("RATE_LIMIT_PER_MINUTE", "100")),
            burst_limit=int(os.getenv("RATE_LIMIT_BURST", "150")),
            cors_origins=self._parse_list(os.getenv("CORS_ORIGINS", "*")),
            cors_methods=self._parse_list(os.getenv("CORS_METHODS", "GET,POST,PUT,DELETE")),
            cors_headers=self._parse_list(os.getenv("CORS_HEADERS", "*"))
        )
        
        self.monitoring = MonitoringConfig(
            metrics_enabled=os.getenv("METRICS_ENABLED", "true").lower() == "true",
            metrics_port=int(os.getenv("METRICS_PORT", "9090")),
            tracing_enabled=os.getenv("TRACING_ENABLED", "true").lower() == "true",
            jaeger_endpoint=os.getenv("JAEGER_ENDPOINT"),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            structured_logging=os.getenv("STRUCTURED_LOGGING", "true").lower() == "true"
        )
        
        self.server = ServerConfig(
            host=os.getenv("HOST", "0.0.0.0"),
            port=int(os.getenv("PORT", "8000")),
            workers=int(os.getenv("WORKERS", "1")),
            reload=os.getenv("RELOAD", "false").lower() == "true",
            debug=os.getenv("DEBUG", "false").lower() == "true",
            max_connections=int(os.getenv("MAX_CONNECTIONS", "1000")),
            timeout_seconds=int(os.getenv("TIMEOUT_SECONDS", "30"))
        )
        
        self.routing = RoutingConfig(
            load_balancer_strategy=os.getenv("LOAD_BALANCER_STRATEGY", "round_robin"),
            health_check_interval=int(os.getenv("HEALTH_CHECK_INTERVAL", "30")),
            circuit_breaker_enabled=os.getenv("CIRCUIT_BREAKER_ENABLED", "true").lower() == "true",
            circuit_breaker_failure_threshold=int(os.getenv("CIRCUIT_BREAKER_FAILURE_THRESHOLD", "5")),
            circuit_breaker_timeout=int(os.getenv("CIRCUIT_BREAKER_TIMEOUT", "60"))
        )
        
        # Legacy properties for backward compatibility
        self.redis_host = self.redis.host
        self.redis_port = self.redis.port
        self.redis_password = self.redis.password
        self.jwt_secret_key = self.jwt.secret_key
        self.jwt_algorithm = self.jwt.algorithm
        self.jwt_expires_in = self.jwt.expires_in
        self.host = self.server.host
        self.port = self.server.port
        self.log_level = self.monitoring.log_level
    
    def _load_env_vars(self):
        """Load environment variables from .env file if present"""
        env_file = Path(".env")
        if env_file.exists():
            with open(env_file) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        key, _, value = line.partition("=")
                        if key and value:
                            os.environ.setdefault(key.strip(), value.strip())
    
    def _parse_list(self, value: str) -> List[str]:
        """Parse comma-separated string into list"""
        if not value:
            return []
        return [item.strip() for item in value.split(",") if item.strip()]
    
    def get_database_url(self) -> str:
        """Get database URL"""
        return self.database.url
    
    def get_redis_url(self) -> str:
        """Get Redis URL"""
        if self.redis.password:
            return f"redis://:{self.redis.password}@{self.redis.host}:{self.redis.port}/{self.redis.db}"
        else:
            return f"redis://{self.redis.host}:{self.redis.port}/{self.redis.db}"
    
    def is_development(self) -> bool:
        """Check if running in development mode"""
        return os.getenv("ENVIRONMENT", "development").lower() == "development"
    
    def is_production(self) -> bool:
        """Check if running in production mode"""
        return os.getenv("ENVIRONMENT", "development").lower() == "production"
    
    def get_service_discovery_config(self) -> Dict[str, Any]:
        """Get service discovery configuration"""
        return {
            "enabled": os.getenv("SERVICE_DISCOVERY_ENABLED", "false").lower() == "true",
            "backend": os.getenv("SERVICE_DISCOVERY_BACKEND", "static"),  # static, consul, etcd
            "endpoint": os.getenv("SERVICE_DISCOVERY_ENDPOINT"),
            "refresh_interval": int(os.getenv("SERVICE_DISCOVERY_REFRESH_INTERVAL", "30"))
        }
    
    def get_cache_config(self) -> Dict[str, Any]:
        """Get caching configuration"""
        return {
            "enabled": os.getenv("CACHE_ENABLED", "true").lower() == "true",
            "ttl_seconds": int(os.getenv("CACHE_TTL_SECONDS", "300")),
            "max_size_mb": int(os.getenv("CACHE_MAX_SIZE_MB", "100")),
            "compression": os.getenv("CACHE_COMPRESSION", "true").lower() == "true"
        }
    
    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration"""
        return {
            "level": self.monitoring.log_level,
            "structured": self.monitoring.structured_logging,
            "format": os.getenv("LOG_FORMAT", "json" if self.monitoring.structured_logging else "text"),
            "file_path": os.getenv("LOG_FILE_PATH"),
            "max_file_size_mb": int(os.getenv("LOG_MAX_FILE_SIZE_MB", "100")),
            "backup_count": int(os.getenv("LOG_BACKUP_COUNT", "5")),
            "correlation_id": True
        }
    
    def get_health_check_config(self) -> Dict[str, Any]:
        """Get health check configuration"""
        return {
            "enabled": os.getenv("HEALTH_CHECK_ENABLED", "true").lower() == "true",
            "endpoint": os.getenv("HEALTH_CHECK_ENDPOINT", "/health"),
            "timeout_seconds": int(os.getenv("HEALTH_CHECK_TIMEOUT", "10")),
            "include_details": os.getenv("HEALTH_CHECK_INCLUDE_DETAILS", "true").lower() == "true"
        }
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of errors"""
        errors = []
        
        # Validate required settings
        if self.jwt.secret_key == "your-secret-key-change-in-production" and self.is_production():
            errors.append("JWT secret key must be changed in production")
        
        if not self.database.url:
            errors.append("Database URL is required")
        
        if self.server.port < 1 or self.server.port > 65535:
            errors.append("Server port must be between 1 and 65535")
        
        if self.redis.port < 1 or self.redis.port > 65535:
            errors.append("Redis port must be between 1 and 65535")
        
        # Validate monitoring settings
        if self.monitoring.tracing_enabled and not self.monitoring.jaeger_endpoint:
            errors.append("Jaeger endpoint is required when tracing is enabled")
        
        return errors
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            "database": {
                "url": "***" if "password" in self.database.url else self.database.url,
                "pool_size": self.database.pool_size,
                "max_overflow": self.database.max_overflow,
                "echo": self.database.echo,
                "ssl_mode": self.database.ssl_mode
            },
            "redis": {
                "host": self.redis.host,
                "port": self.redis.port,
                "password": "***" if self.redis.password else None,
                "db": self.redis.db,
                "max_connections": self.redis.max_connections,
                "socket_timeout": self.redis.socket_timeout,
                "cluster_mode": self.redis.cluster_mode
            },
            "jwt": {
                "secret_key": "***",
                "algorithm": self.jwt.algorithm,
                "expires_in": self.jwt.expires_in,
                "refresh_expires_in": self.jwt.refresh_expires_in
            },
            "security": {
                "rate_limit_enabled": self.security.rate_limit_enabled,
                "requests_per_minute": self.security.requests_per_minute,
                "burst_limit": self.security.burst_limit,
                "cors_origins": self.security.cors_origins,
                "cors_methods": self.security.cors_methods,
                "cors_headers": self.security.cors_headers
            },
            "monitoring": {
                "metrics_enabled": self.monitoring.metrics_enabled,
                "metrics_port": self.monitoring.metrics_port,
                "tracing_enabled": self.monitoring.tracing_enabled,
                "jaeger_endpoint": self.monitoring.jaeger_endpoint,
                "log_level": self.monitoring.log_level,
                "structured_logging": self.monitoring.structured_logging
            },
            "server": {
                "host": self.server.host,
                "port": self.server.port,
                "workers": self.server.workers,
                "reload": self.server.reload,
                "debug": self.server.debug,
                "max_connections": self.server.max_connections,
                "timeout_seconds": self.server.timeout_seconds
            },
            "routing": {
                "load_balancer_strategy": self.routing.load_balancer_strategy,
                "health_check_interval": self.routing.health_check_interval,
                "circuit_breaker_enabled": self.routing.circuit_breaker_enabled,
                "circuit_breaker_failure_threshold": self.routing.circuit_breaker_failure_threshold,
                "circuit_breaker_timeout": self.routing.circuit_breaker_timeout
            }
        }