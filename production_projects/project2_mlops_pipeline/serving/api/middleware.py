"""
Custom Middleware for MLOps API
Request logging, metrics collection, and monitoring
"""

import time
import uuid
import logging
import json
from typing import Callable, Dict, Any
from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from prometheus_client import Counter, Histogram, Gauge
import asyncio

logger = logging.getLogger(__name__)

# Prometheus metrics
REQUEST_COUNT = Counter(
    'http_requests_total', 
    'Total HTTP requests', 
    ['method', 'endpoint', 'status_code']
)

REQUEST_DURATION = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration in seconds',
    ['method', 'endpoint']
)

ACTIVE_REQUESTS = Gauge(
    'http_requests_active',
    'Number of active HTTP requests'
)

ERROR_COUNT = Counter(
    'http_errors_total',
    'Total HTTP errors',
    ['method', 'endpoint', 'error_type']
)

class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for comprehensive request/response logging"""
    
    def __init__(self, app, log_body: bool = False, max_body_size: int = 1024):
        super().__init__(app)
        self.log_body = log_body
        self.max_body_size = max_body_size
        
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Generate request ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        # Start timing
        start_time = time.time()
        
        # Log request
        await self._log_request(request, request_id)
        
        # Process request
        try:
            response = await call_next(request)
            
            # Calculate duration
            duration = time.time() - start_time
            
            # Log response
            await self._log_response(request, response, duration, request_id)
            
            # Add headers
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Response-Time"] = f"{duration:.3f}s"
            
            return response
            
        except Exception as e:
            duration = time.time() - start_time
            
            # Log error
            await self._log_error(request, e, duration, request_id)
            
            # Return error response
            return JSONResponse(
                status_code=500,
                content={
                    "error": "internal_server_error",
                    "message": "An unexpected error occurred",
                    "request_id": request_id,
                    "timestamp": time.time()
                },
                headers={"X-Request-ID": request_id}
            )
    
    async def _log_request(self, request: Request, request_id: str):
        """Log incoming request"""
        try:
            # Basic request info
            log_data = {
                "event": "request_started",
                "request_id": request_id,
                "method": request.method,
                "url": str(request.url),
                "path": request.url.path,
                "query_params": dict(request.query_params),
                "headers": dict(request.headers),
                "client_ip": self._get_client_ip(request),
                "user_agent": request.headers.get("user-agent"),
                "timestamp": time.time()
            }
            
            # Log request body if enabled
            if self.log_body and request.method in ["POST", "PUT", "PATCH"]:
                try:
                    body = await request.body()
                    if len(body) <= self.max_body_size:
                        if request.headers.get("content-type", "").startswith("application/json"):
                            log_data["body"] = json.loads(body.decode())
                        else:
                            log_data["body"] = body.decode()[:self.max_body_size]
                    else:
                        log_data["body_size"] = len(body)
                        log_data["body_truncated"] = True
                except Exception:
                    log_data["body_error"] = "Could not read request body"
            
            logger.info(f"Request started: {json.dumps(log_data, default=str)}")
            
        except Exception as e:
            logger.error(f"Failed to log request: {e}")
    
    async def _log_response(self, request: Request, response: Response, duration: float, request_id: str):
        """Log response"""
        try:
            log_data = {
                "event": "request_completed",
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration_ms": duration * 1000,
                "response_headers": dict(response.headers),
                "timestamp": time.time()
            }
            
            # Determine log level based on status code
            if response.status_code >= 500:
                logger.error(f"Request completed with error: {json.dumps(log_data, default=str)}")
            elif response.status_code >= 400:
                logger.warning(f"Request completed with client error: {json.dumps(log_data, default=str)}")
            else:
                logger.info(f"Request completed successfully: {json.dumps(log_data, default=str)}")
                
        except Exception as e:
            logger.error(f"Failed to log response: {e}")
    
    async def _log_error(self, request: Request, error: Exception, duration: float, request_id: str):
        """Log error"""
        try:
            log_data = {
                "event": "request_error",
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "error_type": type(error).__name__,
                "error_message": str(error),
                "duration_ms": duration * 1000,
                "timestamp": time.time()
            }
            
            logger.error(f"Request failed: {json.dumps(log_data, default=str)}")
            
        except Exception as e:
            logger.error(f"Failed to log error: {e}")
    
    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address"""
        # Check for forwarded headers first
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip
        
        # Fallback to client address
        if hasattr(request, "client") and request.client:
            return request.client.host
        
        return "unknown"

class MetricsMiddleware(BaseHTTPMiddleware):
    """Middleware for Prometheus metrics collection"""
    
    def __init__(self, app):
        super().__init__(app)
        
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Extract endpoint from path
        endpoint = self._get_endpoint(request.url.path)
        method = request.method
        
        # Increment active requests
        ACTIVE_REQUESTS.inc()
        
        # Start timing
        start_time = time.time()
        
        try:
            # Process request
            response = await call_next(request)
            
            # Record metrics
            duration = time.time() - start_time
            status_code = str(response.status_code)
            
            REQUEST_COUNT.labels(
                method=method,
                endpoint=endpoint,
                status_code=status_code
            ).inc()
            
            REQUEST_DURATION.labels(
                method=method,
                endpoint=endpoint
            ).observe(duration)
            
            # Record errors
            if response.status_code >= 400:
                error_type = "client_error" if response.status_code < 500 else "server_error"
                ERROR_COUNT.labels(
                    method=method,
                    endpoint=endpoint,
                    error_type=error_type
                ).inc()
            
            return response
            
        except Exception as e:
            # Record exception
            duration = time.time() - start_time
            
            ERROR_COUNT.labels(
                method=method,
                endpoint=endpoint,
                error_type="exception"
            ).inc()
            
            REQUEST_DURATION.labels(
                method=method,
                endpoint=endpoint
            ).observe(duration)
            
            raise
            
        finally:
            # Decrement active requests
            ACTIVE_REQUESTS.dec()
    
    def _get_endpoint(self, path: str) -> str:
        """Extract endpoint from path for metrics"""
        # Remove query parameters
        path = path.split("?")[0]
        
        # Group similar endpoints
        if path.startswith("/predict"):
            return "/predict"
        elif path.startswith("/health"):
            return "/health"
        elif path.startswith("/metrics"):
            return "/metrics"
        elif path.startswith("/model"):
            return "/model"
        elif path.startswith("/docs"):
            return "/docs"
        else:
            return "/other"

class SecurityMiddleware(BaseHTTPMiddleware):
    """Security middleware for API protection"""
    
    def __init__(self, app, enable_rate_limiting: bool = True, max_requests_per_minute: int = 100):
        super().__init__(app)
        self.enable_rate_limiting = enable_rate_limiting
        self.max_requests_per_minute = max_requests_per_minute
        self.request_counts = {}  # Simple in-memory rate limiting
        self.last_cleanup = time.time()
        
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Security headers
        response = await call_next(request)
        
        # Add security headers
        self._add_security_headers(response)
        
        # Rate limiting
        if self.enable_rate_limiting:
            client_ip = self._get_client_ip(request)
            if not await self._check_rate_limit(client_ip):
                return JSONResponse(
                    status_code=429,
                    content={
                        "error": "rate_limit_exceeded",
                        "message": "Too many requests",
                        "timestamp": time.time()
                    }
                )
        
        return response
    
    def _add_security_headers(self, response: Response):
        """Add security headers to response"""
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    
    async def _check_rate_limit(self, client_ip: str) -> bool:
        """Check if client has exceeded rate limit"""
        current_time = time.time()
        current_minute = int(current_time // 60)
        
        # Cleanup old entries
        if current_time - self.last_cleanup > 60:
            self._cleanup_rate_limit_cache(current_minute)
            self.last_cleanup = current_time
        
        # Check rate limit
        key = f"{client_ip}:{current_minute}"
        count = self.request_counts.get(key, 0)
        
        if count >= self.max_requests_per_minute:
            return False
        
        # Increment counter
        self.request_counts[key] = count + 1
        return True
    
    def _cleanup_rate_limit_cache(self, current_minute: int):
        """Remove old entries from rate limit cache"""
        keys_to_remove = []
        for key in self.request_counts:
            _, minute_str = key.split(":")
            if int(minute_str) < current_minute - 1:  # Keep last 2 minutes
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self.request_counts[key]
    
    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address"""
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        if hasattr(request, "client") and request.client:
            return request.client.host
        
        return "unknown"

class HealthCheckMiddleware(BaseHTTPMiddleware):
    """Middleware for health check optimization"""
    
    def __init__(self, app):
        super().__init__(app)
        self.health_endpoints = {"/health", "/ready", "/live"}
        
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Fast path for health checks
        if request.url.path in self.health_endpoints and request.method == "GET":
            # Skip expensive middleware for health checks
            request.state.is_health_check = True
        
        return await call_next(request)

class CacheMiddleware(BaseHTTPMiddleware):
    """Simple caching middleware for read-only endpoints"""
    
    def __init__(self, app, cache_ttl: int = 300):
        super().__init__(app)
        self.cache_ttl = cache_ttl
        self.cache = {}
        self.cacheable_endpoints = {"/model/info", "/debug/features"}
        
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Only cache GET requests to specific endpoints
        if (request.method == "GET" and 
            request.url.path in self.cacheable_endpoints):
            
            cache_key = f"{request.url.path}?{request.url.query}"
            
            # Check cache
            if cache_key in self.cache:
                cached_response, cached_time = self.cache[cache_key]
                if time.time() - cached_time < self.cache_ttl:
                    # Return cached response
                    response = Response(
                        content=cached_response["content"],
                        status_code=cached_response["status_code"],
                        headers=cached_response["headers"]
                    )
                    response.headers["X-Cache"] = "HIT"
                    return response
            
            # Process request
            response = await call_next(request)
            
            # Cache successful responses
            if response.status_code == 200:
                cached_response = {
                    "content": response.body,
                    "status_code": response.status_code,
                    "headers": dict(response.headers)
                }
                self.cache[cache_key] = (cached_response, time.time())
                response.headers["X-Cache"] = "MISS"
            
            return response
        
        # Non-cacheable request
        return await call_next(request)

class ValidationMiddleware(BaseHTTPMiddleware):
    """Input validation and sanitization middleware"""
    
    def __init__(self, app, max_request_size: int = 10 * 1024 * 1024):  # 10MB
        super().__init__(app)
        self.max_request_size = max_request_size
        
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Check request size
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > self.max_request_size:
            return JSONResponse(
                status_code=413,
                content={
                    "error": "request_too_large",
                    "message": f"Request size exceeds maximum of {self.max_request_size} bytes",
                    "timestamp": time.time()
                }
            )
        
        # Validate content type for prediction endpoints
        if (request.url.path.startswith("/predict") and 
            request.method in ["POST", "PUT", "PATCH"]):
            
            content_type = request.headers.get("content-type", "")
            if not content_type.startswith("application/json"):
                return JSONResponse(
                    status_code=415,
                    content={
                        "error": "unsupported_media_type",
                        "message": "Content-Type must be application/json",
                        "timestamp": time.time()
                    }
                )
        
        return await call_next(request)