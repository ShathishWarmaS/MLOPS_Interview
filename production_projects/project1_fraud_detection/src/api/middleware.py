#!/usr/bin/env python3
"""
API Middleware Components
Custom middleware for the fraud detection API
"""

import time
import logging
import uuid
from typing import Callable, Dict, Any
from fastapi import Request, Response
from fastapi.middleware.base import BaseHTTPMiddleware
from fastapi.responses import JSONResponse
import json
from starlette.middleware.base import BaseHTTPMiddleware as StarletteBaseHTTPMiddleware
from starlette.requests import Request as StarletteRequest
from starlette.responses import Response as StarletteResponse

from shared.config import config

logger = logging.getLogger(__name__)

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware for logging HTTP requests and responses
    
    Features:
    - Request/response logging
    - Request ID generation
    - Performance tracking
    - Request body logging (configurable)
    """
    
    def __init__(self, app, log_body: bool = False):
        super().__init__(app)
        self.log_body = log_body
        
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Generate request ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        # Log request start
        start_time = time.time()
        
        # Extract request info
        client_ip = self._get_client_ip(request)
        user_agent = request.headers.get('user-agent', 'unknown')
        
        # Log request
        logger.info(
            f"🔄 Request started - "
            f"ID: {request_id}, "
            f"Method: {request.method}, "
            f"Path: {request.url.path}, "
            f"IP: {client_ip}, "
            f"User-Agent: {user_agent}"
        )
        
        # Log request body if enabled
        if self.log_body and request.method in ['POST', 'PUT', 'PATCH']:
            try:
                body = await request.body()
                if body:
                    logger.debug(f"Request body [{request_id}]: {body.decode()}")
            except Exception as e:
                logger.warning(f"Failed to read request body: {e}")
        
        # Process request
        try:
            response = await call_next(request)
            
            # Calculate duration
            duration = time.time() - start_time
            
            # Log response
            logger.info(
                f"✅ Request completed - "
                f"ID: {request_id}, "
                f"Status: {response.status_code}, "
                f"Duration: {duration:.3f}s"
            )
            
            # Add request ID to response headers
            response.headers['X-Request-ID'] = request_id
            
            return response
            
        except Exception as e:
            # Calculate duration
            duration = time.time() - start_time
            
            # Log error
            logger.error(
                f"❌ Request failed - "
                f"ID: {request_id}, "
                f"Error: {str(e)}, "
                f"Duration: {duration:.3f}s"
            )
            
            # Return error response
            return JSONResponse(
                status_code=500,
                content={
                    'error': 'Internal server error',
                    'message': 'An unexpected error occurred',
                    'request_id': request_id
                },
                headers={'X-Request-ID': request_id}
            )
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address"""
        # Check for forwarded headers (load balancer/proxy)
        forwarded_for = request.headers.get('x-forwarded-for')
        if forwarded_for:
            return forwarded_for.split(',')[0].strip()
        
        real_ip = request.headers.get('x-real-ip')
        if real_ip:
            return real_ip
        
        # Fall back to direct connection
        return request.client.host if request.client else 'unknown'

class MetricsMiddleware(BaseHTTPMiddleware):
    """
    Middleware for collecting API metrics
    
    Features:
    - Request counting
    - Response time tracking
    - Error rate monitoring
    - Endpoint usage statistics
    """
    
    def __init__(self, app):
        super().__init__(app)
        self.request_count = 0
        self.error_count = 0
        self.endpoint_stats = {}
        
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Record request start
        start_time = time.time()
        endpoint = f"{request.method} {request.url.path}"
        
        # Initialize endpoint stats if needed
        if endpoint not in self.endpoint_stats:
            self.endpoint_stats[endpoint] = {
                'count': 0,
                'total_time': 0,
                'errors': 0,
                'avg_time': 0
            }
        
        # Increment request count
        self.request_count += 1
        self.endpoint_stats[endpoint]['count'] += 1
        
        try:
            # Process request
            response = await call_next(request)
            
            # Calculate duration
            duration = time.time() - start_time
            
            # Update endpoint stats
            self.endpoint_stats[endpoint]['total_time'] += duration
            self.endpoint_stats[endpoint]['avg_time'] = (
                self.endpoint_stats[endpoint]['total_time'] / 
                self.endpoint_stats[endpoint]['count']
            )
            
            # Track errors
            if response.status_code >= 400:
                self.error_count += 1
                self.endpoint_stats[endpoint]['errors'] += 1
            
            # Add metrics headers
            response.headers['X-Response-Time'] = f"{duration:.3f}s"
            response.headers['X-Request-Count'] = str(self.request_count)
            
            return response
            
        except Exception as e:
            # Calculate duration
            duration = time.time() - start_time
            
            # Update error stats
            self.error_count += 1
            self.endpoint_stats[endpoint]['errors'] += 1
            self.endpoint_stats[endpoint]['total_time'] += duration
            self.endpoint_stats[endpoint]['avg_time'] = (
                self.endpoint_stats[endpoint]['total_time'] / 
                self.endpoint_stats[endpoint]['count']
            )
            
            # Re-raise the exception to be handled by other middleware
            raise
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics"""
        return {
            'total_requests': self.request_count,
            'total_errors': self.error_count,
            'error_rate': self.error_count / max(self.request_count, 1),
            'endpoints': self.endpoint_stats
        }

class SecurityMiddleware(BaseHTTPMiddleware):
    """
    Security middleware for API protection
    
    Features:
    - Rate limiting
    - Request validation
    - Security headers
    - IP blocking (if configured)
    """
    
    def __init__(self, app, rate_limit: int = 100, time_window: int = 60):
        super().__init__(app)
        self.rate_limit = rate_limit
        self.time_window = time_window
        self.request_history = {}
        self.blocked_ips = set()
        
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        client_ip = self._get_client_ip(request)
        current_time = time.time()
        
        # Check if IP is blocked
        if client_ip in self.blocked_ips:
            logger.warning(f"🚫 Blocked IP attempted access: {client_ip}")
            return JSONResponse(
                status_code=403,
                content={'error': 'Forbidden', 'message': 'Access denied'}
            )
        
        # Rate limiting
        if not self._check_rate_limit(client_ip, current_time):
            logger.warning(f"⚠️ Rate limit exceeded for IP: {client_ip}")
            return JSONResponse(
                status_code=429,
                content={
                    'error': 'Too Many Requests',
                    'message': f'Rate limit exceeded. Max {self.rate_limit} requests per {self.time_window} seconds'
                }
            )
        
        # Process request
        response = await call_next(request)
        
        # Add security headers
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['X-Frame-Options'] = 'DENY'
        response.headers['X-XSS-Protection'] = '1; mode=block'
        response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
        
        return response
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address"""
        forwarded_for = request.headers.get('x-forwarded-for')
        if forwarded_for:
            return forwarded_for.split(',')[0].strip()
        
        real_ip = request.headers.get('x-real-ip')
        if real_ip:
            return real_ip
        
        return request.client.host if request.client else 'unknown'
    
    def _check_rate_limit(self, client_ip: str, current_time: float) -> bool:
        """Check if request is within rate limit"""
        # Clean old requests
        if client_ip in self.request_history:
            self.request_history[client_ip] = [
                timestamp for timestamp in self.request_history[client_ip]
                if current_time - timestamp < self.time_window
            ]
        else:
            self.request_history[client_ip] = []
        
        # Check rate limit
        if len(self.request_history[client_ip]) >= self.rate_limit:
            return False
        
        # Record current request
        self.request_history[client_ip].append(current_time)
        return True
    
    def block_ip(self, ip: str):
        """Block an IP address"""
        self.blocked_ips.add(ip)
        logger.warning(f"🚫 IP blocked: {ip}")
    
    def unblock_ip(self, ip: str):
        """Unblock an IP address"""
        self.blocked_ips.discard(ip)
        logger.info(f"✅ IP unblocked: {ip}")

class CORSMiddleware(BaseHTTPMiddleware):
    """
    Custom CORS middleware with advanced configuration
    
    Features:
    - Configurable origins
    - Method and header controls
    - Preflight request handling
    - Credential support
    """
    
    def __init__(
        self,
        app,
        allow_origins: list = None,
        allow_methods: list = None,
        allow_headers: list = None,
        allow_credentials: bool = True,
        max_age: int = 600
    ):
        super().__init__(app)
        self.allow_origins = allow_origins or ['*']
        self.allow_methods = allow_methods or ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS']
        self.allow_headers = allow_headers or ['*']
        self.allow_credentials = allow_credentials
        self.max_age = max_age
        
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        origin = request.headers.get('origin')
        
        # Handle preflight requests
        if request.method == 'OPTIONS':
            response = Response()
            self._add_cors_headers(response, origin)
            response.headers['Access-Control-Max-Age'] = str(self.max_age)
            return response
        
        # Process request
        response = await call_next(request)
        
        # Add CORS headers
        self._add_cors_headers(response, origin)
        
        return response
    
    def _add_cors_headers(self, response: Response, origin: str):
        """Add CORS headers to response"""
        # Check if origin is allowed
        if self.allow_origins == ['*'] or origin in self.allow_origins:
            response.headers['Access-Control-Allow-Origin'] = origin or '*'
        
        response.headers['Access-Control-Allow-Methods'] = ', '.join(self.allow_methods)
        response.headers['Access-Control-Allow-Headers'] = ', '.join(self.allow_headers)
        
        if self.allow_credentials:
            response.headers['Access-Control-Allow-Credentials'] = 'true'

class CompressionMiddleware(BaseHTTPMiddleware):
    """
    Response compression middleware
    
    Features:
    - Gzip compression
    - Configurable compression level
    - Content-type based compression
    - Size threshold
    """
    
    def __init__(self, app, minimum_size: int = 1024, compression_level: int = 6):
        super().__init__(app)
        self.minimum_size = minimum_size
        self.compression_level = compression_level
        self.compressible_types = {
            'application/json',
            'application/javascript',
            'text/html',
            'text/css',
            'text/plain',
            'text/xml'
        }
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Check if client accepts gzip
        accept_encoding = request.headers.get('accept-encoding', '')
        if 'gzip' not in accept_encoding:
            return await call_next(request)
        
        # Process request
        response = await call_next(request)
        
        # Check if response should be compressed
        if not self._should_compress(response):
            return response
        
        # Compress response
        try:
            compressed_response = self._compress_response(response)
            return compressed_response
        except Exception as e:
            logger.error(f"Compression failed: {e}")
            return response
    
    def _should_compress(self, response: Response) -> bool:
        """Check if response should be compressed"""
        # Check content type
        content_type = response.headers.get('content-type', '')
        if not any(ct in content_type for ct in self.compressible_types):
            return False
        
        # Check if already compressed
        if response.headers.get('content-encoding'):
            return False
        
        # Check size (this is a simplified check)
        content_length = response.headers.get('content-length')
        if content_length and int(content_length) < self.minimum_size:
            return False
        
        return True
    
    def _compress_response(self, response: Response) -> Response:
        """Compress response body"""
        import gzip
        
        # Get response body
        if hasattr(response, 'body'):
            body = response.body
        else:
            # For streaming responses, we'd need more complex handling
            return response
        
        # Compress body
        compressed_body = gzip.compress(body, compresslevel=self.compression_level)
        
        # Create new response with compressed body
        compressed_response = Response(
            content=compressed_body,
            status_code=response.status_code,
            headers=dict(response.headers),
            media_type=response.media_type
        )
        
        # Add compression headers
        compressed_response.headers['content-encoding'] = 'gzip'
        compressed_response.headers['content-length'] = str(len(compressed_body))
        
        return compressed_response

class HealthCheckMiddleware(BaseHTTPMiddleware):
    """
    Health check middleware for monitoring
    
    Features:
    - Service health tracking
    - Response time monitoring
    - Error rate tracking
    - Health score calculation
    """
    
    def __init__(self, app):
        super().__init__(app)
        self.start_time = time.time()
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.total_response_time = 0
        
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()
        
        try:
            # Process request
            response = await call_next(request)
            
            # Update metrics
            self.total_requests += 1
            response_time = time.time() - start_time
            self.total_response_time += response_time
            
            if response.status_code < 400:
                self.successful_requests += 1
            else:
                self.failed_requests += 1
            
            # Add health metrics to response headers
            response.headers['X-Health-Score'] = str(self._calculate_health_score())
            response.headers['X-Uptime'] = str(int(time.time() - self.start_time))
            
            return response
            
        except Exception as e:
            # Update error metrics
            self.total_requests += 1
            self.failed_requests += 1
            response_time = time.time() - start_time
            self.total_response_time += response_time
            
            # Re-raise the exception
            raise
    
    def _calculate_health_score(self) -> float:
        """Calculate health score (0-100)"""
        if self.total_requests == 0:
            return 100.0
        
        # Calculate success rate
        success_rate = self.successful_requests / self.total_requests
        
        # Calculate average response time score
        avg_response_time = self.total_response_time / self.total_requests
        response_time_score = max(0, 1 - (avg_response_time / 2))  # Penalty for >2s responses
        
        # Calculate uptime score
        uptime_seconds = time.time() - self.start_time
        uptime_score = min(1.0, uptime_seconds / 3600)  # Full score after 1 hour
        
        # Combined health score
        health_score = (success_rate * 0.5 + response_time_score * 0.3 + uptime_score * 0.2) * 100
        
        return round(health_score, 2)
    
    def get_health_metrics(self) -> Dict[str, Any]:
        """Get current health metrics"""
        return {
            'uptime_seconds': int(time.time() - self.start_time),
            'total_requests': self.total_requests,
            'successful_requests': self.successful_requests,
            'failed_requests': self.failed_requests,
            'success_rate': self.successful_requests / max(self.total_requests, 1),
            'average_response_time': self.total_response_time / max(self.total_requests, 1),
            'health_score': self._calculate_health_score()
        }