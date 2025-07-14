"""
MCP Gateway Server - FastAPI Application
Central hub for Model Context Protocol connections and routing
"""

import asyncio
import logging
import time
import json
import uvicorn
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, WebSocket, HTTPException, Depends, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from prometheus_client import make_asgi_app
import redis.asyncio as redis
from contextlib import asynccontextmanager

# Local imports
from .protocol.handler import MCPProtocolHandler
from .protocol.transport import WebSocketTransport, HTTPTransport
from .auth.middleware import AuthMiddleware
from .auth.jwt_handler import JWTHandler
from .routing.router import MCPRouter
from .monitoring.metrics import MetricsCollector
from .monitoring.health import HealthChecker
from .utils.config import Config
from .utils.exceptions import MCPException, AuthenticationError, ValidationError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MCPGateway:
    """Main MCP Gateway Server class"""
    
    def __init__(self):
        self.config = Config()
        self.app = None
        self.redis_client = None
        self.protocol_handler = None
        self.router = None
        self.metrics = None
        self.health_checker = None
        self.jwt_handler = None
        self.auth_middleware = None
        
        # Connection tracking
        self.active_connections: Dict[str, WebSocket] = {}
        self.connection_metadata: Dict[str, Dict[str, Any]] = {}
    
    async def initialize(self):
        """Initialize all gateway components"""
        try:
            logger.info("Initializing MCP Gateway...")
            
            # Initialize Redis client
            self.redis_client = redis.Redis(
                host=self.config.redis_host,
                port=self.config.redis_port,
                password=self.config.redis_password,
                decode_responses=True
            )
            
            # Test Redis connection
            await self.redis_client.ping()
            logger.info("✅ Redis connection established")
            
            # Initialize JWT handler
            self.jwt_handler = JWTHandler(
                secret_key=self.config.jwt_secret_key,
                algorithm=self.config.jwt_algorithm,
                expires_in=self.config.jwt_expires_in
            )
            
            # Initialize authentication middleware
            self.auth_middleware = AuthMiddleware(
                jwt_handler=self.jwt_handler,
                redis_client=self.redis_client
            )
            
            # Initialize metrics collector
            self.metrics = MetricsCollector()
            
            # Initialize health checker
            self.health_checker = HealthChecker(self.redis_client)
            
            # Initialize MCP router
            self.router = MCPRouter(
                redis_client=self.redis_client,
                metrics=self.metrics,
                config=self.config
            )
            await self.router.initialize()
            
            # Initialize protocol handler
            self.protocol_handler = MCPProtocolHandler(
                router=self.router,
                metrics=self.metrics,
                auth_middleware=self.auth_middleware
            )
            
            logger.info("✅ MCP Gateway initialization completed")
            
        except Exception as e:
            logger.error(f"Gateway initialization failed: {e}")
            raise
    
    async def shutdown(self):
        """Cleanup gateway resources"""
        try:
            logger.info("Shutting down MCP Gateway...")
            
            # Close all active connections
            for connection_id, websocket in self.active_connections.items():
                try:
                    await websocket.close()
                except Exception:
                    pass
            
            # Cleanup components
            if self.router:
                await self.router.shutdown()
            
            if self.redis_client:
                await self.redis_client.close()
            
            logger.info("✅ MCP Gateway shutdown completed")
            
        except Exception as e:
            logger.error(f"Gateway shutdown error: {e}")

# Global gateway instance
gateway = MCPGateway()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan manager"""
    # Startup
    await gateway.initialize()
    yield
    # Shutdown
    await gateway.shutdown()

# Create FastAPI application
app = FastAPI(
    title="MCP Gateway Server",
    description="Model Context Protocol Gateway for AI Assistant Integrations",
    version="1.0.0",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

# Add Prometheus metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# Security
security = HTTPBearer()

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Get current authenticated user"""
    try:
        payload = await gateway.auth_middleware.verify_token(credentials.credentials)
        return payload
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

@app.exception_handler(MCPException)
async def mcp_exception_handler(request: Request, exc: MCPException):
    """Handle MCP-specific exceptions"""
    gateway.metrics.increment_error_counter("mcp_exception", exc.__class__.__name__)
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.error_type, "message": exc.message, "details": exc.details}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions"""
    gateway.metrics.increment_error_counter("general_exception", exc.__class__.__name__)
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "internal_server_error", "message": "An internal error occurred"}
    )

# Health and status endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        health_status = await gateway.health_checker.check_health()
        return health_status
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Health check failed: {e}")

@app.get("/status")
async def gateway_status():
    """Gateway status endpoint"""
    return {
        "status": "running",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "active_connections": len(gateway.active_connections),
        "registered_servers": await gateway.router.get_registered_servers_count(),
        "uptime_seconds": time.time() - gateway.config.start_time
    }

# Authentication endpoints
@app.post("/auth/login")
async def login(credentials: Dict[str, str]):
    """Authenticate user and return JWT token"""
    try:
        username = credentials.get("username")
        password = credentials.get("password")
        
        if not username or not password:
            raise HTTPException(
                status_code=400,
                detail="Username and password required"
            )
        
        # Authenticate user (implement your authentication logic)
        auth_result = await gateway.auth_middleware.authenticate_user(username, password)
        
        if not auth_result.success:
            raise HTTPException(
                status_code=401,
                detail="Invalid credentials"
            )
        
        return {
            "access_token": auth_result.token,
            "token_type": "bearer",
            "expires_in": gateway.config.jwt_expires_in,
            "user_id": auth_result.user_id,
            "permissions": auth_result.permissions
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(status_code=500, detail="Authentication failed")

@app.post("/auth/refresh")
async def refresh_token(current_user: Dict = Depends(get_current_user)):
    """Refresh JWT token"""
    try:
        new_token = await gateway.jwt_handler.create_token(
            user_id=current_user["user_id"],
            permissions=current_user.get("permissions", [])
        )
        
        return {
            "access_token": new_token,
            "token_type": "bearer",
            "expires_in": gateway.config.jwt_expires_in
        }
        
    except Exception as e:
        logger.error(f"Token refresh error: {e}")
        raise HTTPException(status_code=500, detail="Token refresh failed")

# MCP Server management endpoints
@app.get("/servers")
async def list_servers(current_user: Dict = Depends(get_current_user)):
    """List registered MCP servers"""
    try:
        servers = await gateway.router.list_servers()
        return {"servers": servers}
    except Exception as e:
        logger.error(f"Failed to list servers: {e}")
        raise HTTPException(status_code=500, detail="Failed to list servers")

@app.post("/servers/register")
async def register_server(server_config: Dict[str, Any], current_user: Dict = Depends(get_current_user)):
    """Register new MCP server"""
    try:
        # Check permissions
        if "server_admin" not in current_user.get("permissions", []):
            raise HTTPException(status_code=403, detail="Insufficient permissions")
        
        server_id = await gateway.router.register_server(server_config)
        return {"server_id": server_id, "status": "registered"}
        
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Server registration failed: {e}")
        raise HTTPException(status_code=500, detail="Server registration failed")

@app.delete("/servers/{server_id}")
async def unregister_server(server_id: str, current_user: Dict = Depends(get_current_user)):
    """Unregister MCP server"""
    try:
        # Check permissions
        if "server_admin" not in current_user.get("permissions", []):
            raise HTTPException(status_code=403, detail="Insufficient permissions")
        
        await gateway.router.unregister_server(server_id)
        return {"server_id": server_id, "status": "unregistered"}
        
    except Exception as e:
        logger.error(f"Server unregistration failed: {e}")
        raise HTTPException(status_code=500, detail="Server unregistration failed")

# HTTP MCP endpoint
@app.post("/mcp/http")
async def handle_http_mcp(request: Request, current_user: Dict = Depends(get_current_user)):
    """Handle HTTP MCP requests"""
    try:
        # Parse request
        body = await request.body()
        message = json.loads(body)
        
        # Create HTTP transport
        transport = HTTPTransport(
            user_id=current_user["user_id"],
            permissions=current_user.get("permissions", [])
        )
        
        # Process message
        response = await gateway.protocol_handler.handle_message(message, transport)
        
        return response
        
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON")
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"HTTP MCP request failed: {e}")
        raise HTTPException(status_code=500, detail="Request processing failed")

# WebSocket MCP endpoint
@app.websocket("/mcp/ws")
async def websocket_mcp_endpoint(websocket: WebSocket):
    """Handle WebSocket MCP connections"""
    connection_id = None
    try:
        # Accept connection
        await websocket.accept()
        connection_id = f"ws_{int(time.time() * 1000)}_{id(websocket)}"
        
        # Store connection
        gateway.active_connections[connection_id] = websocket
        gateway.connection_metadata[connection_id] = {
            "connected_at": datetime.utcnow().isoformat(),
            "user_id": None,
            "authenticated": False
        }
        
        logger.info(f"WebSocket connection established: {connection_id}")
        
        # Create WebSocket transport
        transport = WebSocketTransport(
            websocket=websocket,
            connection_id=connection_id
        )
        
        # Handle messages
        while True:
            try:
                # Receive message
                data = await websocket.receive_text()
                message = json.loads(data)
                
                # Process message
                response = await gateway.protocol_handler.handle_message(message, transport)
                
                # Send response if available
                if response:
                    await websocket.send_text(json.dumps(response))
                
            except json.JSONDecodeError:
                error_response = {
                    "error": "invalid_json",
                    "message": "Invalid JSON format"
                }
                await websocket.send_text(json.dumps(error_response))
                
            except ValidationError as e:
                error_response = {
                    "error": "validation_error",
                    "message": str(e)
                }
                await websocket.send_text(json.dumps(error_response))
                
    except Exception as e:
        logger.error(f"WebSocket error for {connection_id}: {e}")
        
    finally:
        # Cleanup connection
        if connection_id:
            gateway.active_connections.pop(connection_id, None)
            gateway.connection_metadata.pop(connection_id, None)
            logger.info(f"WebSocket connection closed: {connection_id}")

# Development and testing endpoints
@app.get("/debug/connections")
async def debug_connections(current_user: Dict = Depends(get_current_user)):
    """Debug endpoint to show active connections"""
    if "debug" not in current_user.get("permissions", []):
        raise HTTPException(status_code=403, detail="Insufficient permissions")
    
    return {
        "active_connections": len(gateway.active_connections),
        "connections": [
            {
                "connection_id": conn_id,
                "metadata": metadata
            }
            for conn_id, metadata in gateway.connection_metadata.items()
        ]
    }

@app.get("/debug/metrics")
async def debug_metrics(current_user: Dict = Depends(get_current_user)):
    """Debug endpoint to show metrics"""
    if "debug" not in current_user.get("permissions", []):
        raise HTTPException(status_code=403, detail="Insufficient permissions")
    
    return await gateway.metrics.get_current_metrics()

def create_app() -> FastAPI:
    """Factory function to create FastAPI app"""
    return app

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="MCP Gateway Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    parser.add_argument("--log-level", default="info", help="Log level")
    
    args = parser.parse_args()
    
    # Update config
    gateway.config.host = args.host
    gateway.config.port = args.port
    gateway.config.log_level = args.log_level
    
    logger.info(f"Starting MCP Gateway Server on {args.host}:{args.port}")
    
    # Run server
    uvicorn.run(
        "src.mcp_gateway.main:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        reload=args.reload,
        log_level=args.log_level
    )

if __name__ == "__main__":
    main()