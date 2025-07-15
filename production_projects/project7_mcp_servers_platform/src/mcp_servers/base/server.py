"""
Base MCP Server Implementation
Provides foundation for building custom MCP servers
"""

import asyncio
import logging
import json
import time
import uuid
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum

import uvicorn
from fastapi import FastAPI, WebSocket, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class MCPServerCapability(Enum):
    """MCP Server capabilities"""
    TOOLS = "tools"
    RESOURCES = "resources"
    PROMPTS = "prompts"
    SAMPLING = "sampling"

@dataclass
class MCPTool:
    """MCP Tool definition"""
    name: str
    description: str
    input_schema: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.input_schema
        }

@dataclass
class MCPResource:
    """MCP Resource definition"""
    uri: str
    name: str
    description: Optional[str] = None
    mime_type: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "uri": self.uri,
            "name": self.name
        }
        if self.description:
            result["description"] = self.description
        if self.mime_type:
            result["mimeType"] = self.mime_type
        return result

@dataclass
class MCPPrompt:
    """MCP Prompt definition"""
    name: str
    description: str
    arguments: Optional[List[Dict[str, Any]]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "name": self.name,
            "description": self.description
        }
        if self.arguments:
            result["arguments"] = self.arguments
        return result

@dataclass
class MCPServerInfo:
    """MCP Server information"""
    name: str
    version: str
    description: Optional[str] = None
    author: Optional[str] = None
    license: Optional[str] = None
    homepage: Optional[str] = None

class BaseMCPServer(ABC):
    """Base class for MCP servers"""
    
    def __init__(self, 
                 server_info: MCPServerInfo,
                 capabilities: Optional[List[MCPServerCapability]] = None):
        self.server_info = server_info
        self.capabilities = capabilities or []
        self.tools: Dict[str, MCPTool] = {}
        self.resources: Dict[str, MCPResource] = {}
        self.prompts: Dict[str, MCPPrompt] = {}
        self.app = None
        self.active_connections: Dict[str, WebSocket] = {}
        
        # Metrics
        self.start_time = time.time()
        self.request_count = 0
        self.error_count = 0
        
        # Initialize server
        self._initialize_server()
        self._register_tools()
        self._register_resources()
        self._register_prompts()
        
        logger.info(f"MCP Server '{self.server_info.name}' initialized")
    
    def _initialize_server(self):
        """Initialize FastAPI server"""
        self.app = FastAPI(
            title=self.server_info.name,
            description=self.server_info.description or "MCP Server",
            version=self.server_info.version
        )
        
        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Add routes
        self._setup_routes()
    
    def _setup_routes(self):
        """Setup HTTP routes"""
        
        @self.app.get("/mcp/info")
        async def get_server_info():
            """Get server information"""
            return {
                "serverInfo": {
                    "name": self.server_info.name,
                    "version": self.server_info.version,
                    "description": self.server_info.description,
                    "author": self.server_info.author,
                    "license": self.server_info.license,
                    "homepage": self.server_info.homepage
                },
                "capabilities": {
                    cap.value: True for cap in self.capabilities
                },
                "protocolVersion": "2024-11-05"
            }
        
        @self.app.get("/mcp/tools")
        async def list_tools():
            """List available tools"""
            return {
                "tools": [tool.to_dict() for tool in self.tools.values()]
            }
        
        @self.app.post("/mcp/tools/call")
        async def call_tool(request: Request):
            """Call a tool"""
            try:
                body = await request.json()
                tool_name = body.get("name")
                arguments = body.get("arguments", {})
                
                if not tool_name:
                    raise HTTPException(status_code=400, detail="Tool name is required")
                
                if tool_name not in self.tools:
                    raise HTTPException(status_code=404, detail=f"Tool '{tool_name}' not found")
                
                # Call the tool
                result = await self.call_tool(tool_name, arguments)
                
                return {
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(result, indent=2)
                        }
                    ]
                }
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Tool call failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/mcp/resources")
        async def list_resources():
            """List available resources"""
            return {
                "resources": [resource.to_dict() for resource in self.resources.values()]
            }
        
        @self.app.post("/mcp/resources/read")
        async def read_resource(request: Request):
            """Read a resource"""
            try:
                body = await request.json()
                uri = body.get("uri")
                
                if not uri:
                    raise HTTPException(status_code=400, detail="Resource URI is required")
                
                # Find resource by URI
                resource = None
                for res in self.resources.values():
                    if res.uri == uri:
                        resource = res
                        break
                
                if not resource:
                    raise HTTPException(status_code=404, detail=f"Resource '{uri}' not found")
                
                # Read the resource
                content = await self.read_resource(uri)
                
                return {
                    "contents": [
                        {
                            "uri": uri,
                            "mimeType": content.get("mime_type", "text/plain"),
                            "text": content.get("text") if "text" in content else None,
                            "blob": content.get("blob") if "blob" in content else None
                        }
                    ]
                }
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Resource read failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/mcp/prompts")
        async def list_prompts():
            """List available prompts"""
            return {
                "prompts": [prompt.to_dict() for prompt in self.prompts.values()]
            }
        
        @self.app.post("/mcp/prompts/get")
        async def get_prompt(request: Request):
            """Get a prompt"""
            try:
                body = await request.json()
                prompt_name = body.get("name")
                arguments = body.get("arguments", {})
                
                if not prompt_name:
                    raise HTTPException(status_code=400, detail="Prompt name is required")
                
                if prompt_name not in self.prompts:
                    raise HTTPException(status_code=404, detail=f"Prompt '{prompt_name}' not found")
                
                # Get the prompt
                result = await self.get_prompt(prompt_name, arguments)
                
                return result
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Get prompt failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint"""
            return {
                "status": "healthy",
                "timestamp": datetime.utcnow().isoformat(),
                "uptime_seconds": time.time() - self.start_time,
                "request_count": self.request_count,
                "error_count": self.error_count,
                "active_connections": len(self.active_connections)
            }
        
        @self.app.websocket("/mcp/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket endpoint for MCP protocol"""
            connection_id = f"ws_{int(time.time() * 1000)}_{id(websocket)}"
            
            try:
                await websocket.accept()
                self.active_connections[connection_id] = websocket
                logger.info(f"WebSocket connection established: {connection_id}")
                
                while True:
                    try:
                        # Receive message
                        data = await websocket.receive_text()
                        message = json.loads(data)
                        
                        # Process message
                        response = await self._handle_websocket_message(message, connection_id)
                        
                        # Send response if available
                        if response:
                            await websocket.send_text(json.dumps(response))
                        
                    except json.JSONDecodeError:
                        error_response = {
                            "jsonrpc": "2.0",
                            "id": None,
                            "error": {
                                "code": -32700,
                                "message": "Parse error"
                            }
                        }
                        await websocket.send_text(json.dumps(error_response))
                        
            except Exception as e:
                logger.error(f"WebSocket error for {connection_id}: {e}")
                
            finally:
                # Cleanup connection
                self.active_connections.pop(connection_id, None)
                logger.info(f"WebSocket connection closed: {connection_id}")
    
    async def _handle_websocket_message(self, message: Dict[str, Any], connection_id: str) -> Optional[Dict[str, Any]]:
        """Handle WebSocket MCP message"""
        try:
            method = message.get("method")
            params = message.get("params", {})
            message_id = message.get("id")
            
            if method == "initialize":
                result = await self._handle_initialize(params)
            elif method == "tools/list":
                result = {"tools": [tool.to_dict() for tool in self.tools.values()]}
            elif method == "tools/call":
                tool_name = params.get("name")
                arguments = params.get("arguments", {})
                tool_result = await self.call_tool(tool_name, arguments)
                result = {
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(tool_result, indent=2)
                        }
                    ]
                }
            elif method == "resources/list":
                result = {"resources": [resource.to_dict() for resource in self.resources.values()]}
            elif method == "resources/read":
                uri = params.get("uri")
                content = await self.read_resource(uri)
                result = {
                    "contents": [
                        {
                            "uri": uri,
                            "mimeType": content.get("mime_type", "text/plain"),
                            "text": content.get("text") if "text" in content else None,
                            "blob": content.get("blob") if "blob" in content else None
                        }
                    ]
                }
            elif method == "prompts/list":
                result = {"prompts": [prompt.to_dict() for prompt in self.prompts.values()]}
            elif method == "prompts/get":
                prompt_name = params.get("name")
                arguments = params.get("arguments", {})
                result = await self.get_prompt(prompt_name, arguments)
            else:
                raise ValueError(f"Unknown method: {method}")
            
            # Create success response
            if message_id is not None:
                return {
                    "jsonrpc": "2.0",
                    "id": message_id,
                    "result": result
                }
            
            return None
            
        except Exception as e:
            logger.error(f"WebSocket message handling failed: {e}")
            if message_id is not None:
                return {
                    "jsonrpc": "2.0",
                    "id": message_id,
                    "error": {
                        "code": -32603,
                        "message": str(e)
                    }
                }
            return None
    
    async def _handle_initialize(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle initialize request"""
        return {
            "protocolVersion": "2024-11-05",
            "serverInfo": {
                "name": self.server_info.name,
                "version": self.server_info.version,
                "description": self.server_info.description
            },
            "capabilities": {
                cap.value: True for cap in self.capabilities
            }
        }
    
    # Abstract methods to be implemented by subclasses
    
    @abstractmethod
    def _register_tools(self):
        """Register tools provided by this server"""
        pass
    
    @abstractmethod
    def _register_resources(self):
        """Register resources provided by this server"""
        pass
    
    @abstractmethod
    def _register_prompts(self):
        """Register prompts provided by this server"""
        pass
    
    @abstractmethod
    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        """Call a tool by name with given arguments"""
        pass
    
    @abstractmethod
    async def read_resource(self, uri: str) -> Dict[str, Any]:
        """Read a resource by URI"""
        pass
    
    @abstractmethod
    async def get_prompt(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Get a prompt by name with given arguments"""
        pass
    
    # Helper methods for subclasses
    
    def register_tool(self, tool: MCPTool):
        """Register a tool"""
        self.tools[tool.name] = tool
        logger.info(f"Registered tool: {tool.name}")
    
    def register_resource(self, resource: MCPResource):
        """Register a resource"""
        self.resources[resource.uri] = resource
        logger.info(f"Registered resource: {resource.uri}")
    
    def register_prompt(self, prompt: MCPPrompt):
        """Register a prompt"""
        self.prompts[prompt.name] = prompt
        logger.info(f"Registered prompt: {prompt.name}")
    
    def get_tool(self, name: str) -> Optional[MCPTool]:
        """Get a tool by name"""
        return self.tools.get(name)
    
    def get_resource(self, uri: str) -> Optional[MCPResource]:
        """Get a resource by URI"""
        return self.resources.get(uri)
    
    def get_prompt_definition(self, name: str) -> Optional[MCPPrompt]:
        """Get a prompt definition by name"""
        return self.prompts.get(name)
    
    def run(self, host: str = "0.0.0.0", port: int = 8001, **kwargs):
        """Run the MCP server"""
        logger.info(f"Starting MCP server '{self.server_info.name}' on {host}:{port}")
        
        uvicorn.run(
            self.app,
            host=host,
            port=port,
            **kwargs
        )
    
    async def broadcast_notification(self, method: str, params: Dict[str, Any]):
        """Broadcast notification to all connected clients"""
        notification = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params
        }
        
        message = json.dumps(notification)
        
        # Send to all active WebSocket connections
        disconnected = []
        for connection_id, websocket in self.active_connections.items():
            try:
                await websocket.send_text(message)
            except Exception as e:
                logger.warning(f"Failed to send notification to {connection_id}: {e}")
                disconnected.append(connection_id)
        
        # Remove disconnected connections
        for connection_id in disconnected:
            self.active_connections.pop(connection_id, None)
    
    async def send_notification(self, connection_id: str, method: str, params: Dict[str, Any]):
        """Send notification to specific client"""
        if connection_id not in self.active_connections:
            logger.warning(f"Connection {connection_id} not found")
            return
        
        notification = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params
        }
        
        try:
            websocket = self.active_connections[connection_id]
            await websocket.send_text(json.dumps(notification))
        except Exception as e:
            logger.error(f"Failed to send notification to {connection_id}: {e}")
            # Remove disconnected connection
            self.active_connections.pop(connection_id, None)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get server statistics"""
        return {
            "server_info": {
                "name": self.server_info.name,
                "version": self.server_info.version,
                "description": self.server_info.description
            },
            "capabilities": [cap.value for cap in self.capabilities],
            "tools_count": len(self.tools),
            "resources_count": len(self.resources),
            "prompts_count": len(self.prompts),
            "active_connections": len(self.active_connections),
            "uptime_seconds": time.time() - self.start_time,
            "request_count": self.request_count,
            "error_count": self.error_count
        }