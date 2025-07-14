"""
MCP Protocol Handler
Implements the Model Context Protocol message handling and routing
"""

import asyncio
import logging
import time
import json
import uuid
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum

# Local imports
from ..routing.router import MCPRouter
from ..monitoring.metrics import MetricsCollector
from ..auth.middleware import AuthMiddleware
from ..utils.exceptions import MCPException, ValidationError, AuthenticationError
from .validation import MessageValidator
from .transport import Transport

logger = logging.getLogger(__name__)

class MCPMessageType(Enum):
    """MCP message types"""
    # Client requests
    INITIALIZE = "initialize"
    LIST_TOOLS = "tools/list"
    CALL_TOOL = "tools/call"
    LIST_RESOURCES = "resources/list"
    READ_RESOURCE = "resources/read"
    SUBSCRIBE_RESOURCE = "resources/subscribe"
    UNSUBSCRIBE_RESOURCE = "resources/unsubscribe"
    LIST_PROMPTS = "prompts/list"
    GET_PROMPT = "prompts/get"
    
    # Server responses
    RESULT = "result"
    ERROR = "error"
    
    # Notifications
    NOTIFICATION = "notification"
    RESOURCE_UPDATED = "resources/updated"
    RESOURCE_LIST_CHANGED = "resources/list_changed"
    TOOL_LIST_CHANGED = "tools/list_changed"
    PROMPT_LIST_CHANGED = "prompts/list_changed"

@dataclass
class MCPMessage:
    """MCP protocol message"""
    jsonrpc: str = "2.0"
    id: Optional[Union[str, int]] = None
    method: Optional[str] = None
    params: Optional[Dict[str, Any]] = None
    result: Optional[Any] = None
    error: Optional[Dict[str, Any]] = None

@dataclass
class MCPError:
    """MCP error object"""
    code: int
    message: str
    data: Optional[Any] = None

@dataclass
class MCPCapabilities:
    """MCP server/client capabilities"""
    experimental: Optional[Dict[str, Any]] = None
    sampling: Optional[Dict[str, Any]] = None
    tools: Optional[Dict[str, Any]] = None
    resources: Optional[Dict[str, Any]] = None
    prompts: Optional[Dict[str, Any]] = None

@dataclass
class MCPTool:
    """MCP tool definition"""
    name: str
    description: str
    input_schema: Dict[str, Any]

@dataclass
class MCPResource:
    """MCP resource definition"""
    uri: str
    name: str
    description: Optional[str] = None
    mime_type: Optional[str] = None

@dataclass
class MCPPrompt:
    """MCP prompt definition"""
    name: str
    description: str
    arguments: Optional[List[Dict[str, Any]]] = None

class MCPProtocolHandler:
    """Main MCP protocol handler"""
    
    def __init__(self, 
                 router: MCPRouter,
                 metrics: MetricsCollector,
                 auth_middleware: AuthMiddleware):
        self.router = router
        self.metrics = metrics
        self.auth_middleware = auth_middleware
        self.validator = MessageValidator()
        
        # Session management
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.subscriptions: Dict[str, List[str]] = {}  # connection_id -> resource URIs
        
        # Request tracking
        self.pending_requests: Dict[str, Dict[str, Any]] = {}
        
        logger.info("MCP Protocol Handler initialized")
    
    async def handle_message(self, 
                           message: Dict[str, Any], 
                           transport: Transport) -> Optional[Dict[str, Any]]:
        """Handle incoming MCP message"""
        try:
            start_time = time.time()
            
            # Validate message format
            validated_message = self.validator.validate_message(message)
            
            # Track metrics
            self.metrics.increment_message_counter(
                validated_message.method or "unknown",
                transport.transport_type
            )
            
            # Route message based on type
            if validated_message.method:
                # This is a request
                response = await self._handle_request(validated_message, transport)
            elif validated_message.result is not None or validated_message.error is not None:
                # This is a response
                response = await self._handle_response(validated_message, transport)
            else:
                raise ValidationError("Invalid message: missing method or result/error")
            
            # Track processing time
            processing_time = (time.time() - start_time) * 1000
            self.metrics.record_processing_time(
                validated_message.method or "response",
                processing_time
            )
            
            return response
            
        except ValidationError as e:
            logger.warning(f"Message validation failed: {e}")
            return self._create_error_response(message.get("id"), -32602, str(e))
            
        except AuthenticationError as e:
            logger.warning(f"Authentication failed: {e}")
            return self._create_error_response(message.get("id"), -32001, "Authentication required")
            
        except Exception as e:
            logger.error(f"Message handling failed: {e}")
            self.metrics.increment_error_counter("message_handling", e.__class__.__name__)
            return self._create_error_response(
                message.get("id"), 
                -32603, 
                "Internal error"
            )
    
    async def _handle_request(self, 
                            message: MCPMessage, 
                            transport: Transport) -> Optional[Dict[str, Any]]:
        """Handle MCP request message"""
        try:
            method = message.method
            params = message.params or {}
            
            # Check authentication for protected methods
            if method != MCPMessageType.INITIALIZE.value:
                await self._check_authentication(transport)
            
            # Route to appropriate handler
            if method == MCPMessageType.INITIALIZE.value:
                result = await self._handle_initialize(params, transport)
            elif method == MCPMessageType.LIST_TOOLS.value:
                result = await self._handle_list_tools(params, transport)
            elif method == MCPMessageType.CALL_TOOL.value:
                result = await self._handle_call_tool(params, transport)
            elif method == MCPMessageType.LIST_RESOURCES.value:
                result = await self._handle_list_resources(params, transport)
            elif method == MCPMessageType.READ_RESOURCE.value:
                result = await self._handle_read_resource(params, transport)
            elif method == MCPMessageType.SUBSCRIBE_RESOURCE.value:
                result = await self._handle_subscribe_resource(params, transport)
            elif method == MCPMessageType.UNSUBSCRIBE_RESOURCE.value:
                result = await self._handle_unsubscribe_resource(params, transport)
            elif method == MCPMessageType.LIST_PROMPTS.value:
                result = await self._handle_list_prompts(params, transport)
            elif method == MCPMessageType.GET_PROMPT.value:
                result = await self._handle_get_prompt(params, transport)
            else:
                raise ValidationError(f"Unknown method: {method}")
            
            # Create success response
            if message.id is not None:
                return {
                    "jsonrpc": "2.0",
                    "id": message.id,
                    "result": result
                }
            else:
                # No response needed for notifications
                return None
                
        except MCPException as e:
            if message.id is not None:
                return self._create_error_response(message.id, e.error_code, e.message, e.details)
            return None
            
        except Exception as e:
            logger.error(f"Request handling failed: {e}")
            if message.id is not None:
                return self._create_error_response(message.id, -32603, "Internal error")
            return None
    
    async def _handle_response(self, 
                             message: MCPMessage, 
                             transport: Transport) -> Optional[Dict[str, Any]]:
        """Handle MCP response message"""
        try:
            request_id = message.id
            
            if request_id in self.pending_requests:
                # Complete pending request
                request_info = self.pending_requests.pop(request_id)
                
                # Notify waiting coroutine if any
                if "future" in request_info:
                    future = request_info["future"]
                    if not future.done():
                        future.set_result(message)
                
                logger.debug(f"Completed request {request_id}")
            else:
                logger.warning(f"Received response for unknown request: {request_id}")
            
            # Responses don't generate responses
            return None
            
        except Exception as e:
            logger.error(f"Response handling failed: {e}")
            return None
    
    async def _check_authentication(self, transport: Transport):
        """Check if transport is authenticated"""
        if not transport.is_authenticated():
            # Try to authenticate with token if available
            token = transport.get_auth_token()
            if token:
                try:
                    payload = await self.auth_middleware.verify_token(token)
                    transport.set_user_info(
                        user_id=payload["user_id"],
                        permissions=payload.get("permissions", [])
                    )
                except Exception as e:
                    raise AuthenticationError("Invalid authentication token")
            else:
                raise AuthenticationError("Authentication required")
    
    async def _handle_initialize(self, 
                               params: Dict[str, Any], 
                               transport: Transport) -> Dict[str, Any]:
        """Handle initialize request"""
        try:
            # Extract client info
            protocol_version = params.get("protocolVersion", "2024-11-05")
            client_info = params.get("clientInfo", {})
            capabilities = params.get("capabilities", {})
            
            # Validate protocol version
            if protocol_version != "2024-11-05":
                raise ValidationError(f"Unsupported protocol version: {protocol_version}")
            
            # Create session
            session_id = str(uuid.uuid4())
            session_info = {
                "session_id": session_id,
                "client_info": client_info,
                "capabilities": capabilities,
                "created_at": datetime.utcnow().isoformat(),
                "transport_type": transport.transport_type,
                "authenticated": False
            }
            
            # Store session
            transport_id = transport.get_transport_id()
            self.sessions[transport_id] = session_info
            
            logger.info(f"MCP session initialized: {session_id}")
            
            # Return server capabilities
            return {
                "protocolVersion": "2024-11-05",
                "serverInfo": {
                    "name": "MCP Gateway",
                    "version": "1.0.0",
                    "description": "Model Context Protocol Gateway Server"
                },
                "capabilities": {
                    "tools": {
                        "listChanged": True
                    },
                    "resources": {
                        "subscribe": True,
                        "listChanged": True
                    },
                    "prompts": {
                        "listChanged": True
                    },
                    "experimental": {}
                }
            }
            
        except Exception as e:
            logger.error(f"Initialize failed: {e}")
            raise MCPException("Initialize failed", -32000, str(e))
    
    async def _handle_list_tools(self, 
                                params: Dict[str, Any], 
                                transport: Transport) -> Dict[str, Any]:
        """Handle tools/list request"""
        try:
            # Get available tools from all registered servers
            tools = await self.router.list_tools(transport.get_user_id())
            
            # Convert to MCP format
            mcp_tools = []
            for tool in tools:
                mcp_tools.append({
                    "name": tool["name"],
                    "description": tool["description"],
                    "inputSchema": tool.get("input_schema", {})
                })
            
            return {"tools": mcp_tools}
            
        except Exception as e:
            logger.error(f"List tools failed: {e}")
            raise MCPException("Failed to list tools", -32000, str(e))
    
    async def _handle_call_tool(self, 
                               params: Dict[str, Any], 
                               transport: Transport) -> Dict[str, Any]:
        """Handle tools/call request"""
        try:
            tool_name = params.get("name")
            arguments = params.get("arguments", {})
            
            if not tool_name:
                raise ValidationError("Tool name is required")
            
            # Route tool call to appropriate server
            result = await self.router.call_tool(
                tool_name=tool_name,
                arguments=arguments,
                user_id=transport.get_user_id(),
                permissions=transport.get_permissions()
            )
            
            return {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(result, indent=2)
                    }
                ]
            }
            
        except Exception as e:
            logger.error(f"Tool call failed: {e}")
            raise MCPException("Tool call failed", -32000, str(e))
    
    async def _handle_list_resources(self, 
                                   params: Dict[str, Any], 
                                   transport: Transport) -> Dict[str, Any]:
        """Handle resources/list request"""
        try:
            # Get available resources from all registered servers
            resources = await self.router.list_resources(transport.get_user_id())
            
            # Convert to MCP format
            mcp_resources = []
            for resource in resources:
                mcp_resources.append({
                    "uri": resource["uri"],
                    "name": resource["name"],
                    "description": resource.get("description"),
                    "mimeType": resource.get("mime_type")
                })
            
            return {"resources": mcp_resources}
            
        except Exception as e:
            logger.error(f"List resources failed: {e}")
            raise MCPException("Failed to list resources", -32000, str(e))
    
    async def _handle_read_resource(self, 
                                  params: Dict[str, Any], 
                                  transport: Transport) -> Dict[str, Any]:
        """Handle resources/read request"""
        try:
            uri = params.get("uri")
            
            if not uri:
                raise ValidationError("Resource URI is required")
            
            # Route resource read to appropriate server
            content = await self.router.read_resource(
                uri=uri,
                user_id=transport.get_user_id(),
                permissions=transport.get_permissions()
            )
            
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
            
        except Exception as e:
            logger.error(f"Resource read failed: {e}")
            raise MCPException("Resource read failed", -32000, str(e))
    
    async def _handle_subscribe_resource(self, 
                                       params: Dict[str, Any], 
                                       transport: Transport) -> Dict[str, Any]:
        """Handle resources/subscribe request"""
        try:
            uri = params.get("uri")
            
            if not uri:
                raise ValidationError("Resource URI is required")
            
            # Add subscription
            transport_id = transport.get_transport_id()
            if transport_id not in self.subscriptions:
                self.subscriptions[transport_id] = []
            
            if uri not in self.subscriptions[transport_id]:
                self.subscriptions[transport_id].append(uri)
            
            # Register with router
            await self.router.subscribe_resource(
                uri=uri,
                transport_id=transport_id,
                user_id=transport.get_user_id()
            )
            
            logger.info(f"Subscribed to resource: {uri}")
            
            return {}
            
        except Exception as e:
            logger.error(f"Resource subscription failed: {e}")
            raise MCPException("Resource subscription failed", -32000, str(e))
    
    async def _handle_unsubscribe_resource(self, 
                                         params: Dict[str, Any], 
                                         transport: Transport) -> Dict[str, Any]:
        """Handle resources/unsubscribe request"""
        try:
            uri = params.get("uri")
            
            if not uri:
                raise ValidationError("Resource URI is required")
            
            # Remove subscription
            transport_id = transport.get_transport_id()
            if transport_id in self.subscriptions:
                if uri in self.subscriptions[transport_id]:
                    self.subscriptions[transport_id].remove(uri)
            
            # Unregister with router
            await self.router.unsubscribe_resource(
                uri=uri,
                transport_id=transport_id
            )
            
            logger.info(f"Unsubscribed from resource: {uri}")
            
            return {}
            
        except Exception as e:
            logger.error(f"Resource unsubscription failed: {e}")
            raise MCPException("Resource unsubscription failed", -32000, str(e))
    
    async def _handle_list_prompts(self, 
                                 params: Dict[str, Any], 
                                 transport: Transport) -> Dict[str, Any]:
        """Handle prompts/list request"""
        try:
            # Get available prompts from all registered servers
            prompts = await self.router.list_prompts(transport.get_user_id())
            
            # Convert to MCP format
            mcp_prompts = []
            for prompt in prompts:
                mcp_prompts.append({
                    "name": prompt["name"],
                    "description": prompt["description"],
                    "arguments": prompt.get("arguments", [])
                })
            
            return {"prompts": mcp_prompts}
            
        except Exception as e:
            logger.error(f"List prompts failed: {e}")
            raise MCPException("Failed to list prompts", -32000, str(e))
    
    async def _handle_get_prompt(self, 
                               params: Dict[str, Any], 
                               transport: Transport) -> Dict[str, Any]:
        """Handle prompts/get request"""
        try:
            prompt_name = params.get("name")
            arguments = params.get("arguments", {})
            
            if not prompt_name:
                raise ValidationError("Prompt name is required")
            
            # Route prompt request to appropriate server
            result = await self.router.get_prompt(
                prompt_name=prompt_name,
                arguments=arguments,
                user_id=transport.get_user_id(),
                permissions=transport.get_permissions()
            )
            
            return {
                "description": result.get("description", ""),
                "messages": result.get("messages", [])
            }
            
        except Exception as e:
            logger.error(f"Get prompt failed: {e}")
            raise MCPException("Get prompt failed", -32000, str(e))
    
    def _create_error_response(self, 
                             request_id: Optional[Union[str, int]], 
                             code: int, 
                             message: str, 
                             data: Any = None) -> Dict[str, Any]:
        """Create MCP error response"""
        error_response = {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {
                "code": code,
                "message": message
            }
        }
        
        if data is not None:
            error_response["error"]["data"] = data
        
        return error_response
    
    async def send_notification(self, 
                              transport_id: str, 
                              method: str, 
                              params: Dict[str, Any]):
        """Send notification to client"""
        try:
            # This would be implemented by the transport layer
            # For now, just log the notification
            logger.info(f"Notification {method} for {transport_id}: {params}")
            
        except Exception as e:
            logger.error(f"Failed to send notification: {e}")
    
    async def cleanup_transport(self, transport_id: str):
        """Cleanup resources for disconnected transport"""
        try:
            # Remove session
            self.sessions.pop(transport_id, None)
            
            # Remove subscriptions
            self.subscriptions.pop(transport_id, None)
            
            # Cleanup pending requests
            to_remove = []
            for req_id, req_info in self.pending_requests.items():
                if req_info.get("transport_id") == transport_id:
                    to_remove.append(req_id)
            
            for req_id in to_remove:
                self.pending_requests.pop(req_id, None)
            
            logger.info(f"Cleaned up transport: {transport_id}")
            
        except Exception as e:
            logger.error(f"Transport cleanup failed: {e}")