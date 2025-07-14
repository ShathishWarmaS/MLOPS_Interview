"""
MCP Message Validation
Validates MCP protocol messages according to the specification
"""

import json
import logging
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
import jsonschema
from jsonschema import validate, ValidationError as JsonSchemaValidationError

from ..utils.exceptions import ValidationError

logger = logging.getLogger(__name__)

@dataclass
class MCPMessage:
    """Validated MCP message"""
    jsonrpc: str
    id: Optional[Union[str, int]] = None
    method: Optional[str] = None
    params: Optional[Dict[str, Any]] = None
    result: Optional[Any] = None
    error: Optional[Dict[str, Any]] = None

class MessageValidator:
    """MCP message validator"""
    
    def __init__(self):
        self.schemas = self._load_schemas()
        logger.info("MCP message validator initialized")
    
    def _load_schemas(self) -> Dict[str, Dict[str, Any]]:
        """Load JSON schemas for MCP messages"""
        
        # Base JSON-RPC schema
        base_schema = {
            "type": "object",
            "properties": {
                "jsonrpc": {"type": "string", "enum": ["2.0"]},
                "id": {"oneOf": [{"type": "string"}, {"type": "number"}, {"type": "null"}]}
            },
            "required": ["jsonrpc"]
        }
        
        # Request schema
        request_schema = {
            "allOf": [
                base_schema,
                {
                    "type": "object",
                    "properties": {
                        "method": {"type": "string"},
                        "params": {"type": "object"}
                    },
                    "required": ["method"]
                }
            ]
        }
        
        # Response schema
        response_schema = {
            "allOf": [
                base_schema,
                {
                    "type": "object",
                    "oneOf": [
                        {"properties": {"result": {}}, "required": ["result"]},
                        {
                            "properties": {
                                "error": {
                                    "type": "object",
                                    "properties": {
                                        "code": {"type": "integer"},
                                        "message": {"type": "string"},
                                        "data": {}
                                    },
                                    "required": ["code", "message"]
                                }
                            },
                            "required": ["error"]
                        }
                    ]
                }
            ]
        }
        
        # Notification schema (request without id)
        notification_schema = {
            "allOf": [
                {
                    "type": "object",
                    "properties": {
                        "jsonrpc": {"type": "string", "enum": ["2.0"]},
                        "method": {"type": "string"},
                        "params": {"type": "object"}
                    },
                    "required": ["jsonrpc", "method"]
                }
            ]
        }
        
        # Initialize method schemas
        initialize_schema = {
            "type": "object",
            "properties": {
                "protocolVersion": {"type": "string"},
                "clientInfo": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "version": {"type": "string"}
                    },
                    "required": ["name", "version"]
                },
                "capabilities": {"type": "object"}
            },
            "required": ["protocolVersion", "clientInfo"]
        }
        
        # Tool call schema
        tool_call_schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "arguments": {"type": "object"}
            },
            "required": ["name"]
        }
        
        # Resource read schema
        resource_read_schema = {
            "type": "object",
            "properties": {
                "uri": {"type": "string"}
            },
            "required": ["uri"]
        }
        
        # Resource subscribe schema
        resource_subscribe_schema = {
            "type": "object",
            "properties": {
                "uri": {"type": "string"}
            },
            "required": ["uri"]
        }
        
        # Prompt get schema
        prompt_get_schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "arguments": {"type": "object"}
            },
            "required": ["name"]
        }
        
        return {
            "base": base_schema,
            "request": request_schema,
            "response": response_schema,
            "notification": notification_schema,
            "initialize": initialize_schema,
            "tool_call": tool_call_schema,
            "resource_read": resource_read_schema,
            "resource_subscribe": resource_subscribe_schema,
            "prompt_get": prompt_get_schema
        }
    
    def validate_message(self, message: Dict[str, Any]) -> MCPMessage:
        """Validate MCP message and return typed object"""
        try:
            # Basic structure validation
            if not isinstance(message, dict):
                raise ValidationError("Message must be a JSON object")
            
            # Check JSON-RPC version
            if message.get("jsonrpc") != "2.0":
                raise ValidationError("Invalid JSON-RPC version, must be '2.0'")
            
            # Determine message type and validate
            if "method" in message:
                # This is a request or notification
                if "id" in message:
                    # Request
                    self._validate_against_schema(message, "request")
                    return self._validate_request(message)
                else:
                    # Notification
                    self._validate_against_schema(message, "notification")
                    return self._validate_notification(message)
            elif "result" in message or "error" in message:
                # This is a response
                self._validate_against_schema(message, "response")
                return self._validate_response(message)
            else:
                raise ValidationError("Invalid message: missing method or result/error")
                
        except JsonSchemaValidationError as e:
            raise ValidationError(f"Schema validation failed: {e.message}")
        except Exception as e:
            if isinstance(e, ValidationError):
                raise
            raise ValidationError(f"Message validation failed: {str(e)}")
    
    def _validate_against_schema(self, message: Dict[str, Any], schema_name: str):
        """Validate message against JSON schema"""
        try:
            schema = self.schemas.get(schema_name)
            if schema:
                validate(instance=message, schema=schema)
        except JsonSchemaValidationError as e:
            raise ValidationError(f"Schema validation failed for {schema_name}: {e.message}")
    
    def _validate_request(self, message: Dict[str, Any]) -> MCPMessage:
        """Validate and parse request message"""
        method = message["method"]
        params = message.get("params", {})
        
        # Validate method-specific parameters
        if method == "initialize":
            self._validate_against_schema(params, "initialize")
        elif method == "tools/call":
            self._validate_against_schema(params, "tool_call")
        elif method == "resources/read":
            self._validate_against_schema(params, "resource_read")
        elif method in ["resources/subscribe", "resources/unsubscribe"]:
            self._validate_against_schema(params, "resource_subscribe")
        elif method == "prompts/get":
            self._validate_against_schema(params, "prompt_get")
        
        # Additional method-specific validation
        self._validate_method_params(method, params)
        
        return MCPMessage(
            jsonrpc=message["jsonrpc"],
            id=message["id"],
            method=method,
            params=params
        )
    
    def _validate_notification(self, message: Dict[str, Any]) -> MCPMessage:
        """Validate and parse notification message"""
        method = message["method"]
        params = message.get("params", {})
        
        # Additional validation for notifications
        self._validate_method_params(method, params)
        
        return MCPMessage(
            jsonrpc=message["jsonrpc"],
            method=method,
            params=params
        )
    
    def _validate_response(self, message: Dict[str, Any]) -> MCPMessage:
        """Validate and parse response message"""
        if "id" not in message:
            raise ValidationError("Response must include id")
        
        result = message.get("result")
        error = message.get("error")
        
        # Validate error format if present
        if error is not None:
            if not isinstance(error, dict):
                raise ValidationError("Error must be an object")
            if "code" not in error or "message" not in error:
                raise ValidationError("Error must include code and message")
            if not isinstance(error["code"], int):
                raise ValidationError("Error code must be an integer")
            if not isinstance(error["message"], str):
                raise ValidationError("Error message must be a string")
        
        return MCPMessage(
            jsonrpc=message["jsonrpc"],
            id=message["id"],
            result=result,
            error=error
        )
    
    def _validate_method_params(self, method: str, params: Dict[str, Any]):
        """Additional method-specific parameter validation"""
        try:
            if method == "initialize":
                self._validate_initialize_params(params)
            elif method == "tools/call":
                self._validate_tool_call_params(params)
            elif method == "resources/read":
                self._validate_resource_read_params(params)
            elif method in ["resources/subscribe", "resources/unsubscribe"]:
                self._validate_resource_subscribe_params(params)
            elif method == "prompts/get":
                self._validate_prompt_get_params(params)
            
        except Exception as e:
            raise ValidationError(f"Parameter validation failed for {method}: {str(e)}")
    
    def _validate_initialize_params(self, params: Dict[str, Any]):
        """Validate initialize parameters"""
        protocol_version = params.get("protocolVersion")
        
        # Check supported protocol versions
        supported_versions = ["2024-11-05"]
        if protocol_version not in supported_versions:
            raise ValidationError(f"Unsupported protocol version: {protocol_version}")
        
        # Validate client info
        client_info = params.get("clientInfo", {})
        if not client_info.get("name"):
            raise ValidationError("Client name is required")
        if not client_info.get("version"):
            raise ValidationError("Client version is required")
    
    def _validate_tool_call_params(self, params: Dict[str, Any]):
        """Validate tool call parameters"""
        tool_name = params.get("name")
        
        if not tool_name:
            raise ValidationError("Tool name is required")
        
        if not isinstance(tool_name, str):
            raise ValidationError("Tool name must be a string")
        
        # Validate tool name format
        if not tool_name.replace("_", "").replace("-", "").isalnum():
            raise ValidationError("Tool name must contain only alphanumeric characters, hyphens, and underscores")
        
        # Validate arguments
        arguments = params.get("arguments", {})
        if not isinstance(arguments, dict):
            raise ValidationError("Tool arguments must be an object")
    
    def _validate_resource_read_params(self, params: Dict[str, Any]):
        """Validate resource read parameters"""
        uri = params.get("uri")
        
        if not uri:
            raise ValidationError("Resource URI is required")
        
        if not isinstance(uri, str):
            raise ValidationError("Resource URI must be a string")
        
        # Basic URI format validation
        if not uri.strip():
            raise ValidationError("Resource URI cannot be empty")
    
    def _validate_resource_subscribe_params(self, params: Dict[str, Any]):
        """Validate resource subscribe/unsubscribe parameters"""
        uri = params.get("uri")
        
        if not uri:
            raise ValidationError("Resource URI is required")
        
        if not isinstance(uri, str):
            raise ValidationError("Resource URI must be a string")
        
        # Basic URI format validation
        if not uri.strip():
            raise ValidationError("Resource URI cannot be empty")
    
    def _validate_prompt_get_params(self, params: Dict[str, Any]):
        """Validate prompt get parameters"""
        prompt_name = params.get("name")
        
        if not prompt_name:
            raise ValidationError("Prompt name is required")
        
        if not isinstance(prompt_name, str):
            raise ValidationError("Prompt name must be a string")
        
        # Validate prompt name format
        if not prompt_name.replace("_", "").replace("-", "").isalnum():
            raise ValidationError("Prompt name must contain only alphanumeric characters, hyphens, and underscores")
        
        # Validate arguments
        arguments = params.get("arguments", {})
        if not isinstance(arguments, dict):
            raise ValidationError("Prompt arguments must be an object")
    
    def validate_tool_definition(self, tool: Dict[str, Any]) -> bool:
        """Validate tool definition format"""
        try:
            required_fields = ["name", "description", "input_schema"]
            
            for field in required_fields:
                if field not in tool:
                    raise ValidationError(f"Tool missing required field: {field}")
            
            # Validate name
            if not isinstance(tool["name"], str) or not tool["name"].strip():
                raise ValidationError("Tool name must be a non-empty string")
            
            # Validate description
            if not isinstance(tool["description"], str):
                raise ValidationError("Tool description must be a string")
            
            # Validate input schema (must be valid JSON Schema)
            input_schema = tool["input_schema"]
            if not isinstance(input_schema, dict):
                raise ValidationError("Tool input_schema must be an object")
            
            # Basic JSON Schema validation
            if "type" not in input_schema:
                raise ValidationError("Tool input_schema must include type")
            
            return True
            
        except ValidationError:
            raise
        except Exception as e:
            raise ValidationError(f"Tool definition validation failed: {str(e)}")
    
    def validate_resource_definition(self, resource: Dict[str, Any]) -> bool:
        """Validate resource definition format"""
        try:
            required_fields = ["uri", "name"]
            
            for field in required_fields:
                if field not in resource:
                    raise ValidationError(f"Resource missing required field: {field}")
            
            # Validate URI
            if not isinstance(resource["uri"], str) or not resource["uri"].strip():
                raise ValidationError("Resource URI must be a non-empty string")
            
            # Validate name
            if not isinstance(resource["name"], str) or not resource["name"].strip():
                raise ValidationError("Resource name must be a non-empty string")
            
            # Validate optional fields
            if "description" in resource and not isinstance(resource["description"], str):
                raise ValidationError("Resource description must be a string")
            
            if "mime_type" in resource and not isinstance(resource["mime_type"], str):
                raise ValidationError("Resource mime_type must be a string")
            
            return True
            
        except ValidationError:
            raise
        except Exception as e:
            raise ValidationError(f"Resource definition validation failed: {str(e)}")
    
    def validate_prompt_definition(self, prompt: Dict[str, Any]) -> bool:
        """Validate prompt definition format"""
        try:
            required_fields = ["name", "description"]
            
            for field in required_fields:
                if field not in prompt:
                    raise ValidationError(f"Prompt missing required field: {field}")
            
            # Validate name
            if not isinstance(prompt["name"], str) or not prompt["name"].strip():
                raise ValidationError("Prompt name must be a non-empty string")
            
            # Validate description
            if not isinstance(prompt["description"], str):
                raise ValidationError("Prompt description must be a string")
            
            # Validate optional arguments
            if "arguments" in prompt:
                arguments = prompt["arguments"]
                if not isinstance(arguments, list):
                    raise ValidationError("Prompt arguments must be an array")
                
                for arg in arguments:
                    if not isinstance(arg, dict):
                        raise ValidationError("Prompt argument must be an object")
                    if "name" not in arg:
                        raise ValidationError("Prompt argument missing name")
                    if not isinstance(arg["name"], str):
                        raise ValidationError("Prompt argument name must be a string")
            
            return True
            
        except ValidationError:
            raise
        except Exception as e:
            raise ValidationError(f"Prompt definition validation failed: {str(e)}")