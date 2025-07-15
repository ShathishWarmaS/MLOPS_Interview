"""
Custom exceptions for MCP Gateway
Defines error types and error handling utilities
"""

from typing import Any, Optional, Dict
from enum import Enum

class MCPErrorCode(Enum):
    """Standard MCP error codes based on JSON-RPC 2.0"""
    
    # Standard JSON-RPC errors
    PARSE_ERROR = -32700
    INVALID_REQUEST = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMS = -32602
    INTERNAL_ERROR = -32603
    
    # MCP-specific errors
    AUTHENTICATION_ERROR = -32001
    AUTHORIZATION_ERROR = -32002
    RESOURCE_NOT_FOUND = -32003
    TOOL_NOT_FOUND = -32004
    PROMPT_NOT_FOUND = -32005
    SERVER_UNAVAILABLE = -32006
    RATE_LIMITED = -32007
    VALIDATION_ERROR = -32008
    TIMEOUT_ERROR = -32009
    CIRCUIT_BREAKER_OPEN = -32010

class MCPException(Exception):
    """Base exception for MCP-related errors"""
    
    def __init__(self, 
                 message: str, 
                 error_code: int = MCPErrorCode.INTERNAL_ERROR.value,
                 details: Optional[Any] = None,
                 status_code: int = 500):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details
        self.status_code = status_code
        self.error_type = self._get_error_type()
    
    def _get_error_type(self) -> str:
        """Get error type string from error code"""
        for error_enum in MCPErrorCode:
            if error_enum.value == self.error_code:
                return error_enum.name.lower()
        return "unknown_error"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for JSON serialization"""
        result = {
            "error_type": self.error_type,
            "message": self.message,
            "error_code": self.error_code,
            "status_code": self.status_code
        }
        
        if self.details is not None:
            result["details"] = self.details
        
        return result
    
    def __str__(self) -> str:
        return f"MCPException({self.error_type}): {self.message}"

class ValidationError(MCPException):
    """Exception for validation errors"""
    
    def __init__(self, message: str, details: Optional[Any] = None):
        super().__init__(
            message=message,
            error_code=MCPErrorCode.VALIDATION_ERROR.value,
            details=details,
            status_code=400
        )

class AuthenticationError(MCPException):
    """Exception for authentication errors"""
    
    def __init__(self, message: str = "Authentication required", details: Optional[Any] = None):
        super().__init__(
            message=message,
            error_code=MCPErrorCode.AUTHENTICATION_ERROR.value,
            details=details,
            status_code=401
        )

class AuthorizationError(MCPException):
    """Exception for authorization errors"""
    
    def __init__(self, message: str = "Insufficient permissions", details: Optional[Any] = None):
        super().__init__(
            message=message,
            error_code=MCPErrorCode.AUTHORIZATION_ERROR.value,
            details=details,
            status_code=403
        )

class ResourceNotFoundError(MCPException):
    """Exception for resource not found errors"""
    
    def __init__(self, resource_uri: str, details: Optional[Any] = None):
        message = f"Resource not found: {resource_uri}"
        super().__init__(
            message=message,
            error_code=MCPErrorCode.RESOURCE_NOT_FOUND.value,
            details=details,
            status_code=404
        )

class ToolNotFoundError(MCPException):
    """Exception for tool not found errors"""
    
    def __init__(self, tool_name: str, details: Optional[Any] = None):
        message = f"Tool not found: {tool_name}"
        super().__init__(
            message=message,
            error_code=MCPErrorCode.TOOL_NOT_FOUND.value,
            details=details,
            status_code=404
        )

class PromptNotFoundError(MCPException):
    """Exception for prompt not found errors"""
    
    def __init__(self, prompt_name: str, details: Optional[Any] = None):
        message = f"Prompt not found: {prompt_name}"
        super().__init__(
            message=message,
            error_code=MCPErrorCode.PROMPT_NOT_FOUND.value,
            details=details,
            status_code=404
        )

class ServerUnavailableError(MCPException):
    """Exception for server unavailable errors"""
    
    def __init__(self, server_id: str, details: Optional[Any] = None):
        message = f"Server unavailable: {server_id}"
        super().__init__(
            message=message,
            error_code=MCPErrorCode.SERVER_UNAVAILABLE.value,
            details=details,
            status_code=503
        )

class RateLimitedError(MCPException):
    """Exception for rate limiting errors"""
    
    def __init__(self, message: str = "Rate limit exceeded", details: Optional[Any] = None):
        super().__init__(
            message=message,
            error_code=MCPErrorCode.RATE_LIMITED.value,
            details=details,
            status_code=429
        )

class TimeoutError(MCPException):
    """Exception for timeout errors"""
    
    def __init__(self, operation: str, timeout_seconds: int, details: Optional[Any] = None):
        message = f"Operation '{operation}' timed out after {timeout_seconds} seconds"
        super().__init__(
            message=message,
            error_code=MCPErrorCode.TIMEOUT_ERROR.value,
            details=details,
            status_code=408
        )

class CircuitBreakerOpenError(MCPException):
    """Exception for circuit breaker open errors"""
    
    def __init__(self, service: str, details: Optional[Any] = None):
        message = f"Circuit breaker open for service: {service}"
        super().__init__(
            message=message,
            error_code=MCPErrorCode.CIRCUIT_BREAKER_OPEN.value,
            details=details,
            status_code=503
        )

class ParseError(MCPException):
    """Exception for JSON parsing errors"""
    
    def __init__(self, message: str = "Invalid JSON", details: Optional[Any] = None):
        super().__init__(
            message=message,
            error_code=MCPErrorCode.PARSE_ERROR.value,
            details=details,
            status_code=400
        )

class InvalidRequestError(MCPException):
    """Exception for invalid request errors"""
    
    def __init__(self, message: str = "Invalid request", details: Optional[Any] = None):
        super().__init__(
            message=message,
            error_code=MCPErrorCode.INVALID_REQUEST.value,
            details=details,
            status_code=400
        )

class MethodNotFoundError(MCPException):
    """Exception for method not found errors"""
    
    def __init__(self, method: str, details: Optional[Any] = None):
        message = f"Method not found: {method}"
        super().__init__(
            message=message,
            error_code=MCPErrorCode.METHOD_NOT_FOUND.value,
            details=details,
            status_code=404
        )

class InvalidParamsError(MCPException):
    """Exception for invalid parameters errors"""
    
    def __init__(self, message: str = "Invalid parameters", details: Optional[Any] = None):
        super().__init__(
            message=message,
            error_code=MCPErrorCode.INVALID_PARAMS.value,
            details=details,
            status_code=400
        )

# Exception handling utilities

def handle_exception(exc: Exception) -> MCPException:
    """Convert generic exceptions to MCP exceptions"""
    if isinstance(exc, MCPException):
        return exc
    
    # Map common exceptions
    if isinstance(exc, ValueError):
        return ValidationError(str(exc))
    elif isinstance(exc, KeyError):
        return ValidationError(f"Missing required field: {exc}")
    elif isinstance(exc, TimeoutError):
        return TimeoutError("operation", 30)
    elif isinstance(exc, ConnectionError):
        return ServerUnavailableError("unknown")
    else:
        return MCPException(
            message=f"Internal error: {str(exc)}",
            error_code=MCPErrorCode.INTERNAL_ERROR.value,
            details={"exception_type": exc.__class__.__name__}
        )

def create_error_response(error: MCPException, request_id: Optional[Any] = None) -> Dict[str, Any]:
    """Create JSON-RPC error response"""
    response = {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {
            "code": error.error_code,
            "message": error.message
        }
    }
    
    if error.details is not None:
        response["error"]["data"] = error.details
    
    return response

def validate_required_fields(data: Dict[str, Any], required_fields: list) -> None:
    """Validate that required fields are present in data"""
    missing_fields = []
    for field in required_fields:
        if field not in data or data[field] is None:
            missing_fields.append(field)
    
    if missing_fields:
        raise ValidationError(
            f"Missing required fields: {', '.join(missing_fields)}",
            details={"missing_fields": missing_fields}
        )

def validate_field_types(data: Dict[str, Any], field_types: Dict[str, type]) -> None:
    """Validate that fields have correct types"""
    type_errors = []
    for field, expected_type in field_types.items():
        if field in data and data[field] is not None:
            if not isinstance(data[field], expected_type):
                type_errors.append({
                    "field": field,
                    "expected_type": expected_type.__name__,
                    "actual_type": type(data[field]).__name__
                })
    
    if type_errors:
        raise ValidationError(
            "Invalid field types",
            details={"type_errors": type_errors}
        )

def validate_uri_format(uri: str) -> None:
    """Validate URI format"""
    if not uri:
        raise ValidationError("URI cannot be empty")
    
    if not uri.startswith(('http://', 'https://', 'file://', 'database://', 'memory://')):
        raise ValidationError(
            f"Invalid URI scheme: {uri}",
            details={"valid_schemes": ["http://", "https://", "file://", "database://", "memory://"]}
        )

def validate_tool_name(name: str) -> None:
    """Validate tool name format"""
    if not name:
        raise ValidationError("Tool name cannot be empty")
    
    if not name.replace("_", "").replace("-", "").isalnum():
        raise ValidationError(
            f"Invalid tool name: {name}. Tool names can only contain letters, numbers, hyphens, and underscores"
        )

def validate_json_schema(schema: Dict[str, Any]) -> None:
    """Basic JSON schema validation"""
    if not isinstance(schema, dict):
        raise ValidationError("Schema must be a dictionary")
    
    if "type" not in schema:
        raise ValidationError("Schema must have a 'type' field")
    
    valid_types = ["string", "number", "integer", "boolean", "array", "object", "null"]
    if schema["type"] not in valid_types:
        raise ValidationError(
            f"Invalid schema type: {schema['type']}",
            details={"valid_types": valid_types}
        )