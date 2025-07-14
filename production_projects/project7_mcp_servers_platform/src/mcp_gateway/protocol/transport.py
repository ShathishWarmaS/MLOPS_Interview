"""
MCP Transport Layer
Handles WebSocket and HTTP transport for MCP protocol
"""

import asyncio
import logging
import json
import time
from typing import Dict, List, Any, Optional, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime

from fastapi import WebSocket

logger = logging.getLogger(__name__)

class Transport(ABC):
    """Abstract base class for MCP transports"""
    
    def __init__(self, transport_type: str):
        self.transport_type = transport_type
        self.transport_id = None
        self.user_id = None
        self.permissions = []
        self.authenticated = False
        self.connected_at = time.time()
        self.last_activity = time.time()
        self.metadata = {}
    
    @abstractmethod
    async def send_message(self, message: Dict[str, Any]) -> bool:
        """Send message through transport"""
        pass
    
    @abstractmethod
    async def receive_message(self) -> Optional[Dict[str, Any]]:
        """Receive message from transport"""
        pass
    
    @abstractmethod
    async def close(self):
        """Close transport connection"""
        pass
    
    @abstractmethod
    def is_connected(self) -> bool:
        """Check if transport is connected"""
        pass
    
    def get_transport_id(self) -> str:
        """Get unique transport identifier"""
        return self.transport_id
    
    def get_user_id(self) -> Optional[str]:
        """Get authenticated user ID"""
        return self.user_id
    
    def get_permissions(self) -> List[str]:
        """Get user permissions"""
        return self.permissions.copy()
    
    def is_authenticated(self) -> bool:
        """Check if transport is authenticated"""
        return self.authenticated
    
    def set_user_info(self, user_id: str, permissions: List[str] = None):
        """Set user authentication info"""
        self.user_id = user_id
        self.permissions = permissions or []
        self.authenticated = True
        self.last_activity = time.time()
    
    def get_auth_token(self) -> Optional[str]:
        """Get authentication token from transport"""
        return None
    
    def update_activity(self):
        """Update last activity timestamp"""
        self.last_activity = time.time()
    
    def get_connection_info(self) -> Dict[str, Any]:
        """Get connection information"""
        return {
            "transport_id": self.transport_id,
            "transport_type": self.transport_type,
            "user_id": self.user_id,
            "authenticated": self.authenticated,
            "connected_at": datetime.fromtimestamp(self.connected_at).isoformat(),
            "last_activity": datetime.fromtimestamp(self.last_activity).isoformat(),
            "permissions": self.permissions,
            "metadata": self.metadata
        }

class WebSocketTransport(Transport):
    """WebSocket transport implementation"""
    
    def __init__(self, websocket: WebSocket, connection_id: str):
        super().__init__("websocket")
        self.websocket = websocket
        self.transport_id = connection_id
        self.auth_token = None
        self._closed = False
        
        logger.debug(f"WebSocket transport created: {connection_id}")
    
    async def send_message(self, message: Dict[str, Any]) -> bool:
        """Send message through WebSocket"""
        try:
            if self._closed or not self.is_connected():
                return False
            
            message_str = json.dumps(message)
            await self.websocket.send_text(message_str)
            self.update_activity()
            
            logger.debug(f"Sent WebSocket message: {message.get('method', 'response')}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send WebSocket message: {e}")
            self._closed = True
            return False
    
    async def receive_message(self) -> Optional[Dict[str, Any]]:
        """Receive message from WebSocket"""
        try:
            if self._closed or not self.is_connected():
                return None
            
            data = await self.websocket.receive_text()
            message = json.loads(data)
            self.update_activity()
            
            # Extract auth token if present
            if "params" in message and "auth" in message.get("params", {}):
                self.auth_token = message["params"]["auth"].get("token")
            
            logger.debug(f"Received WebSocket message: {message.get('method', 'response')}")
            return message
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in WebSocket message: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to receive WebSocket message: {e}")
            self._closed = True
            return None
    
    async def close(self):
        """Close WebSocket connection"""
        try:
            if not self._closed:
                await self.websocket.close()
                self._closed = True
                logger.debug(f"WebSocket transport closed: {self.transport_id}")
        except Exception as e:
            logger.error(f"Error closing WebSocket: {e}")
    
    def is_connected(self) -> bool:
        """Check if WebSocket is connected"""
        return not self._closed and self.websocket.client_state.value == 1  # CONNECTED
    
    def get_auth_token(self) -> Optional[str]:
        """Get authentication token from WebSocket"""
        return self.auth_token
    
    async def ping(self) -> bool:
        """Send ping to check connection"""
        try:
            if self.is_connected():
                await self.websocket.ping()
                return True
            return False
        except Exception:
            return False

class HTTPTransport(Transport):
    """HTTP transport implementation"""
    
    def __init__(self, 
                 user_id: Optional[str] = None, 
                 permissions: List[str] = None,
                 auth_token: Optional[str] = None):
        super().__init__("http")
        self.transport_id = f"http_{int(time.time() * 1000)}"
        self.auth_token = auth_token
        self.response_message = None
        
        # Set user info if provided
        if user_id:
            self.set_user_info(user_id, permissions or [])
        
        logger.debug(f"HTTP transport created: {self.transport_id}")
    
    async def send_message(self, message: Dict[str, Any]) -> bool:
        """Store message for HTTP response"""
        try:
            self.response_message = message
            self.update_activity()
            
            logger.debug(f"Stored HTTP response: {message.get('method', 'response')}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store HTTP response: {e}")
            return False
    
    async def receive_message(self) -> Optional[Dict[str, Any]]:
        """HTTP doesn't receive messages in this context"""
        return None
    
    async def close(self):
        """No-op for HTTP transport"""
        logger.debug(f"HTTP transport closed: {self.transport_id}")
    
    def is_connected(self) -> bool:
        """HTTP is always considered connected for single request"""
        return True
    
    def get_response(self) -> Optional[Dict[str, Any]]:
        """Get stored response message"""
        return self.response_message
    
    def get_auth_token(self) -> Optional[str]:
        """Get authentication token from HTTP"""
        return self.auth_token

@dataclass
class TransportMetrics:
    """Transport metrics tracking"""
    transport_id: str
    transport_type: str
    messages_sent: int = 0
    messages_received: int = 0
    bytes_sent: int = 0
    bytes_received: int = 0
    errors: int = 0
    last_error: Optional[str] = None
    connection_duration: float = 0.0

class TransportManager:
    """Manages multiple transport connections"""
    
    def __init__(self):
        self.transports: Dict[str, Transport] = {}
        self.metrics: Dict[str, TransportMetrics] = {}
        self.cleanup_tasks: Dict[str, asyncio.Task] = {}
        
        logger.info("Transport manager initialized")
    
    def register_transport(self, transport: Transport):
        """Register a new transport"""
        transport_id = transport.get_transport_id()
        self.transports[transport_id] = transport
        
        # Initialize metrics
        self.metrics[transport_id] = TransportMetrics(
            transport_id=transport_id,
            transport_type=transport.transport_type
        )
        
        # Start cleanup task for inactive connections
        self.cleanup_tasks[transport_id] = asyncio.create_task(
            self._monitor_transport(transport)
        )
        
        logger.info(f"Transport registered: {transport_id} ({transport.transport_type})")
    
    def unregister_transport(self, transport_id: str):
        """Unregister a transport"""
        if transport_id in self.transports:
            transport = self.transports.pop(transport_id)
            
            # Cancel cleanup task
            if transport_id in self.cleanup_tasks:
                task = self.cleanup_tasks.pop(transport_id)
                if not task.done():
                    task.cancel()
            
            # Update metrics
            if transport_id in self.metrics:
                metrics = self.metrics[transport_id]
                metrics.connection_duration = time.time() - transport.connected_at
            
            logger.info(f"Transport unregistered: {transport_id}")
    
    def get_transport(self, transport_id: str) -> Optional[Transport]:
        """Get transport by ID"""
        return self.transports.get(transport_id)
    
    def get_active_transports(self) -> List[Transport]:
        """Get all active transports"""
        return [t for t in self.transports.values() if t.is_connected()]
    
    def get_transports_by_user(self, user_id: str) -> List[Transport]:
        """Get all transports for a specific user"""
        return [
            t for t in self.transports.values() 
            if t.get_user_id() == user_id and t.is_connected()
        ]
    
    async def broadcast_message(self, message: Dict[str, Any], 
                              user_filter: Optional[str] = None):
        """Broadcast message to all or filtered transports"""
        try:
            transports = self.transports.values()
            
            if user_filter:
                transports = [t for t in transports if t.get_user_id() == user_filter]
            
            send_tasks = []
            for transport in transports:
                if transport.is_connected():
                    task = asyncio.create_task(transport.send_message(message))
                    send_tasks.append(task)
            
            if send_tasks:
                results = await asyncio.gather(*send_tasks, return_exceptions=True)
                
                # Count successful sends
                successful = sum(1 for r in results if r is True)
                logger.info(f"Broadcast message sent to {successful}/{len(send_tasks)} transports")
            
        except Exception as e:
            logger.error(f"Broadcast failed: {e}")
    
    async def send_to_transport(self, transport_id: str, message: Dict[str, Any]) -> bool:
        """Send message to specific transport"""
        transport = self.get_transport(transport_id)
        if transport and transport.is_connected():
            return await transport.send_message(message)
        return False
    
    async def cleanup_inactive_transports(self, max_idle_seconds: int = 3600):
        """Cleanup inactive transports"""
        try:
            current_time = time.time()
            to_remove = []
            
            for transport_id, transport in self.transports.items():
                idle_time = current_time - transport.last_activity
                
                if idle_time > max_idle_seconds or not transport.is_connected():
                    to_remove.append(transport_id)
            
            for transport_id in to_remove:
                transport = self.transports.get(transport_id)
                if transport:
                    await transport.close()
                    self.unregister_transport(transport_id)
            
            if to_remove:
                logger.info(f"Cleaned up {len(to_remove)} inactive transports")
                
        except Exception as e:
            logger.error(f"Transport cleanup failed: {e}")
    
    async def _monitor_transport(self, transport: Transport):
        """Monitor transport health and cleanup if needed"""
        try:
            while transport.is_connected():
                await asyncio.sleep(30)  # Check every 30 seconds
                
                # For WebSocket, try ping
                if isinstance(transport, WebSocketTransport):
                    if not await transport.ping():
                        logger.warning(f"Transport ping failed: {transport.transport_id}")
                        break
                
                # Check for idle timeout
                idle_time = time.time() - transport.last_activity
                if idle_time > 3600:  # 1 hour timeout
                    logger.info(f"Transport idle timeout: {transport.transport_id}")
                    break
            
            # Cleanup disconnected transport
            await transport.close()
            self.unregister_transport(transport.transport_id)
            
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Transport monitoring error: {e}")
    
    def get_transport_metrics(self) -> Dict[str, TransportMetrics]:
        """Get all transport metrics"""
        # Update connection durations
        current_time = time.time()
        for transport_id, transport in self.transports.items():
            if transport_id in self.metrics:
                self.metrics[transport_id].connection_duration = current_time - transport.connected_at
        
        return self.metrics.copy()
    
    async def shutdown(self):
        """Shutdown transport manager"""
        try:
            logger.info("Shutting down transport manager...")
            
            # Cancel all cleanup tasks
            for task in self.cleanup_tasks.values():
                if not task.done():
                    task.cancel()
            
            # Close all transports
            close_tasks = []
            for transport in self.transports.values():
                close_tasks.append(asyncio.create_task(transport.close()))
            
            if close_tasks:
                await asyncio.gather(*close_tasks, return_exceptions=True)
            
            self.transports.clear()
            self.metrics.clear()
            self.cleanup_tasks.clear()
            
            logger.info("Transport manager shutdown completed")
            
        except Exception as e:
            logger.error(f"Transport manager shutdown error: {e}")

# Global transport manager instance
transport_manager = TransportManager()