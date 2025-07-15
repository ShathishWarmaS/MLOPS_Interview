"""
PostgreSQL MCP Server
Provides database access through MCP protocol with security and validation
"""

import asyncio
import logging
import json
import re
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from datetime import datetime

import asyncpg
from asyncpg import Connection, Pool
from pydantic import BaseModel, Field

from ..base.server import BaseMCPServer, MCPServerInfo, MCPServerCapability, MCPTool, MCPResource
from ...shared.models import DatabaseConfig

logger = logging.getLogger(__name__)

class QueryRequest(BaseModel):
    """Query request model"""
    query: str = Field(..., description="SQL query to execute")
    parameters: Optional[Dict[str, Any]] = Field(default=None, description="Query parameters")
    limit: Optional[int] = Field(default=1000, description="Maximum number of rows to return")
    timeout: Optional[int] = Field(default=30, description="Query timeout in seconds")

class TableInfo(BaseModel):
    """Table information model"""
    table_name: str
    schema_name: str = "public"
    include_data: bool = False
    limit: Optional[int] = 100

@dataclass
class QueryResult:
    """Query execution result"""
    rows: List[Dict[str, Any]]
    row_count: int
    columns: List[str]
    execution_time_ms: float
    query: str

class PostgreSQLMCPServer(BaseMCPServer):
    """PostgreSQL MCP Server implementation"""
    
    def __init__(self, database_config: DatabaseConfig):
        # Initialize server info
        server_info = MCPServerInfo(
            name="postgresql-mcp-server",
            version="1.0.0",
            description="PostgreSQL database access through MCP protocol",
            author="MCP Team",
            license="MIT"
        )
        
        # Define capabilities
        capabilities = [
            MCPServerCapability.TOOLS,
            MCPServerCapability.RESOURCES
        ]
        
        super().__init__(server_info, capabilities)
        
        self.database_config = database_config
        self.pool: Optional[Pool] = None
        
        # Safe query patterns (read-only operations)
        self.safe_query_patterns = [
            r'^\\s*SELECT\\s+',
            r'^\\s*WITH\\s+.*\\s+SELECT\\s+',
            r'^\\s*EXPLAIN\\s+',
            r'^\\s*DESCRIBE\\s+',
            r'^\\s*SHOW\\s+'
        ]
        
        # Dangerous query patterns (to block)
        self.dangerous_patterns = [
            r'\\b(DROP|DELETE|UPDATE|INSERT|ALTER|CREATE|TRUNCATE)\\b',
            r'\\b(GRANT|REVOKE)\\b',
            r'\\bEXEC(UTE)?\\b',
            r'\\b(CALL|PERFORM)\\b',
            r'\\bCOPY\\b',
            r'\\bBACKUP\\b',
            r'\\bRESTORE\\b'
        ]
        
        # Initialize database connection
        asyncio.create_task(self._initialize_database())
    
    async def _initialize_database(self):
        """Initialize database connection pool"""
        try:
            self.pool = await asyncpg.create_pool(
                dsn=self.database_config.url,
                min_size=2,
                max_size=10,
                command_timeout=60
            )
            logger.info("Database connection pool initialized")
        except Exception as e:
            logger.error(f"Failed to initialize database pool: {e}")
            raise
    
    def _register_tools(self):
        """Register database tools"""
        
        # SQL Query tool
        self.register_tool(MCPTool(
            name="sql_query",
            description="Execute a safe SQL query against the PostgreSQL database",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "SQL query to execute (SELECT, EXPLAIN, DESCRIBE, SHOW only)"
                    },
                    "parameters": {
                        "type": "object",
                        "description": "Named parameters for the query",
                        "additionalProperties": True
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of rows to return",
                        "default": 1000,
                        "minimum": 1,
                        "maximum": 10000
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Query timeout in seconds",
                        "default": 30,
                        "minimum": 1,
                        "maximum": 300
                    }
                },
                "required": ["query"]
            }
        ))
        
        # Table Schema tool
        self.register_tool(MCPTool(
            name="table_schema",
            description="Get the schema information for a database table",
            input_schema={
                "type": "object",
                "properties": {
                    "table_name": {
                        "type": "string",
                        "description": "Name of the table to inspect"
                    },
                    "schema_name": {
                        "type": "string",
                        "description": "Schema name (default: public)",
                        "default": "public"
                    }
                },
                "required": ["table_name"]
            }
        ))
        
        # List Tables tool
        self.register_tool(MCPTool(
            name="list_tables",
            description="List all tables in the database",
            input_schema={
                "type": "object",
                "properties": {
                    "schema_name": {
                        "type": "string",
                        "description": "Schema name to list tables from (default: public)",
                        "default": "public"
                    },
                    "include_views": {
                        "type": "boolean",
                        "description": "Whether to include views in the result",
                        "default": false
                    }
                }
            }
        ))
        
        # Database Stats tool
        self.register_tool(MCPTool(
            name="database_stats",
            description="Get database statistics and information",
            input_schema={
                "type": "object",
                "properties": {}
            }
        ))
    
    def _register_resources(self):
        """Register database resources"""
        
        # Database schema resource
        self.register_resource(MCPResource(
            uri="database://schema/public",
            name="Public Schema",
            description="Complete schema information for the public database schema",
            mime_type="application/json"
        ))
        
        # Database statistics resource
        self.register_resource(MCPResource(
            uri="database://stats/general",
            name="Database Statistics",
            description="General database statistics and performance metrics",
            mime_type="application/json"
        ))
    
    def _register_prompts(self):
        """Register database prompts"""
        pass  # No prompts for this server
    
    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        """Execute a database tool"""
        if not self.pool:
            raise RuntimeError("Database connection not initialized")
        
        if name == "sql_query":
            return await self._execute_sql_query(arguments)
        elif name == "table_schema":
            return await self._get_table_schema(arguments)
        elif name == "list_tables":
            return await self._list_tables(arguments)
        elif name == "database_stats":
            return await self._get_database_stats()
        else:
            raise ValueError(f"Unknown tool: {name}")
    
    async def read_resource(self, uri: str) -> Dict[str, Any]:
        """Read a database resource"""
        if not self.pool:
            raise RuntimeError("Database connection not initialized")
        
        if uri == "database://schema/public":
            return await self._get_schema_resource()
        elif uri == "database://stats/general":
            return await self._get_stats_resource()
        else:
            raise ValueError(f"Unknown resource: {uri}")
    
    async def get_prompt(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Get a database prompt"""
        raise ValueError(f"No prompts available in this server")
    
    def _validate_query_safety(self, query: str) -> bool:
        """Validate that the query is safe to execute"""
        query_normalized = query.strip().upper()
        
        # Check if query matches safe patterns
        safe_query = False
        for pattern in self.safe_query_patterns:
            if re.match(pattern, query_normalized, re.IGNORECASE):
                safe_query = True
                break
        
        if not safe_query:
            return False
        
        # Check for dangerous patterns
        for pattern in self.dangerous_patterns:
            if re.search(pattern, query_normalized, re.IGNORECASE):
                return False
        
        # Additional safety checks
        if '--' in query or '/*' in query or '*/' in query:
            return False
        
        return True
    
    async def _execute_sql_query(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute SQL query"""
        try:
            query = arguments.get("query", "").strip()
            parameters = arguments.get("parameters", {})
            limit = min(arguments.get("limit", 1000), 10000)
            timeout = min(arguments.get("timeout", 30), 300)
            
            if not query:
                raise ValueError("Query cannot be empty")
            
            # Validate query safety
            if not self._validate_query_safety(query):
                raise ValueError("Query contains unsafe operations. Only SELECT, EXPLAIN, DESCRIBE, and SHOW queries are allowed.")
            
            # Add LIMIT if not present
            if "LIMIT" not in query.upper() and query.upper().strip().startswith("SELECT"):
                query = f"{query} LIMIT {limit}"
            
            start_time = datetime.now()
            
            async with self.pool.acquire() as conn:
                # Set query timeout
                await conn.execute(f"SET statement_timeout = '{timeout}s'")
                
                # Execute query
                if parameters:
                    rows = await conn.fetch(query, *parameters.values())
                else:
                    rows = await conn.fetch(query)
                
                # Convert to dict format
                result_rows = []
                columns = []
                
                if rows:
                    columns = list(rows[0].keys())
                    for row in rows:
                        result_rows.append(dict(row))
                
                execution_time = (datetime.now() - start_time).total_seconds() * 1000
                
                return {
                    "success": True,
                    "rows": result_rows,
                    "row_count": len(result_rows),
                    "columns": columns,
                    "execution_time_ms": round(execution_time, 2),
                    "query": query,
                    "limited": "LIMIT" in query.upper()
                }
                
        except Exception as e:
            logger.error(f"SQL query execution failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "query": arguments.get("query", "")
            }
    
    async def _get_table_schema(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Get table schema information"""
        try:
            table_name = arguments.get("table_name")
            schema_name = arguments.get("schema_name", "public")
            
            if not table_name:
                raise ValueError("Table name is required")
            
            async with self.pool.acquire() as conn:
                # Get table information
                table_info_query = '''
                    SELECT 
                        column_name,
                        data_type,
                        is_nullable,
                        column_default,
                        character_maximum_length,
                        numeric_precision,
                        numeric_scale,
                        ordinal_position
                    FROM information_schema.columns 
                    WHERE table_schema = $1 AND table_name = $2
                    ORDER BY ordinal_position
                '''
                
                columns = await conn.fetch(table_info_query, schema_name, table_name)
                
                if not columns:
                    raise ValueError(f"Table '{schema_name}.{table_name}' not found")
                
                # Get primary keys
                pk_query = '''
                    SELECT column_name
                    FROM information_schema.key_column_usage k
                    JOIN information_schema.table_constraints t
                        ON k.constraint_name = t.constraint_name
                    WHERE t.table_schema = $1 
                        AND t.table_name = $2 
                        AND t.constraint_type = 'PRIMARY KEY'
                    ORDER BY k.ordinal_position
                '''
                
                primary_keys = await conn.fetch(pk_query, schema_name, table_name)
                
                # Get foreign keys
                fk_query = '''
                    SELECT 
                        kcu.column_name,
                        ccu.table_schema AS foreign_table_schema,
                        ccu.table_name AS foreign_table_name,
                        ccu.column_name AS foreign_column_name
                    FROM information_schema.key_column_usage kcu
                    JOIN information_schema.table_constraints tc
                        ON kcu.constraint_name = tc.constraint_name
                    JOIN information_schema.constraint_column_usage ccu
                        ON ccu.constraint_name = tc.constraint_name
                    WHERE tc.table_schema = $1 
                        AND tc.table_name = $2
                        AND tc.constraint_type = 'FOREIGN KEY'
                '''
                
                foreign_keys = await conn.fetch(fk_query, schema_name, table_name)
                
                # Get indexes
                index_query = '''
                    SELECT 
                        indexname,
                        indexdef
                    FROM pg_indexes
                    WHERE schemaname = $1 AND tablename = $2
                '''
                
                indexes = await conn.fetch(index_query, schema_name, table_name)
                
                return {
                    "table_name": table_name,
                    "schema_name": schema_name,
                    "columns": [dict(col) for col in columns],
                    "primary_keys": [pk["column_name"] for pk in primary_keys],
                    "foreign_keys": [dict(fk) for fk in foreign_keys],
                    "indexes": [dict(idx) for idx in indexes]
                }
                
        except Exception as e:
            logger.error(f"Get table schema failed: {e}")
            return {"error": str(e)}
    
    async def _list_tables(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """List database tables"""
        try:
            schema_name = arguments.get("schema_name", "public")
            include_views = arguments.get("include_views", False)
            
            async with self.pool.acquire() as conn:
                if include_views:
                    query = '''
                        SELECT 
                            table_name,
                            table_type,
                            table_schema
                        FROM information_schema.tables 
                        WHERE table_schema = $1
                        ORDER BY table_name
                    '''
                else:
                    query = '''
                        SELECT 
                            table_name,
                            table_type,
                            table_schema
                        FROM information_schema.tables 
                        WHERE table_schema = $1 AND table_type = 'BASE TABLE'
                        ORDER BY table_name
                    '''
                
                tables = await conn.fetch(query, schema_name)
                
                return {
                    "schema_name": schema_name,
                    "tables": [dict(table) for table in tables],
                    "count": len(tables)
                }
                
        except Exception as e:
            logger.error(f"List tables failed: {e}")
            return {"error": str(e)}
    
    async def _get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        try:
            async with self.pool.acquire() as conn:
                # Database size
                db_size_query = "SELECT pg_size_pretty(pg_database_size(current_database())) as size"
                db_size = await conn.fetchval(db_size_query)
                
                # Table count
                table_count_query = '''
                    SELECT COUNT(*) 
                    FROM information_schema.tables 
                    WHERE table_schema NOT IN ('information_schema', 'pg_catalog')
                '''
                table_count = await conn.fetchval(table_count_query)
                
                # Database version
                version_query = "SELECT version()"
                version = await conn.fetchval(version_query)
                
                # Current connections
                connections_query = '''
                    SELECT COUNT(*) 
                    FROM pg_stat_activity 
                    WHERE state = 'active'
                '''
                active_connections = await conn.fetchval(connections_query)
                
                return {
                    "database_size": db_size,
                    "table_count": table_count,
                    "version": version,
                    "active_connections": active_connections,
                    "current_database": await conn.fetchval("SELECT current_database()"),
                    "current_user": await conn.fetchval("SELECT current_user"),
                    "timestamp": datetime.utcnow().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Get database stats failed: {e}")
            return {"error": str(e)}
    
    async def _get_schema_resource(self) -> Dict[str, Any]:
        """Get complete schema as a resource"""
        try:
            async with self.pool.acquire() as conn:
                # Get all tables with their schemas
                schema_query = '''
                    SELECT 
                        t.table_name,
                        t.table_type,
                        c.column_name,
                        c.data_type,
                        c.is_nullable,
                        c.column_default,
                        c.ordinal_position
                    FROM information_schema.tables t
                    LEFT JOIN information_schema.columns c
                        ON t.table_name = c.table_name 
                        AND t.table_schema = c.table_schema
                    WHERE t.table_schema = 'public'
                    ORDER BY t.table_name, c.ordinal_position
                '''
                
                rows = await conn.fetch(schema_query)
                
                # Organize by table
                tables = {}
                for row in rows:
                    table_name = row['table_name']
                    if table_name not in tables:
                        tables[table_name] = {
                            "table_name": table_name,
                            "table_type": row['table_type'],
                            "columns": []
                        }
                    
                    if row['column_name']:  # Some tables might have no columns in the join
                        tables[table_name]["columns"].append({
                            "column_name": row['column_name'],
                            "data_type": row['data_type'],
                            "is_nullable": row['is_nullable'],
                            "column_default": row['column_default'],
                            "ordinal_position": row['ordinal_position']
                        })
                
                schema_data = {
                    "schema_name": "public",
                    "tables": list(tables.values()),
                    "generated_at": datetime.utcnow().isoformat()
                }
                
                return {
                    "mime_type": "application/json",
                    "text": json.dumps(schema_data, indent=2)
                }
                
        except Exception as e:
            logger.error(f"Get schema resource failed: {e}")
            return {
                "mime_type": "application/json",
                "text": json.dumps({"error": str(e)})
            }
    
    async def _get_stats_resource(self) -> Dict[str, Any]:
        """Get database statistics as a resource"""
        stats = await self._get_database_stats()
        return {
            "mime_type": "application/json",
            "text": json.dumps(stats, indent=2)
        }

# Example usage
if __name__ == "__main__":
    # Example configuration
    config = DatabaseConfig(
        url="postgresql://user:password@localhost:5432/database"
    )
    
    server = PostgreSQLMCPServer(config)
    server.run(host="0.0.0.0", port=8001)