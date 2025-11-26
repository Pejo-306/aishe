---
type: agent_requested
description: "mcp client description"
keywords: ["mcp", "anthropic", "model context protocol", "server", "client", "python-sdk"]
---

# Model Context Protocol (MCP) Python SDK

## SDK Choice

**Always use the official MCP Python SDK** from https://github.com/modelcontextprotocol/python-sdk for building MCP servers and clients.

This is the official Python implementation maintained by Anthropic for the Model Context Protocol.

## Installation

```bash
pip install mcp
```

For server development with all features:
```bash
pip install "mcp[cli]"
```

## Core Concepts

MCP defines three core primitives:

1. **Prompts** (User-controlled): Interactive templates invoked by user choice
2. **Resources** (Application-controlled): Contextual data managed by the client application
3. **Tools** (Model-controlled): Functions exposed to the LLM to take actions

## Server Development Best Practices

### 1. Use FastMCP for High-Level Server Development

FastMCP is the recommended way to build MCP servers. It provides a simple, decorator-based API:

```python
from mcp.server.fastmcp import FastMCP

# Create server instance
mcp = FastMCP("My Server")

@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b

@mcp.resource("config://app")
def get_config() -> str:
    """Get application configuration."""
    return "Configuration data"

@mcp.prompt()
def review_code(code: str) -> str:
    """Generate a code review prompt."""
    return f"Please review this code:\n\n{code}"

if __name__ == "__main__":
    mcp.run()
```

### 2. Use Type Hints for Automatic Validation

Always use type hints - FastMCP automatically generates JSON schemas:

```python
from typing import Optional
from pathlib import Path

@mcp.tool()
def process_file(
    input_path: Path,
    output_path: Optional[Path] = None,
    verbose: bool = False
) -> str:
    """Process a file with optional output."""
    # Type hints automatically create the input schema
    return f"Processed {input_path}"
```

### 3. Use Context for Request Information

Access request context, session info, and server metadata:

```python
from mcp.server.fastmcp import Context

@mcp.tool()
async def get_user_info(ctx: Context) -> dict:
    """Get information about the current session."""
    return {
        "server_name": ctx.fastmcp.name,
        "client_params": ctx.session.client_params,
        "request_id": ctx.request_context.request_id,
    }
```

### 4. Use Lifespan for Resource Management

Initialize resources on startup and clean them up on shutdown:

```python
from contextlib import asynccontextmanager
from typing import Any

@asynccontextmanager
async def app_lifespan(server: FastMCP):
    """Manage server lifecycle."""
    # Startup: initialize resources
    db = await Database.connect()
    cache = await Cache.connect()

    try:
        yield {"db": db, "cache": cache}
    finally:
        # Shutdown: cleanup resources
        await db.disconnect()
        await cache.disconnect()

mcp = FastMCP("My Server", lifespan=app_lifespan)

@mcp.tool()
async def query_data(query: str, ctx: Context) -> str:
    """Query the database."""
    db = ctx.request_context.lifespan_context["db"]
    result = await db.execute(query)
    return str(result)
```

### 5. Use Progress Notifications for Long Operations

Keep clients informed during long-running operations:

```python
@mcp.tool()
async def process_large_dataset(items: list[str], ctx: Context) -> str:
    """Process a large dataset with progress updates."""
    total = len(items)

    for i, item in enumerate(items):
        # Send progress update
        await ctx.session.send_progress_notification(
            progress=i + 1,
            total=total,
            message=f"Processing {item}"
        )
        await process_item(item)

    return f"Processed {total} items"
```

### 6. Use Dependencies for Shared Logic

Share logic across tools using dependencies:

```python
from mcp.server.fastmcp import Context

async def get_api_client(ctx: Context):
    """Dependency that provides an API client."""
    api_key = ctx.request_context.lifespan_context.get("api_key")
    return APIClient(api_key)

@mcp.tool()
async def fetch_data(
    endpoint: str,
    api_client = mcp.depends(get_api_client)
) -> dict:
    """Fetch data from an API endpoint."""
    return await api_client.get(endpoint)
```

### 7. Implement Proper Error Handling

Handle errors gracefully and return meaningful messages:

```python
@mcp.tool()
def divide(a: float, b: float) -> float:
    """Divide two numbers."""
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b

@mcp.tool()
async def fetch_user(user_id: int) -> dict:
    """Fetch user information."""
    try:
        user = await db.get_user(user_id)
        if not user:
            raise ValueError(f"User {user_id} not found")
        return user
    except DatabaseError as e:
        raise RuntimeError(f"Database error: {e}")
```

### 8. Use Resource Templates for Dynamic Resources

For resources with dynamic URIs:

```python
@mcp.resource("file://{path}")
def read_file(path: str) -> str:
    """Read a file from the filesystem."""
    with open(path) as f:
        return f.read()
```

### 9. Send Notifications for Resource Changes

Notify clients when resources change:

```python
@mcp.tool()
async def update_config(key: str, value: str, ctx: Context) -> str:
    """Update configuration and notify clients."""
    # Update the configuration
    await config_store.set(key, value)

    # Notify clients that the resource changed
    await ctx.session.send_resource_updated(AnyUrl("config://app"))

    return f"Updated {key} = {value}"
```

### 10. Use Structured Output for Tool Results

Return structured data that can be validated:

```python
from pydantic import BaseModel

class WeatherData(BaseModel):
    temperature: float
    condition: str
    humidity: int

@mcp.tool()
def get_weather(city: str) -> WeatherData:
    """Get weather for a city with structured output."""
    return WeatherData(
        temperature=22.5,
        condition="partly cloudy",
        humidity=65
    )
```



## Running MCP Servers

### Development Mode

Use the MCP Inspector for testing and debugging:

```bash
# Basic usage
uv run mcp dev server.py

# With dependencies
uv run mcp dev server.py --with pandas --with numpy

# Mount local code
uv run mcp dev server.py --with-editable .
```

### Claude Desktop Integration

Install your server in Claude Desktop:

```bash
# Basic installation
uv run mcp install server.py

# Custom name
uv run mcp install server.py --name "My Server"

# With environment variables
uv run mcp install server.py -v API_KEY=abc123 -v DB_URL=postgres://...
uv run mcp install server.py -f .env
```

### Direct Execution

For custom deployments:

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("My App")

@mcp.tool()
def hello(name: str = "World") -> str:
    """Say hello."""
    return f"Hello, {name}!"

if __name__ == "__main__":
    mcp.run()
```

Run with:
```bash
python server.py
# or
uv run mcp run server.py
```

### Streamable HTTP Transport (Recommended for Production)

Use Streamable HTTP for production deployments:

```python
from mcp.server.fastmcp import FastMCP

# Stateless server with JSON responses (recommended)
mcp = FastMCP("MyServer", stateless_http=True, json_response=True)

@mcp.tool()
def greet(name: str = "World") -> str:
    """Greet someone."""
    return f"Hello, {name}!"

if __name__ == "__main__":
    mcp.run(transport="streamable-http")
```

### Mounting in Starlette/FastAPI

Mount multiple MCP servers in a web application:

```python
from starlette.applications import Starlette
from starlette.routing import Mount
from mcp.server.fastmcp import FastMCP

# Create servers
api_mcp = FastMCP("API Server", stateless_http=True, json_response=True)
chat_mcp = FastMCP("Chat Server", stateless_http=True, json_response=True)

# Configure mount paths
api_mcp.settings.streamable_http_path = "/"
chat_mcp.settings.streamable_http_path = "/"

# Create Starlette app
app = Starlette(
    routes=[
        Mount("/api", app=api_mcp.streamable_http_app()),
        Mount("/chat", app=chat_mcp.streamable_http_app()),
    ]
)
```

### CORS Configuration for Browser Clients

Enable CORS for browser-based clients:

```python
from starlette.middleware.cors import CORSMiddleware

app = CORSMiddleware(
    starlette_app,
    allow_origins=["*"],  # Configure for production
    allow_methods=["GET", "POST", "DELETE"],
    expose_headers=["Mcp-Session-Id"],  # Required for session management
)
```

## Client Development Best Practices

### 1. Use ClientSession for Server Communication

```python
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def main():
    server_params = StdioServerParameters(
        command="uv",
        args=["run", "server.py"]
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize connection
            await session.initialize()

            # List available tools
            tools = await session.list_tools()
            print(f"Tools: {[t.name for t in tools.tools]}")

            # Call a tool
            result = await session.call_tool("add", arguments={"a": 5, "b": 3})
            print(f"Result: {result.content}")

asyncio.run(main())
```

### 2. Use Streamable HTTP for Production Clients

```python
from mcp.client.streamable_http import streamablehttp_client

async def main():
    async with streamablehttp_client("http://localhost:8000/mcp") as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            print(f"Available tools: {[t.name for t in tools.tools]}")
```

### 3. Implement Sampling Callback for Model Requests

Handle server requests for LLM sampling:

```python
from mcp import types
from mcp.shared.context import RequestContext

async def handle_sampling(
    context: RequestContext[ClientSession, None],
    params: types.CreateMessageRequestParams
) -> types.CreateMessageResult:
    """Handle sampling requests from the server."""
    # Call your LLM here
    return types.CreateMessageResult(
        role="assistant",
        content=types.TextContent(type="text", text="Response from model"),
        model="gpt-4",
        stopReason="endTurn"
    )

async with ClientSession(read, write, sampling_callback=handle_sampling) as session:
    await session.initialize()
```

### 4. Parse Tool Results Properly

Handle different content types in tool results:

```python
result = await session.call_tool("get_data", {})

# Parse text content
for content in result.content:
    if isinstance(content, types.TextContent):
        print(f"Text: {content.text}")
    elif isinstance(content, types.ImageContent):
        print(f"Image: {len(content.data)} bytes")
    elif isinstance(content, types.EmbeddedResource):
        print(f"Resource: {content.resource.uri}")

# Access structured content (if available)
if hasattr(result, "structuredContent") and result.structuredContent:
    data = result.structuredContent
    print(f"Structured: {data}")
```

### 5. Use Display Utilities for Human-Readable Names

```python
from mcp.shared.metadata_utils import get_display_name

tools = await session.list_tools()
for tool in tools.tools:
    # Returns title if available, otherwise name
    display_name = get_display_name(tool)
    print(f"Tool: {display_name}")
```

### 6. Implement OAuth Authentication

For protected servers:

```python
from mcp.client.auth import OAuthClientProvider, TokenStorage
from mcp.shared.auth import OAuthClientMetadata

class MyTokenStorage(TokenStorage):
    """Implement token storage."""
    async def get_tokens(self):
        # Load tokens from secure storage
        pass

    async def set_tokens(self, tokens):
        # Save tokens to secure storage
        pass

oauth_auth = OAuthClientProvider(
    server_url="http://localhost:8001",
    client_metadata=OAuthClientMetadata(
        client_name="My Client",
        redirect_uris=["http://localhost:3000/callback"],
        grant_types=["authorization_code", "refresh_token"],
        response_types=["code"],
        scope="user"
    ),
    storage=MyTokenStorage(),
    redirect_handler=handle_redirect,
    callback_handler=handle_callback
)

async with streamablehttp_client(
    "http://localhost:8001/mcp",
    auth=oauth_auth
) as (read, write, _):
    async with ClientSession(read, write) as session:
        await session.initialize()
```

## Transport Options

### STDIO Transport

- **Use for**: Local development, Claude Desktop integration
- **Pros**: Simple, no network configuration
- **Cons**: Single process, not suitable for web deployment

```python
# Server
mcp.run()  # Default is stdio

# Client
from mcp.client.stdio import stdio_client
```

### SSE Transport (Legacy)

- **Use for**: Legacy deployments
- **Note**: Being superseded by Streamable HTTP
- **Pros**: Simple HTTP-based
- **Cons**: Less scalable than Streamable HTTP

```python
# Server
mcp.run(transport="sse")
```

### Streamable HTTP Transport (Recommended)

- **Use for**: Production deployments, multi-node setups
- **Pros**: Stateless/stateful modes, resumability, scalability
- **Cons**: More complex setup

```python
# Server - Stateless with JSON (recommended)
mcp = FastMCP("Server", stateless_http=True, json_response=True)
mcp.run(transport="streamable-http")

# Client
from mcp.client.streamable_http import streamablehttp_client
```

## Low-Level Server API

For advanced use cases requiring full control:

```python
from mcp.server.lowlevel import Server
import mcp.types as types

server = Server("my-server")

@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="my_tool",
            description="My tool",
            inputSchema={
                "type": "object",
                "properties": {
                    "arg": {"type": "string"}
                }
            }
        )
    ]

@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    return [types.TextContent(type="text", text="Result")]
```

## Testing

### Test Servers with CliRunner

```python
from typer.testing import CliRunner
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Test Server")

@mcp.tool()
def add(a: int, b: int) -> int:
    return a + b

def test_server():
    runner = CliRunner()
    # Test your server logic
```

### Test Clients with Mock Servers

```python
import pytest
from mcp import ClientSession

@pytest.mark.asyncio
async def test_client():
    # Create mock server streams
    # Test client interactions
    pass
```

## Common Patterns

### Resource Subscriptions

```python
@mcp.resource("data://live")
def get_live_data() -> str:
    """Get live data that changes."""
    return get_current_data()

@mcp.tool()
async def update_data(value: str, ctx: Context) -> str:
    """Update data and notify subscribers."""
    update_current_data(value)
    await ctx.session.send_resource_updated(AnyUrl("data://live"))
    return "Updated"
```

### Pagination for Large Datasets

```python
from mcp.types import ListResourcesRequest, ListResourcesResult

@server.list_resources()
async def list_resources(request: ListResourcesRequest) -> ListResourcesResult:
    page_size = 10
    cursor = request.params.cursor if request.params else None
    start = 0 if cursor is None else int(cursor)
    end = start + page_size

    items = get_items()[start:end]
    next_cursor = str(end) if end < total_items() else None

    return ListResourcesResult(
        resources=items,
        nextCursor=next_cursor
    )
```

## Summary

- **Use FastMCP** for high-level server development
- **Use type hints** for automatic schema generation
- **Use Context** to access request information
- **Use lifespan** for resource management
- **Use Streamable HTTP** for production deployments
- **Handle errors gracefully** with meaningful messages
- **Send progress notifications** for long operations
- **Use structured output** for type-safe results
- **Test thoroughly** with proper test infrastructure
- **Follow OAuth patterns** for authentication when needed
