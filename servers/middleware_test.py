from fastmcp import FastMCP
from fastmcp.server.middleware import Middleware, MiddlewareContext

class LoggingMiddleware(Middleware):
    async def on_message(self, context: MiddlewareContext, call_next):
        print(f"→ {context.method}")
        result = await call_next(context)
        print(f"← {context.method}")
        return result

mcp = FastMCP("Middleware Test Server", "0.1")
mcp.add_middleware(LoggingMiddleware())

print("Starting PDF MCP Middleware Test File Server …")
mcp.run(transport='http', port=8788)
