from fastmcp import Client
import asyncio


async def main():
    async with Client("http://localhost:8788/mcp") as client:
        result = await client.call_tool(name="echo", arguments={"message": "Hello, MCP!"})
asyncio.run(main())