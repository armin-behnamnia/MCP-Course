from fastmcp import Client
import asyncio
import json

async def main():
    async with Client("http://localhost:8787/mcp") as client:
        resources = await client.list_resources()
        print(resources)

        tools = await client.list_tools()
        print(tools)
        
asyncio.run(main())