from fastmcp import Client
import asyncio
import json
from openai import AsyncOpenAI
openai_client = AsyncOpenAI(
    base_url="http://localhost:8015/v1",
    api_key=""
)
MODEL_NAME="Qwen/Qwen3-30B-A3B-Thinking-2507-FP8"

async def convert_to_openai_format(mcp_tools):
    available_tools = [
        {
            "type": "function",
            "function": {
                "name": tool.dict()["name"],
                "description": tool.dict().get("description", ""),
                "parameters": tool.dict().get("inputSchema", {}),
            },
        }
        for tool in mcp_tools
    ]
    return available_tools

async def main():
    async with Client("http://localhost:8031/mcp") as client:
        # available_tools = await client.list_tools()
        # ctx_tools = [tool for tool in available_tools if "rag" in tool.meta.get('fastmcp', {}).get('tags', [])]
        result = await client.call_tool("search_research_papers", arguments={"query": "RLHF most critical challenge", "n": 3})
        print(result.content[0].text)
        # available_tools = await convert_to_openai_format(ctx_tools)
        # result = await client.call_tool("search_research_papers", arguments={"query": "reinforcement learning", "n": 4})
        # print(result.content[0].text)
        # print(json.dumps(json.loads(result.content[0].text), indent=1))


asyncio.run(main())