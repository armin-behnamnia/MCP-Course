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
    async with Client("http://localhost:8787/mcp") as client:
        u = await client.list_prompts()
        available_tools = await client.list_tools()
        available_tools = await convert_to_openai_format(available_tools)
        for prompt in u:
            print(prompt.name)
        selected_prompt = await client.get_prompt("find_and_summarize", arguments={"keyword": "RL", "section": "Introduction"})
        messages = []
        for message in selected_prompt.messages:
            messages.append({"role": message.role, "content": message.content.text})
        print(messages)
        response = await openai_client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.4,
            max_completion_tokens=2048,
            tools = available_tools
        )
        print("Response:", response.choices[0].message.content)
        print("Tool Calls:", response.choices[0].message.tool_calls)

asyncio.run(main())