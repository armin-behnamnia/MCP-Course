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
        available_tools = await client.list_tools()
        ctx_tools = [tool for tool in available_tools if "context_reshaping" in tool.meta.get('fastmcp', {}).get('tags', [])]
        # print(ctx_tools)
        # available_tools = await convert_to_openai_format(ctx_tools)
        # result = await client.call_tool("list_pdf_files", arguments={"keyword": "", "token": 'MCI-ACADEMY-MCP-COURSE'})
        # print(json.dumps(json.loads(result.content[0].text), indent=1))
        # result = await client.call_tool("extract_headers", arguments={
        #     "file_id": "1011/Miao et al. - The Energy Loss Phenomenon in RLHFA New Perspective on Mitigating Reward Hacking.pdf",
        #     "folder": "restricted",
        #     "token": 'MCI-ACADEMY-MCP-COURSE'
        # })
        # print(json.dumps(json.loads(result.content[0].text), indent=1))
        result = await client.call_tool("summarize_filtered_sections", arguments={
            "keyword": "RL",
            "section_target": "introduction",
            "token": 'MCI-ACADEMY-MCP-COURSE'
        })
        print(json.loads(result.content[0].text))

        # #print(json.dumps(json.loads(result.content[0].text), indent=1))
        # print(result.content[0].text)
        # # messages = [{
        # #     "role": "user",
        # #     "content": [{
        # #         "type": "text",
        # #         "text": "Find a summary of RL in the repository. Consider only the `introduction` section of available documents."
        # #     }]
        # # }]
        # # response = await openai_client.chat.completions.create(
        # #     model=MODEL_NAME,
        # #     messages=messages,
        # #     temperature=0.4,
        # #     max_completion_tokens=2048,
        # #     tools = available_tools
        # # )
        # # print("Response:", response.choices[0].message.content)
        # # print("Tool Calls:", response.choices[0].message.tool_calls)

asyncio.run(main())