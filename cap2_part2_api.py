from fastmcp import Client
import asyncio
import json

async def main():
    async with Client("http://localhost:8787/mcp") as client:
        all_tools = await client.list_tools()
        print("Available tools:", [item.name for item in all_tools])
        # result = await client.call_tool("search_research_papers", arguments={"query": "reinforcement learning", "n": 4})
        # result = await client.call_tool("extract_title", arguments={"file_id": "1020/Ackermann et al. - 2026 - Gradient Regularization Prevents Reward Hacking in Reinforcement Learning from Human Feedback and Ve.pdf"})
        result = await client.call_tool(name="validate_and_fetch_metadata", arguments={"title": "Inverse ^((Reinforcement++++/ Learning"})
        res = json.loads(result.content[0].text)
        print(json.dumps(res, indent=1))
        # title = result.content[0].text
        # print("Extracted Title:", title)
        # result = await client.call_tool("validate_and_fetch_metadata", arguments={"title": title})
        # print(result)
        # print(json.dumps(json.loads(result.content[0].text), indent=1))
        # result = await client.call_tool("validate_and_fetch_metadata", arguments={"title": "Contrastive Inverse Reinforcement Learning"})
        # print(json.dumps(json.loads(result.content[0].text), indent=1))

asyncio.run(main())