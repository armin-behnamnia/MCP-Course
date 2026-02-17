from fastmcp import Client
import asyncio
import json

async def main():
    async with Client("http://localhost:8787/mcp") as client:
        result = await client.call_tool(name='list_pdf_files', arguments={'keyword': 'entropy', 'token': 'MCI-ACADEMY-MCP-COURSE'})
        result = eval(result.content[0].text) if len(result.content) else []
        print(result)
        result = await client.call_tool(name='read_pdf', 
                                        arguments={
                                            'file_id': '4WHS992U/Deng et al. - 2025 - Decomposing the Entropy-Performance Exchange The Missing Keys to Unlocking Effective Reinforcement.pdf',
                                            'folder': 'restricted',  
                                            'token': 'MCI-ACADEMY-MCP-COURSE'})
        result = result.content[0].text if len(result.content) else ""
        print(result)

asyncio.run(main())