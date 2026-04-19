from fastmcp import Client
import asyncio
import json

async def main():
    async with Client("http://localhost:8787/mcp") as client:
        # result = await client.list_tools()
        # print(result[0].model_dump()['inputSchema']['properties'])
        result = await client.list_tools()
        pdf_list = await client.call_tool(name='list_pdf_files', arguments={'keyword': 'RL', 'token': 'MCI-ACADEMY-MCP-COURSE'})
        pdf_list = json.loads(pdf_list.content[0].text)
        pdf0 = pdf_list[0]
        # print(result[1].name, result[1].model_dump()['inputSchema']['properties'])
        pdf_content = await client.call_tool(name='read_pdf', arguments={'file_id': pdf0['id'], 'folder': pdf0['folder'], 'token': 'MCI-ACADEMY-MCP-COURSE'})
        print(pdf_content.content[0].text)
        # for tool in result:
        #     print(tool.name, ":", tool.description, "Arguments")
        # result = await client.call_tool(name='list_pdf_files', arguments={'keyword': 'Energy Loss', 'token': 'MCI-ACADEMY-MCP-COURSE'})
        # result = eval(result.content[0].text) if len(result.content) else []
        # print(result)
        # result = await client.call_tool(name='read_pdf', 
        #                                 arguments={
        #                                     'file_id': '1011/Miao et al. - The Energy Loss Phenomenon in RLHFA New Perspective on Mitigating Reward Hacking.pdf',
        #                                     'folder': 'restricted',  
        #                                     'token': 'MCI-ACADEMY-MCP-COURSE'})
        # result = result.content[0].text if len(result.content) else ""
        # print(result)

asyncio.run(main())