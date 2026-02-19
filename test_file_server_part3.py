from fastmcp import Client
import asyncio
import json

async def main():
    async with Client("http://localhost:8787/mcp") as client:
        result = await client.call_tool(name='list_pdf_files', arguments={'keyword': 'ent', 'token': 'MCI-ACADEMY-MCP-COURSE'})
        result = eval(result.content[0].text) if len(result.content) else []
        for doc in result:
            headers = await client.call_tool(name='extract_headers', arguments={'file_id': doc['id'], 'folder': doc['folder'], 'token': 'MCI-ACADEMY-MCP-COURSE'})
            has_header = False
            for header in eval(headers.content[0].text):
                print(header)
                if header == 'introduction':
                    has_header = True
            print('-' * 100)
            if has_header:
                result = await client.call_tool(name='extract_section', arguments={'file_id': doc['id'], 'folder': doc['folder'], 'header': 'introduction', 'token': 'MCI-ACADEMY-MCP-COURSE'})
                print("Intro of the Doc:", result.content[0].text)
            else:
                print("No Intro for the Doc.")
            print('-' * 100)
            print('-' * 100)
        # result = await client.call_tool(name='read_pdf', 
        #                                 arguments={
        #                                     'file_id': '4WHS992U/Deng et al. - 2025 - Decomposing the Entropy-Performance Exchange The Missing Keys to Unlocking Effective Reinforcement.pdf',
        #                                     'folder': 'restricted',  
        #                                     'token': 'MCI-ACADEMY-MCP-COURSE'})
        # result = result.content[0].text if len(result.content) else ""
        # print(result)

asyncio.run(main())