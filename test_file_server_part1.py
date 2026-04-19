from fastmcp import Client
import asyncio
import json

async def main():
    async with Client("http://localhost:8787/mcp") as client:
        result = await client.list_resources()
        for res in result:
            print(res.name, ":", res.uri)
            result = await client.read_resource(uri=res.uri)
            res = json.loads(result[0].text)
            print(f"Call result=", res['name'])
        result = await client.list_tools()
        for tool in result:
            print(tool.name, ":", tool.description)
        #result = await client.read_resource(uri='')        

asyncio.run(main())