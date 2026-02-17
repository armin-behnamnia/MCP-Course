from fastmcp import Client
import asyncio
import json

async def main():
    async with Client("http://localhost:8787/mcp") as client:
        # result = await client.read_resource(uri='config://server')
        # print(json.loads(result[0].text))
        # result = await client.read_resource(uri='stats://files')
        # print(json.loads(result[0].text))
        result = await client.read_resource(uri='catalog://allowed')
        print(json.loads(result[0].text))

asyncio.run(main())