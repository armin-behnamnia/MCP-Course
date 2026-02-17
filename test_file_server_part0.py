from fastmcp import Client
import asyncio
import json

async def main():
    async with Client("http://localhost:8787/mcp") as client:
        con_info = await client.initialize()
        tools = await client.list_tools()
        resources = await client.list_resources()
        resource_templates = await client.list_resource_templates()
        prompts = await client.list_prompts()
        print(resource_templates)
        tools_dict = [json.loads(tool.model_dump_json()) for tool in tools]
        resources_dict = [json.loads(res.model_dump_json()) for res in resources]
        resource_templates_dict = [json.loads(res.model_dump_json()) for res in resource_templates]
        prompts_dict = [json.loads(prompt.model_dump_json()) for prompt in prompts]
        manifest = {
            "tools": tools_dict,
            "resources": resources_dict,
            "prompts": prompts_dict,
            "resource_templates": resource_templates_dict
        }
        print(manifest)
        with open('manifest.json', 'w') as f:
            json.dump(manifest, f, indent=1)
        
        # with open('tools.json', 'w') as f:
        #     manifest_string = tools.model_dump_json()
        #     manifest_dict = json.loads(manifest_string)
        #     json.dump(manifest_dict, f, indent=1)
        # with open('resources.json', 'w') as f:
        #     manifest_string = resources.model_dump_json()
        #     manifest_dict = json.loads(manifest_string)
        #     json.dump(manifest_dict, f, indent=1)
        # with open('prompts.json', 'w') as f:
        #     manifest_string = prompts.model_dump_json()
        #     manifest_dict = json.loads(manifest_string)
        #     json.dump(manifest_dict, f, indent=1)
        
asyncio.run(main())