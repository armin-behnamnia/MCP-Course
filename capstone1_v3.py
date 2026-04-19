import requests
import json
from dotenv import load_dotenv
from logging import Logger
from openai import AsyncOpenAI
import logging
from fastmcp import Client as MCPClient
import os
import asyncio
from servers.utils import _generate_mini_summary
import httpx 

# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s - %(levelname)s - %(message)s",
#     force=True,  # <-- IMPORTANT (Python 3.8+)
# )

# logger = logging.getLogger()

# load_dotenv()

BASE_URL = "https://api.avalai.ir/v1/" #"http://localhost:11434/v1"
API_KEY = "aa-xJ9pYmEj0xNrvRND8y3QNRJqmhE90muFHwclBx8mxnHhODp0"
MODEL_NAME = "gemini-2.5-flash-lite" #"qwen3:0.6b"
PROXY = "http://192.168.10.2:3129"

client = AsyncOpenAI(
    base_url=BASE_URL,
    api_key=API_KEY,
    http_client=httpx.AsyncClient(proxy=PROXY)
)


# AGENT_SYSTEM_PROMPT = """
# You are a research assistant with access to a local PDF library via MCP tools.
 
# When answering questions:
# 1. Use `list_pdf_files` first to discover relevant documents.
# 2. Use `extract_headers` to inspect a document's structure before reading it.
# 3. Use `extract_section` to fetch only the relevant section — never load full PDFs unless necessary.
# 4. Use `summarize_filtered_sections` for cross-document comparison of a keyword in a section.
# 5. Always cite sources as [filename, Section: X].
 
# Be concise. If the answer is not in the documents, say so explicitly.
# """
# 1. Use `list_pdf_files` first to discover relevant documents.
# 2. Use `extract_headers` to inspect a document's structure before reading it.
# 3. Use `extract_section` to fetch only the relevant section — never load full PDFs unless necessary.

AGENT_SYSTEM_PROMPT = """
You are a research assistant with access to a local PDF library via MCP tools.

List of tool calls:
1. Use `summarize_filtered_sections` for cross-document comparison of a keyword in a section.
The only sections that you should fetch are:
- introduction
- related work
- conclusion
- discussion
Be concise. If the answer is not in the documents, say so explicitly.
"""

def print_header(text: str) -> None:
    width = 70
    print("\n" + "=" * width)
    print(f"  {text}")
    print("=" * width)


# def print_assistant(text: str) -> None:
#     print(f"\n🤖 Assistant:\n{text}\n")


# def print_user(text: str) -> None:
#     print(f"\n👤 You: {text}")

async def convert_to_openai_too_format(mcp_tools):
    available_tools = [
        {
            "type": "function",
            "function": {
                "name": tool.model_dump()["name"],
                "description": tool.model_dump().get("description", ""),
                "parameters": tool.model_dump().get("inputSchema", {}),
            },
        }
        for tool in mcp_tools
    ]
    return available_tools
    
# async def main():
#     print_header("PDF Document Agent")
#     print("\nThis agent has access to your PDF file server.")
#     print("It will intelligently query documents using targeted section extraction.")
#     print("\nType 'quit' or 'exit' to end the session.\n")
    
#     # Collect server environment variables
#     # server_env = {}
#     # for var in ["ALLOWED_DIR", "RESTRICTED_DIR", "RESTRICTED_TOKEN"]:
#     #     val = os.environ.get(var)
#     #     if val:
#     #         server_env[var] = val
    
#     # if not server_env.get("ALLOWED_DIR"):
#     #     print("⚠️  Warning: ALLOWED_DIR not set. Using server default (/tmp/pdf_allowed)")
#     # if not server_env.get("RESTRICTED_DIR"):
#     #     print("⚠️  Warning: RESTRICTED_DIR not set. Using server default (/tmp/pdf_restricted)")
    
#     async with MCPClient("http://localhost:8787/mcp") as mcp_client:
        
#         conversation_history = []
        
#         print("\n" + "-" * 70)
#         print("Ready! Ask me anything about your documents.")
#         print("Example: 'What documents do you have about machine learning?'")
#         print("-" * 70)
#         wait_for_user = True
#         while True:
#             try:
#                 if wait_for_user:
#                     user_input = input("\n👤 You: ").strip()
                    
#                     if not user_input:
#                         continue
                    
#                     if user_input.lower() in ("quit", "exit", "q"):
#                         print("\nGoodbye!")
#                         break
                    
#                     # Add user message to history
#                     conversation_history.append({
#                         "role": "user",
#                         "content": user_input,
#                     })
#                 available_tools = await mcp_client.list_tools()
#                 available_tools = await convert_to_openai_too_format(available_tools)
#                 # Get response
#                 print("\n  (Processing...)")
#                 response = await client.chat.completions.create(
#                     model=MODEL_NAME,
#                     messages=conversation_history,
#                     temperature=0.4,
#                     max_completion_tokens=4096,
#                     tools=available_tools,
#                     tool_choice="auto"
#                 )
#                 assistant_message = response.choices[0].message
#                 print_assistant(assistant_message.content)
#                 # Update history with assistant response
#                 conversation_history.append({
#                     "role": "assistant",
#                     "content": response.choices[0].message.content,
#                 })
#                 print_assistant("Tool calls: " + str(assistant_message.tool_calls))
#                 if assistant_message.tool_calls:
#                     for tool_call in assistant_message.tool_calls:
#                         tool_name = tool_call.function.name
#                         tool_args = json.loads(tool_call.function.arguments)
#                         print_assistant(f"---> TOOLCALL: NAME {tool_name} -- ARGS {tool_args}")
#                         result = await mcp_client.call_tool(tool_name, tool_args)
#                         result = result.content[0].text if result.content else ""
#                         print_assistant(f"---> TOOLCALL RESULT: {_generate_mini_summary(result)}")                        
#                         conversation_history.append({
#                             "role": "tool",
#                             "tool_call_id": tool_call.id,
#                             "content": json.dumps(result),
#                         })
#                 if assistant_message.tool_calls and not assistant_message.content:
#                     wait_for_user = False
#                     print("#### Still working with tool calls, don't need user prompt...")
#                 else:
#                     wait_for_user = True

                
#             except KeyboardInterrupt:
#                 print("\n\nInterrupted. Type 'quit' to exit.\n")
#                 continue
#             except Exception as e:
#                 print(f"\n❌ Error: {e}\n")
#                 import traceback
#                 traceback.print_exc()

async def main():
    async with MCPClient("http://localhost:8787/mcp") as mcp_client:
        all_tools = await mcp_client.list_tools()
        all_tools = [tool for tool in all_tools if "context_reshaping" in tool.meta.get('fastmcp', {}).get('tags', [])]
        all_tools = await convert_to_openai_too_format(all_tools)
        user_message = input("--> Ask a question about the documents: ").strip()        
        messages = [
            {"role": "system", "content": AGENT_SYSTEM_PROMPT},
            {"role": "user", "content": f"{user_message} Search Documents"}#with keyword RLHF in the 'introduction' section and summarize using summarize_filtered_sections tool."},
        ]
        response = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.4,
            max_completion_tokens=4096,
            tools=all_tools,
            tool_choice="required"
        )
        tool_calls = response.choices[0].message.tool_calls
        content = response.choices[0].message.content
        # reasoning = response.choices[0].message.reasoning
        print("------ Content Response:", content[:50] + "..." if content else "None")
        # print("------ Response Reasoning:", reasoning[:50] + "..." if reasoning else "None")
        if tool_calls:
            for tool_call in tool_calls:
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)
                print(f"Calling Tool: {tool_name} with args {tool_args}...")
                if tool_name == 'summarize_filtered_sections':
                    tool_args['token'] = 'MCI-ACADEMY-MCP-COURSE'
                result = await mcp_client.call_tool(name=tool_name, arguments=tool_args)
                result = result.content[0].text if result.content else ""
                print(f"Tool Result: {result}")
                messages.append({
                    'role': 'tool',
                    'tool_call_id': tool_call.id,
                    'tool_name': tool_name,
                    'content': json.dumps(result)
                })
            response = await client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.4,
                max_completion_tokens=4096
            )
            print(f"--> {response.choices[0].message.content}")
                
        else:
            print("No tool calls made by the model.")
        
        
if __name__ == "__main__":
    asyncio.run(main())
