from fastmcp import Client
from fastmcp.tools import Tool
import asyncio
import json
import requests
import json
from dotenv import load_dotenv
from logging import Logger
from openai import AsyncOpenAI
import logging
from fastmcp import Client as MCPClient
import os
import asyncio
from servers.utility_servers.utils import _generate_mini_summary
import httpx 
import tiktoken

config = {
    "mcpServers": {
        "file": {
            "transport": "http",
            "url": "http://localhost:8030/mcp"
        },
        "api": {
            "transport": "http",
            "url": "http://localhost:8032/mcp"
        },
        "rag": {
            "transport": "http",
            "url": "http://localhost:8031/mcp"
        },
    }
}



logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    force=True,  # <-- IMPORTANT (Python 3.8+)
)

logger = logging.getLogger()

load_dotenv()

BASE_URL = "https://api.avalai.ir/v1/" #"http://localhost:11434/v1"
API_KEY = "aa-xJ9pYmEj0xNrvRND8y3QNRJqmhE90muFHwclBx8mxnHhODp0"
MODEL_NAME = "gpt-5.4-nano" #"qwen3:0.6b"
PROXY = "http://192.168.10.2:3129"
RESTRICTED_TOKEN = os.environ.get("RESTRICTED_TOKEN", "")


def count_chat_tokens(messages, model=MODEL_NAME):
    """
    messages = [
        {"role": "system", "content": "..."},
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."}
    ]
    """

    # enc = tiktoken.encoding_for_model(model)

    # Per-message overhead depends on model family
    tokens_per_message = 3
    tokens_per_name = 1

    total_tokens = 0

    for msg in messages:
        total_tokens += tokens_per_message

        for key, value in msg.items():
            if value:
                total_tokens += len(value) / 4

            if key == "name":
                total_tokens += tokens_per_name

    # every reply is primed with <assistant>
    total_tokens += 3

    return total_tokens


client = AsyncOpenAI(
    base_url=BASE_URL,
    api_key=API_KEY,
    http_client=httpx.AsyncClient(proxy=PROXY)
)


AGENT_SYSTEM_PROMPT = """
You are a research assistant with access to a local PDF library via MCP tools.
There are three MCP Servers. one for file access, one for API access, and one for RAG (retrieval-augmented generation) tasks.
Use tools whenever you see fit. Note that the file access is designed for keyword-based section retrieval, 
but the RAG server can perform semantic search across the document repository. The API server can validate paper titles and fetch metadata from Crossref.
Be concise. If the answer is not in the documents, say so explicitly.
Don't pass tokens to tools that can receive them, tokens are given automatically from the environment variable and you don't need to worry about them.
"""

def print_header(text: str) -> None:
    width = 70
    print("\n" + "=" * width)
    print(f"  {text}")
    print("=" * width)


def print_assistant(text: str) -> None:
    print(f"\n🤖 Assistant:\n{text}\n")


def print_user(text: str) -> None:
    print(f"\n👤 You: {text}")

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

async def main(client: AsyncOpenAI):
#     print_header("PDF Document Agent")
#     print("\nThis agent has access to your PDF file server.")
#     print("It will intelligently query documents using targeted section extraction.")
#     print("\nType 'quit' or 'exit' to end the session.\n")
    
    async with MCPClient(config) as mcp_client:
        
        conversation_history = [
            {"role": "system", "content": AGENT_SYSTEM_PROMPT}
        ]
        
        print("\n" + "-" * 70)
        print("Ready! Ask me anything about your documents.")
        print("Example: 'What documents do you have about machine learning?'")
        print("-" * 70)
        wait_for_user = True
        while True:
            try:
                if wait_for_user:
                    user_input = input("\n👤 You: ").strip()
                    
                    if not user_input:
                        continue
                    
                    if user_input.lower() in ("quit", "exit", "q"):
                        print("\nGoodbye!")
                        break
                    
                    # Add user message to history
                    conversation_history.append({
                        "role": "user",
                        "content": user_input,
                    })
                    print(f"current token count = {count_chat_tokens(conversation_history)}")
                available_tools = await mcp_client.list_tools()
                # for tool in available_tools:
                #     print(tool.meta['fastmcp']['tags'], tool.name)
                # exit()
                token_based_tools = [tool.name for tool in available_tools if "requires_token" in tool.meta.get("fastmcp", {}).get("tags", [])]
                # print("TOKEN-BASED TOOLS:", token_based_tools )
                available_tools = await convert_to_openai_too_format(available_tools)
                # Get response
                print("\n  (Processing...)")
                # print("Messages=", conversation_history)
                response = await client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=conversation_history,
                    temperature=0.4,
                    max_completion_tokens=4096,
                    tools=available_tools,
                    tool_choice="auto"
                )
                assistant_message = response.choices[0].message
                print_assistant(assistant_message.content)
                assistant_response = assistant_message.content
                if assistant_response:
                    # Update history with assistant response
                    conversation_history.append({
                        "role": "assistant",
                        "content": response.choices[0].message.content,
                    })
                    print(f"current token count = {count_chat_tokens(conversation_history)}")
                
                # print_assistant("Tool calls: " + str(assistant_message.tool_calls))
                if assistant_message.tool_calls:
                    tool_requests = {
                        "role": "assistant",
                        "tool_calls": [{"id": tool.id, 
                                        "type": "function", 
                                        "function": {"name": tool.function.name, 
                                                     "arguments": tool.function.arguments}
                                        } for tool in assistant_message.tool_calls]
                    }
                    conversation_history.append(tool_requests)
                    for tool_call in assistant_message.tool_calls:
                        tool_name = tool_call.function.name
                        tool_args = json.loads(tool_call.function.arguments)
                        print_assistant(f"---> TOOLCALL: NAME {tool_name} -- ARGS {tool_args}")
                        requires_token = tool_call.function.name in token_based_tools
                        if requires_token:
                            print(f"Tool {tool_call.function.name} requires a token. It will be provided automatically from the environment variable.")
                            tool_args['token'] = RESTRICTED_TOKEN
                        result = await mcp_client.call_tool(tool_name, tool_args)
                        result = result.content[0].text if result.content else ""
                        print_assistant(f"---> TOOLCALL RESULT: {_generate_mini_summary(result)}")                        
                        conversation_history.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": result,
                        })
                print(f"current token count = {count_chat_tokens(conversation_history)}\n")
                if assistant_message.tool_calls and not assistant_message.content:
                    wait_for_user = False
                    print("#### Still working with tool calls, don't need user prompt...")
                else:
                    wait_for_user = True

                
            except KeyboardInterrupt:
                print("\n\nInterrupted. Type 'quit' to exit.\n")
                continue
            except Exception as e:
                print(f"\n❌ Error: {e}\n")
                import traceback
                traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main(client))
