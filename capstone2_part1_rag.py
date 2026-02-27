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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    force=True,  # <-- IMPORTANT (Python 3.8+)
)

logger = logging.getLogger()

load_dotenv()

BASE_URL = "http://localhost:8015/v1"
API_KEY = ""
MODEL_NAME = "Qwen/Qwen3-30B-A3B-Thinking-2507-FP8"

client = AsyncOpenAI(
    base_url=BASE_URL,
    api_key=API_KEY,
)


AGENT_SYSTEM_PROMPT = """You are a precise, citation-focused research assistant with access to a PDF document server.
You should answer questions based on semantic searching of research papers with search_research_papers tool call.
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
                "name": tool.dict()["name"],
                "description": tool.dict().get("description", ""),
                "parameters": tool.dict().get("inputSchema", {}),
            },
        }
        for tool in mcp_tools
    ]
    return available_tools
    
async def main():
    print_header("PDF Document Agent")
    print("\nThis agent has access to your PDF file server.")
    print("It will intelligently query documents using targeted section extraction.")
    print("\nType 'quit' or 'exit' to end the session.\n")
    
    # Collect server environment variables
    # server_env = {}
    # for var in ["ALLOWED_DIR", "RESTRICTED_DIR", "RESTRICTED_TOKEN"]:
    #     val = os.environ.get(var)
    #     if val:
    #         server_env[var] = val
    
    # if not server_env.get("ALLOWED_DIR"):
    #     print("⚠️  Warning: ALLOWED_DIR not set. Using server default (/tmp/pdf_allowed)")
    # if not server_env.get("RESTRICTED_DIR"):
    #     print("⚠️  Warning: RESTRICTED_DIR not set. Using server default (/tmp/pdf_restricted)")
    
    async with MCPClient("http://localhost:8787/mcp") as mcp_client:
        
        conversation_history = []
        
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
                available_tools = await mcp_client.list_tools()
                available_tools = [tool for tool in available_tools if "rag" in tool.meta.get('fastmcp', {}).get('tags', [])]
                available_tools = await convert_to_openai_too_format(available_tools)
                # available_tools = []
                # Get response
                print("\n  (Processing...)")
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
                # Update history with assistant response
                conversation_history.append({
                    "role": "assistant",
                    "content": response.choices[0].message.content,
                })
                print_assistant("Tool calls: " + str(assistant_message.tool_calls))
                if assistant_message.tool_calls:
                    for tool_call in assistant_message.tool_calls:
                        tool_name = tool_call.function.name
                        tool_args = json.loads(tool_call.function.arguments)
                        print_assistant(f"---> TOOLCALL: NAME {tool_name} -- ARGS {tool_args}")
                        result = await mcp_client.call_tool(tool_name, tool_args)
                        result = result.content[0].text if result.content else ""
                        print_assistant(f"---> TOOLCALL RESULT: {_generate_mini_summary(result)}")                        
                        conversation_history.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": json.dumps(result),
                        })
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
    asyncio.run(main())
