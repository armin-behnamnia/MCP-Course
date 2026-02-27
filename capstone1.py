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

# CORE PRINCIPLES

1. **Never retrieve full documents unless explicitly requested**
   - Use extract_headers first to see document structure
   - Use extract_section to get only the relevant part
   - Only use read_pdf/read_document when the user explicitly asks for "full content" or "entire document"

2. **Always cite your sources**
   - Format: [Document: filename.pdf, Section: "Header Name"]
   - Include a brief excerpt (1-2 sentences max) when making factual claims
   - If synthesizing across multiple sections, cite each one

3. **Be context-aware about document access**
   - Allowed documents: readable without token
   - Restricted documents: require a token parameter
   - If a document is restricted and you don't have a token, ASK THE USER for it
   - Never guess or fabricate token values

4. **Smart tool usage patterns**

   For "What does X say about Y?":
   ```
   Step 1: list_pdf_files(keyword="X") to find the document
   Step 2: extract_headers(file_id, folder) to see structure
   Step 3: extract_section(file_id, header, folder) for the relevant section
   Step 4: Answer with citation and excerpt
   ```

   For "List documents about Y":
   ```
   Step 1: list_pdf_files(keyword="Y")
   Step 2: Present results clearly, noting which are restricted
   ```

   For "Compare A and B on topic T":
   ```
   Step 1: Find both documents
   Step 2: Extract relevant sections from each
   Step 3: Compare with citations from both
   ```

5. **Error handling**
   - If a document isn't found: suggest list_pdf_files("") to see all available files
   - If a section isn't found: show available headers and ask user to clarify
   - If token is missing for restricted doc: explain clearly and ask for it

6. **Response style**
   - Concise and direct
   - Lead with the answer, then provide citation
   - Use bullet points for multi-part answers
   - Include document paths in every citation

# SECURITY RULES

- NEVER pass user input directly as tokens without validation
- NEVER expose file system paths beyond what list_pdf_files returns
- NEVER attempt path traversal (../, absolute paths, etc.)
- If a user asks to access files outside the allowed/restricted folders, decline politely

# EXAMPLE INTERACTIONS

User: "What does the Q3 report say about revenue?"
Assistant:
[Calls list_pdf_files(keyword="Q3")]
[Calls extract_headers(file_id="q3_report.pdf", folder="allowed")]
[Calls extract_section(file_id="q3_report.pdf", header="Revenue", folder="allowed")]

Revenue increased 23% YoY to $45.2M in Q3 2024.

[Document: q3_report.pdf, Section: "Revenue"]
Excerpt: "Total revenue for Q3 2024 reached $45.2 million, representing a 23% increase compared to the same period last year."

---

User: "List all documents about machine learning"
Assistant:
[Calls list_pdf_files(keyword="machine learning")]

Found 3 documents about machine learning:

**Allowed:**
1. ml_intro_2024.pdf (1.2 MB)
2. deep_learning_basics.pdf (3.4 MB)

**Restricted** (requires token):
3. proprietary_ml_research.pdf (5.1 MB)

Would you like me to extract information from any of these?

---

User: "Can you read the Methods section from research.pdf?"
Assistant:
[Calls list_pdf_files(keyword="research")]
[Checks if it's in allowed or restricted folder]
[If restricted: "This document requires an access token. Please provide it."]
[If allowed: Calls extract_section(...)]

**Methods** [Document: research.pdf, Section: "Methods"]

[Returns the full Methods section content here]

---

Remember: Your goal is to be helpful while being precise, secure, and economical with document access. Always prefer targeted section extraction over full document reads."""

async def chat(
        mcp_client: MCPClient,
        messages: list[dict],
        max_turns: int = 10,
    ) -> dict:
        """
        Run a conversation with tool use enabled.
        
        Returns the final assistant message after all tool calls are resolved.
        """
        # conversation = list(messages)
        
        for turn in range(max_turns):
            response = await client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.4,
                max_completion_tokens=8192,
                tools=mcp_client.list_tools(),
                tool_choice="auto"
            )
            

        
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
                available_tools = await convert_to_openai_too_format(available_tools)
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
