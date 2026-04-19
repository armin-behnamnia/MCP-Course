import requests
import json
from dotenv import load_dotenv
from logging import Logger
from openai import OpenAI
import logging
import httpx

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    force=True,  # <-- IMPORTANT (Python 3.8+)
)

logger = logging.getLogger()

load_dotenv()

BASE_URL = "https://api.avalai.ir/v1"
API_KEY = "aa-xJ9pYmEj0xNrvRND8y3QNRJqmhE90muFHwclBx8mxnHhODp0"
MODEL_NAME = "gemini-2.5-flash-lite" #"qwen3:0.6b"
PROXY = "http://192.168.10.2:3129"

client = OpenAI(
    base_url=BASE_URL,
    api_key=API_KEY,
    http_client=httpx.Client(proxy=PROXY)
)

prompt = "Explain MCP (Model Context Protocol) in one sentence. write"
prompt = "summarize the file nips_paper2021.pdf and pinpoint the keypoints"
response = client.chat.completions.create(
    model=MODEL_NAME,
    messages=[
        {"role": "user", "content": prompt}
    ],
    temperature=0.4,
    max_completion_tokens=1024
)

# ---- Log Output ----
logger.info("Model response: %s", response.choices[0].message.content)