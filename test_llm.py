import requests
import json
from dotenv import load_dotenv
from logging import Logger
from openai import OpenAI
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    force=True,  # <-- IMPORTANT (Python 3.8+)
)

logger = logging.getLogger()

load_dotenv()

BASE_URL = "http://localhost:8000/v1"
API_KEY = ""
MODEL_NAME = "Qwen/Qwen3-14B"

client = OpenAI(
    base_url=BASE_URL,
    api_key=API_KEY,
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