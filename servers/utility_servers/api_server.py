from fastmcp import FastMCP
from dotenv import load_dotenv
import json
import requests
import re
from pydantic import BaseModel, Field

PROXY = "http://192.168.10.2:3129" 

load_dotenv()

class ArticleInfoContent(BaseModel):
    publisher: str = Field("", description="The publisher of the article")
    doi: str = Field("", description="The DOI of the article")
    source: str = Field("", description="The source of the article metadata")
    title: str = Field("", description="The official title of the article")
    author: list[dict] = Field([], description="The authors of the article")
    year: int = Field(0, description="The year of publication")
    citations: int = Field(0, description="The number of citations")

class ArticleInfo(BaseModel):
    status: str = Field("ok", description="API response status")
    content: ArticleInfoContent = Field(..., description="The metadata content of the article")
    error: str = Field("", description="Error message if the article info is invalid")


mcp = FastMCP(
    name="MCP API Server",
    instructions=(
        "MCP API Server: Provides tools for PDF content extraction and metadata retrieval. "
        "Use the validate_and_fetch_metadata tool to retrieve official paper metadata from Crossref based on a title search. "
    ),
)


def sanitize_input(text: str) -> str:
    return re.sub(r'[^\w\s\-\.,]', '', text).strip()

def sanitize_data(data: str) -> str:
    """Removes potential script tags or malicious characters from API responses."""
    if not data: 
        return ""
    return re.sub(r'<[^>]*?>', '', data).strip()


@mcp.tool()
def validate_and_fetch_metadata(title: str) -> ArticleInfo:
    """
    Searches a paper title and fetches official metadata from Crossref.
    Handles malicious input and sanitizes external responses.
    
    Args:
        title (str): the title string of the paper.
    Returns:
        str: the serialized dictionary of official_title, publisher, year of the paper
    """
    title = sanitize_input(title)
    print(f"Searching for title: {title}")
    params = {"query.title": title, "rows": 1}
    try:
        response = requests.get("https://api.crossref.org/works", timeout=5, params=params, proxies={"http": PROXY, "https": PROXY})
    except requests.RequestException as e:
        return ArticleInfo(
            status="error",
            content=ArticleInfoContent(),
            error=f"Failed to fetch metadata: {str(e)}"
        )
        #{"error": f"Failed to fetch metadata: {str(e)}"}
    data = sanitize_data(response.text)
    try:
        data = json.loads(data)
    except json.JSONDecodeError as e:
        return ArticleInfo(
            status="error",
            content=ArticleInfoContent(),
            error=f"Failed to parse metadata: {str(e)}"
        )
    status = data.get('status', 'error')
    message = data.get('message', dict())
    results = message.get('items', [])
    if not results:
        return ArticleInfo(
            status="error",
            content=ArticleInfoContent(),
            error=f"No metadata found for title: {title}"
        )
    return ArticleInfo(
        status = status,
        content = ArticleInfoContent(
            publisher = results[0]['publisher'],
            doi = results[0]['DOI'],
            source = results[0]['source'],
            title = results[0]['title'][0],
            author= results[0]['author'],
            year = results[0]['created']['date-parts'][0][0],
            citations = results[0]['is-referenced-by-count']
        )
    )

if __name__ == "__main__":
    mcp.run(transport='http', port=8032)