from fastmcp import FastMCP

mcp = FastMCP(
    name="PDFFileServer",
    instructions=(
        "A read-only PDF file server. "
        "Use `list_pdf_files` to discover files by keyword. "
        "Use `read_pdf` to load a PDF and receive its content as Markdown. "
        "Restricted files require a valid access token."
    ),
)

@mcp.resource(
    "config://server",
    name="ServerConfig",
    mime_type="application/json",
)
def resource_server_config() -> str:
    return "{'name': 'pdfserver'}"

if __name__ == "__main__":
    mcp.run(transport='http', port=8787)