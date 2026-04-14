"""MCP server exposing LightOn RAG operations as tools."""

import logging
import os
import time

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

from sdk import LightOn, LightOnError

load_dotenv()

PORT = int(os.environ.get("MCP_PORT", "8000"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("lighton-mcp")

api_key = os.environ["LIGHTON_API_KEY"]
client = LightOn(api_key=api_key)
log.info("LightOn client initialized")

mcp = FastMCP("LightOn MCP", stateless_http=True, json_response=True, port=PORT, host="0.0.0.0")


@mcp.tool()
def list_workspaces(
    page: int = 1,
    page_size: int = 20,
    workspace_type: str | None = None,
) -> str:
    """List available workspaces.

    Use this to discover workspace IDs before uploading files or querying.
    """
    log.info("list_workspaces page=%d page_size=%d type=%s", page, page_size, workspace_type)
    result = client.list_workspaces(
        page=page, page_size=page_size, workspace_type=workspace_type,
    )
    log.info("list_workspaces returned %d workspaces", result.count)
    lines = [f"Total: {result.count}"]
    for ws in result.results:
        lines.append(
            f"  [{ws.id}] {ws.name} (type={ws.workspace_type}, files={ws.files_count})"
        )
    return "\n".join(lines)


@mcp.tool()
def get_workspace(workspace_id: int) -> str:
    """Get details of a specific workspace by ID."""
    log.info("get_workspace id=%d", workspace_id)
    ws = client.get_workspace(workspace_id)
    return (
        f"ID: {ws.id}\n"
        f"Name: {ws.name}\n"
        f"Type: {ws.workspace_type}\n"
        f"Files: {ws.files_count}\n"
        f"Storage: {ws.used_storage}\n"
        f"Role: {ws.user_role}\n"
        f"Description: {ws.description}"
    )


@mcp.tool()
def list_files(
    workspace_id: int | None = None,
    search: str | None = None,
    status: str | None = None,
    page: int = 1,
    page_size: int = 20,
) -> str:
    """List indexed documents, optionally filtered by workspace, search term, or status."""
    log.info("list_files workspace=%s search=%s status=%s", workspace_id, search, status)
    result = client.list_files(
        workspace_id=workspace_id, search=search, status=status,
        page=page, page_size=page_size,
    )
    log.info("list_files returned %d files", result.count)
    lines = [f"Total: {result.count}"]
    for f in result.results:
        tags = ", ".join(t.name for t in f.tags) if f.tags else "none"
        lines.append(
            f"  [{f.id}] {f.filename} (status={f.status}, pages={f.total_pages}, tags={tags})"
        )
    return "\n".join(lines)


@mcp.tool()
def get_file(file_id: int, include_content: bool = False) -> str:
    """Get details of a specific file. Set include_content=True to see the parsed text."""
    log.info("get_file id=%d include_content=%s", file_id, include_content)
    f = client.get_file(file_id, include_content=include_content)
    result = (
        f"ID: {f.id}\n"
        f"Filename: {f.filename}\n"
        f"Title: {f.title}\n"
        f"Status: {f.status}\n"
        f"Pages: {f.total_pages}\n"
        f"Extension: {f.extension}\n"
        f"Workspace: {f.workspace.get('name', '?')}\n"
        f"Created: {f.created_at}"
    )
    if f.content:
        result += f"\n\nContent:\n{f.content}"
    return result


@mcp.tool()
def upload_file(file_path: str, workspace_id: int, title: str | None = None) -> str:
    """Upload a document for indexing into a workspace.

    The file will be processed asynchronously. Check its status with get_file().
    Supported formats: PDF, DOCX, PPTX, TXT, MD, HTML, XLSX, CSV.
    """
    log.info("upload_file path=%s workspace=%d title=%s", file_path, workspace_id, title)
    f = client.upload_file(file_path, workspace_id, title=title)
    log.info("upload_file created id=%d status=%s", f.id, f.status)
    return f"Uploaded: id={f.id} filename={f.filename} status={f.status}"


@mcp.tool()
def delete_file(file_id: int) -> str:
    """Delete a file by ID."""
    log.info("delete_file id=%d", file_id)
    client.delete_file(file_id)
    log.info("delete_file id=%d done", file_id)
    return f"Deleted file {file_id}"


@mcp.tool()
def retrieve(
    query: str,
    workspace_ids: list[int] | None = None,
    file_ids: list[int] | None = None,
    top_n: int = 5,
    top_k: int = 20,
) -> str:
    """Search indexed documents and return the most relevant text chunks.

    This is pure retrieval with no LLM generation. Use this when you
    want raw document excerpts to work with yourself.

    Args:
        query: Natural language search query.
        workspace_ids: Limit search to specific workspaces.
        file_ids: Limit search to specific files.
        top_n: Number of chunks to return (max 50).
        top_k: Candidates before reranking (max 100).
    """
    log.info("retrieve query=%r workspaces=%s top_n=%d", query, workspace_ids, top_n)
    t0 = time.monotonic()
    resp = client.retrieve(
        query, workspace_ids=workspace_ids, file_ids=file_ids,
        top_k=top_k, top_n=top_n,
    )
    elapsed = time.monotonic() - t0
    log.info("retrieve returned %d results in %.2fs", len(resp.results), elapsed)

    if not resp.results:
        return "No results found."

    lines = []
    for i, r in enumerate(resp.results, 1):
        doc_name = r.document.get("name", "unknown")
        lines.append(
            f"--- Result {i} (score={r.scoring.score:.3f}, doc={doc_name}) ---"
        )
        lines.append(r.chunk.text)
        lines.append("")
    return "\n".join(lines)


@mcp.tool()
def ask(
    query: str,
    workspace_ids: list[int] | None = None,
    file_ids: list[int] | None = None,
    force_tool: str | None = None,
    system_prompt_suffix: str | None = None,
) -> str:
    """Ask a question and get an LLM-generated answer grounded in your documents.

    This creates a conversation thread, sends the query to LightOn's LLM
    (which retrieves relevant documents automatically), and returns the answer.

    Args:
        query: The question to ask.
        workspace_ids: Scope document search to specific workspaces.
        file_ids: Scope document search to specific files.
        force_tool: Force a tool (e.g. "document_search").
        system_prompt_suffix: Extra instructions for the LLM.
    """
    log.info("ask query=%r workspaces=%s force_tool=%s", query, workspace_ids, force_tool)
    t0 = time.monotonic()
    try:
        turn = client.ask(
            query,
            workspace_ids=workspace_ids,
            file_ids=file_ids,
            force_tool=force_tool,
            system_prompt_suffix=system_prompt_suffix,
        )
    except LightOnError as e:
        log.error("ask failed: %s", e)
        return f"Error: {e}"

    elapsed = time.monotonic() - t0
    log.info("ask completed status=%s in %.2fs", turn.status, elapsed)

    answer = turn.answer or "No answer generated."
    reasoning = turn.reasoning

    result = f"Status: {turn.status}\n\n"
    if reasoning:
        result += f"Reasoning: {reasoning}\n\n"
    result += f"Answer: {answer}"
    return result


@mcp.tool()
def whoami() -> str:
    """Get the profile of the currently authenticated user."""
    log.info("whoami")
    user = client.me()
    log.info("whoami user=%s", user.username)
    return f"{user.first_name} {user.last_name} ({user.username}, {user.email})"


if __name__ == "__main__":
    log.info("Server running on port %d", PORT)
    mcp.run(transport="streamable-http")
