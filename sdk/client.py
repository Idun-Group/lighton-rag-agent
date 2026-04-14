"""LightOn Paradigm API client for RAG workflows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx

from sdk.exceptions import (
    AuthenticationError,
    LightOnError,
    NotFoundError,
    RateLimitError,
    ValidationError,
)
from sdk.models import (
    File,
    FileList,
    RetrieveResponse,
    Thread,
    ThreadList,
    Turn,
    TurnList,
    User,
    Workspace,
    WorkspaceList,
)

DEFAULT_BASE_URL = "https://paradigm.lighton.ai"


class LightOn:
    """Client for the LightOn Paradigm v3 API.

    Covers the core RAG workflow: manage workspaces, upload and index
    documents, retrieve relevant chunks, and generate answers using
    LightOn's LLM via the threads API.

    Usage::

        from sdk import LightOn

        with LightOn(api_key="your_key") as client:
            # Upload a document
            client.upload_file("report.pdf", workspace_id=1)

            # Search for chunks
            results = client.retrieve("quarterly revenue", workspace_ids=[1])

            # Ask a question and get an LLM answer grounded in your docs
            turn = client.ask("What was Q1 revenue?", workspace_ids=[1])
            print(turn.answer)

    Args:
        api_key: Your Paradigm API key (Bearer token).
        base_url: Base URL of the Paradigm instance.
        timeout: HTTP timeout in seconds for all requests.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 30.0,
    ):
        self._base_url = base_url.rstrip("/")
        self._client = httpx.Client(
            base_url=f"{self._base_url}/api/v3",
            headers={
                "Authorization": f"Bearer {api_key}",
            },
            timeout=timeout,
        )

    def close(self) -> None:
        """Close the underlying HTTP connection pool."""
        self._client.close()

    def __enter__(self) -> LightOn:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def _handle_response(self, response: httpx.Response) -> Any:
        if response.status_code == 204:
            return None

        try:
            body = response.json()
        except json.JSONDecodeError:
            body = None

        if response.is_success:
            return body

        msg = str(body) if body else response.text
        match response.status_code:
            case 401:
                raise AuthenticationError(msg, response.status_code, body)
            case 403:
                raise AuthenticationError(msg, response.status_code, body)
            case 404:
                raise NotFoundError(msg, response.status_code, body)
            case 400:
                raise ValidationError(msg, response.status_code, body)
            case 429:
                raise RateLimitError(msg, response.status_code, body)
            case _:
                raise LightOnError(msg, response.status_code, body)

    # Workspaces

    def list_workspaces(
        self,
        *,
        page: int = 1,
        page_size: int = 20,
        name: str | None = None,
        workspace_type: str | None = None,
    ) -> WorkspaceList:
        """List workspaces accessible to the authenticated user.

        Args:
            page: Page number (starts at 1).
            page_size: Results per page (max 100).
            name: Filter by workspace name prefix.
            workspace_type: Filter by type ("personal", "company", or "custom").
        """
        params: dict[str, Any] = {"page": page, "page_size": page_size}
        if name is not None:
            params["name"] = name
        if workspace_type is not None:
            params["workspace_type"] = workspace_type

        data = self._handle_response(self._client.get("/workspaces", params=params))
        return WorkspaceList.model_validate(data)

    def get_workspace(self, workspace_id: int) -> Workspace:
        """Get a single workspace by ID."""
        data = self._handle_response(self._client.get(f"/workspaces/{workspace_id}"))
        return Workspace.model_validate(data)

    def create_workspace(self, name: str, description: str = "") -> Workspace:
        """Create a new workspace."""
        data = self._handle_response(
            self._client.post(
                "/workspaces", json={"name": name, "description": description}
            )
        )
        return Workspace.model_validate(data)

    # Files

    def upload_file(
        self,
        file_path: str | Path,
        workspace_id: int,
        *,
        title: str | None = None,
        filename: str | None = None,
        tags: list[int] | None = None,
    ) -> File:
        """Upload a document for indexing.

        The file is sent as multipart form data. Processing is async;
        the returned File will have status "pending". Poll with
        get_file() until status becomes "embedded" (ready for retrieval)
        or a failure state.

        Supported formats: PDF, DOCX, PPTX, TXT, MD, HTML, XLSX, CSV, and more.
        Max size: 25MB per file (instance-configurable).

        Args:
            file_path: Path to the file on disk.
            workspace_id: Target workspace ID.
            title: Custom title (defaults to filename without extension).
            filename: Override the uploaded filename.
            tags: List of tag IDs to assign.
        """
        file_path = Path(file_path)
        form_data: dict[str, Any] = {"workspace_id": str(workspace_id)}
        if title is not None:
            form_data["title"] = title
        if filename is not None:
            form_data["filename"] = filename
        if tags is not None:
            form_data["tags[]"] = [str(t) for t in tags]

        with open(file_path, "rb") as f:
            data = self._handle_response(
                self._client.post(
                    "/files", data=form_data, files={"file": (file_path.name, f)}
                )
            )
        return File.model_validate(data)

    def list_files(
        self,
        *,
        page: int = 1,
        page_size: int = 50,
        workspace_id: int | None = None,
        status: str | None = None,
        search: str | None = None,
        extension: str | None = None,
    ) -> FileList:
        """List files accessible to the authenticated user.

        Args:
            page: Page number.
            page_size: Results per page (max 500).
            workspace_id: Filter to a specific workspace.
            status: Filter by processing status
                    ("pending", "parsing", "embedded", "fail", etc.).
            search: Fuzzy search across filenames and content.
            extension: Filter by file extension (e.g. "pdf").
        """
        params: dict[str, Any] = {"page": page, "page_size": page_size}
        if workspace_id is not None:
            params["workspace_id"] = workspace_id
        if status is not None:
            params["status"] = status
        if search is not None:
            params["search"] = search
        if extension is not None:
            params["extension"] = extension

        data = self._handle_response(self._client.get("/files", params=params))
        return FileList.model_validate(data)

    def get_file(self, file_id: int, *, include_content: bool = False) -> File:
        """Get a single file by ID.

        Args:
            file_id: The file ID.
            include_content: If True, the parsed text content is included
                             in the response (can be large).
        """
        params: dict[str, Any] = {}
        if include_content:
            params["include_content"] = True

        data = self._handle_response(
            self._client.get(f"/files/{file_id}", params=params)
        )
        return File.model_validate(data)

    def delete_file(self, file_id: int) -> None:
        """Delete a file by ID."""
        self._handle_response(self._client.delete(f"/files/{file_id}"))

    def update_file(
        self,
        file_id: int,
        *,
        title: str | None = None,
        tags: list[int] | None = None,
        external_metadata: dict[str, Any] | None = None,
    ) -> File:
        """Update a file's metadata.

        Args:
            file_id: The file ID.
            title: New title.
            tags: Replace all tags with this list. Pass [0] to remove all.
            external_metadata: Arbitrary metadata dict (merged with existing).
        """
        form_data: dict[str, Any] = {}
        if title is not None:
            form_data["title"] = title
        if tags is not None:
            form_data["tags[]"] = [str(t) for t in tags]
        if external_metadata is not None:
            form_data["external_metadata"] = external_metadata

        data = self._handle_response(
            self._client.patch(f"/files/{file_id}", data=form_data)
        )
        return File.model_validate(data)

    # Retrieval

    def retrieve(
        self,
        query: str,
        *,
        workspace_ids: list[int] | None = None,
        file_ids: list[int] | None = None,
        tag_ids: list[int] | None = None,
        top_k: int = 20,
        top_n: int = 10,
        mode: str = "text",
        skip_rerank: bool = False,
    ) -> RetrieveResponse:
        """Search indexed documents and return ranked chunks.

        This is the core RAG retrieval endpoint. It performs hybrid search
        (dense embeddings + BM25 lexical), then reranks results.

        Returns chunks only, no LLM generation. Use ask() if you want
        the LLM to generate an answer from the retrieved context.

        Args:
            query: Natural language search query.
            workspace_ids: Limit search to these workspaces.
            file_ids: Limit search to these files.
            tag_ids: Limit search to files with these tags.
            top_k: Candidates before reranking (1 to 100).
            top_n: Final chunks returned after reranking (1 to 50).
            mode: "text" for standard search, "vision" for image search.
            skip_rerank: If True, skip the reranker and return raw results.
        """
        body: dict[str, Any] = {
            "query": query,
            "top_k": top_k,
            "top_n": top_n,
            "mode": mode,
            "skip_rerank": skip_rerank,
        }
        if workspace_ids is not None:
            body["workspace_id"] = workspace_ids
        if file_ids is not None:
            body["file_id"] = file_ids
        if tag_ids is not None:
            body["tag_id"] = tag_ids

        data = self._handle_response(self._client.post("/retrieve", json=body))
        return RetrieveResponse.model_validate(data)

    # Threads

    def ask(
        self,
        query: str,
        *,
        workspace_ids: list[int] | None = None,
        file_ids: list[int] | None = None,
        tag_ids: list[int] | None = None,
        agent_id: int | None = None,
        force_tool: str | None = None,
        system_prompt_suffix: str | None = None,
        poll_interval: float = 2.0,
        max_polls: int = 30,
    ) -> Turn:
        """Ask a question and get an LLM-generated answer.

        This is the high-level RAG method. It creates a thread, sends
        your query to the LLM (which can retrieve documents, run tools,
        and reason), then polls until the answer is ready.

        The returned Turn has convenience properties:
            turn.answer    the final text response
            turn.reasoning the model's reasoning (if any)

        Uses background mode internally because the synchronous endpoint
        has a known server-side bug that returns 500 even on success.

        Args:
            query: The question to ask.
            workspace_ids: Scope document search to these workspaces.
            file_ids: Scope document search to these files.
            tag_ids: Scope document search to files with these tags.
            agent_id: Agent to use. Falls back to the company default.
            force_tool: Force a specific tool (e.g. "document_search").
            system_prompt_suffix: Extra instructions appended to the system prompt.
            poll_interval: Seconds between status checks.
            max_polls: Max status checks before returning whatever we have.
        """
        import time

        body: dict[str, Any] = {"query": query, "background": True}
        if workspace_ids is not None:
            body["workspace_ids"] = workspace_ids
        if file_ids is not None:
            body["file_ids"] = file_ids
        if tag_ids is not None:
            body["tag_ids"] = tag_ids
        if agent_id is not None:
            body["agent_id"] = agent_id
        if force_tool is not None:
            body["force_tool"] = force_tool
        if system_prompt_suffix is not None:
            body["system_prompt_suffix"] = system_prompt_suffix

        data = self._handle_response(self._client.post("/threads/turns", json=body))
        turn = Turn.model_validate(data)

        if turn.status == "completed":
            return turn

        for _ in range(max_polls):
            time.sleep(poll_interval)
            turns = self.list_turns(turn.thread)
            latest = turns.results[-1]
            if latest.status in ("completed", "failed", "cancelled"):
                return latest

        return turn

    def create_thread(
        self, *, name: str | None = None, agent_id: int | None = None
    ) -> Thread:
        """Create an empty conversation thread."""
        body: dict[str, Any] = {}
        if name is not None:
            body["name"] = name
        if agent_id is not None:
            body["agent_id"] = agent_id

        data = self._handle_response(self._client.post("/threads", json=body))
        return Thread.model_validate(data)

    def list_threads(self, *, limit: int = 10) -> ThreadList:
        """List conversation threads, most recent first."""
        data = self._handle_response(
            self._client.get("/threads", params={"limit": limit})
        )
        return ThreadList.model_validate(data)

    def get_thread(self, thread_id: str) -> Thread:
        """Get a single thread by ID."""
        data = self._handle_response(self._client.get(f"/threads/{thread_id}"))
        return Thread.model_validate(data)

    def delete_thread(self, thread_id: str) -> None:
        """Delete a thread and all its turns."""
        self._handle_response(self._client.delete(f"/threads/{thread_id}"))

    def create_turn(
        self,
        thread_id: str,
        query: str,
        *,
        force_tool: str | None = None,
        workspace_ids: list[int] | None = None,
        background: bool = True,
    ) -> Turn:
        """Add a follow-up turn to an existing thread.

        Unlike ask(), this continues a conversation rather than
        starting a new one. The LLM sees the full thread history.

        Args:
            thread_id: The thread to continue.
            query: The follow-up question.
            force_tool: Force a specific tool.
            workspace_ids: Scope document search to these workspaces.
            background: Process asynchronously (recommended due to server bug).
        """
        body: dict[str, Any] = {"query": query, "background": background}
        if force_tool is not None:
            body["force_tool"] = force_tool
        if workspace_ids is not None:
            body["workspace_ids"] = workspace_ids

        data = self._handle_response(
            self._client.post(f"/threads/{thread_id}/turns", json=body)
        )
        return Turn.model_validate(data)

    def list_turns(self, thread_id: str) -> TurnList:
        """List all turns in a thread."""
        data = self._handle_response(self._client.get(f"/threads/{thread_id}/turns"))
        return TurnList.model_validate(data)

    # User

    def me(self) -> User:
        """Get the profile of the currently authenticated user."""
        data = self._handle_response(self._client.get("/users/me"))
        return User.model_validate(data["profile"])
