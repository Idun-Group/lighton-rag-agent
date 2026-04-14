"""Data models for the LightOn SDK."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class WorkspaceSummary(BaseModel):
    language: str = ""
    summary: str = ""


class Workspace(BaseModel):
    id: int = 0
    name: str = ""
    workspace_type: str = ""
    document_upload_method: str = ""
    description: str = ""
    created_at: str = ""
    updated_at: str = ""
    user_role: str = ""
    used_storage: float = 0.0
    files_count: int = 0
    summaries: list[WorkspaceSummary] = Field(default_factory=list)


class Tag(BaseModel):
    id: int = 0
    name: str = ""
    auto_assigned: bool = False


class CreatedBy(BaseModel):
    id: int = 0
    first_name: str = ""
    last_name: str = ""
    username: str = ""


class File(BaseModel):
    id: int = 0
    filename: str = ""
    workspace: dict[str, Any] = Field(default_factory=dict)
    title: str = ""
    extension: str = ""
    status: str = ""
    status_vision: str | None = None
    created_at: str = ""
    updated_at: str = ""
    total_pages: int = 0
    size: int | None = None
    tags: list[Tag] = Field(default_factory=list)
    created_by: CreatedBy | None = None
    upload_session_uuid: str = ""
    external_metadata: dict[str, Any] | None = None
    content: str | None = None
    summaries: list[dict[str, Any]] = Field(default_factory=list)
    message: str = ""


class Chunk(BaseModel):
    id: int = 0
    uuid: str = ""
    content_id: str = ""
    text: str = ""
    chunk_type: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str = ""
    updated_at: str = ""


class ChunkScoring(BaseModel):
    score: float = 0.0
    distance: float = 0.0
    lexical_score: float = 0.0
    certainty: float = 0.0


class RetrieveResult(BaseModel):
    chunk: Chunk = Field(default_factory=Chunk)
    scoring: ChunkScoring = Field(default_factory=ChunkScoring)
    workspace: dict[str, Any] = Field(default_factory=dict)
    document: dict[str, Any] = Field(default_factory=dict)


class RetrieveResponse(BaseModel):
    results: list[RetrieveResult] = Field(default_factory=list)


class User(BaseModel):
    id: int = 0
    first_name: str = ""
    last_name: str = ""
    username: str = ""
    email: str = ""


class MessagePart(BaseModel):
    type: str = ""
    text: str | None = None
    reasoning: str | None = None
    tool_call: dict[str, Any] | None = None
    document: dict[str, Any] | None = None
    workspace: dict[str, Any] | None = None
    tag: dict[str, Any] | None = None


class Message(BaseModel):
    id: str = ""
    object: str = ""
    role: str = ""
    parts: list[MessagePart] = Field(default_factory=list)
    created_at: str = ""


class Turn(BaseModel):
    id: str = ""
    object: str = ""
    thread: str = ""
    status: str = ""
    error: dict[str, Any] | None = None
    messages: list[Message] = Field(default_factory=list)
    created_at: str = ""
    liked: bool | None = None

    @property
    def answer(self) -> str | None:
        """Extract the final text answer from the assistant's last message."""
        for msg in reversed(self.messages):
            if msg.role == "assistant":
                for part in reversed(msg.parts):
                    if part.text:
                        return part.text
        return None

    @property
    def reasoning(self) -> str | None:
        """Extract the reasoning from the assistant's message, if any."""
        for msg in reversed(self.messages):
            if msg.role == "assistant":
                for part in msg.parts:
                    if part.reasoning:
                        return part.reasoning
        return None


class Thread(BaseModel):
    id: str = ""
    object: str = ""
    name: str = ""
    status: str = ""
    ml_model: str = ""
    agent_id: int = 0
    agent_name: str = ""
    creation_source: str = ""
    created_at: str = ""
    updated_at: str = ""
    is_ephemeral: bool = False


class ThreadList(BaseModel):
    count: int | None = None
    next: str | None = None
    previous: str | None = None
    results: list[Thread] = Field(default_factory=list)


class TurnList(BaseModel):
    count: int | None = None
    next: str | None = None
    previous: str | None = None
    results: list[Turn] = Field(default_factory=list)


class WorkspaceList(BaseModel):
    count: int = 0
    next: str | None = None
    previous: str | None = None
    results: list[Workspace] = Field(default_factory=list)


class FileList(BaseModel):
    count: int = 0
    next: str | None = None
    previous: str | None = None
    results: list[File] = Field(default_factory=list)
