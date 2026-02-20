"""Pydantic models for Total Recall."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class MemoryRecord(BaseModel):
    """A memory record stored in the database."""

    key: str
    value: str
    content_type: str = "text/plain"
    tags: list[str] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime
    accessed_at: datetime
    access_count: int = 0
    embedding: Optional[list[float]] = None
    embedding_model: Optional[str] = None


class MemoryListItem(BaseModel):
    """Summary of a memory for list responses."""

    key: str
    content_type: str
    tags: list[str]
    updated_at: datetime


class MemorySearchResult(BaseModel):
    """A memory result from semantic search."""

    key: str
    value: str
    content_type: str
    score: float
    tags: list[str]


class MemoryFulltextResult(BaseModel):
    """A memory result from fulltext search."""

    key: str
    value: str
    snippet: str
    content_type: str
    tags: list[str]


class HistoryEntry(BaseModel):
    """A version history entry."""

    value: str
    content_type: Optional[str]
    changed_at: datetime


class TagCount(BaseModel):
    """Tag with usage count."""

    tag: str
    count: int


# Response models


class SetResponse(BaseModel):
    """Response from memory_set."""

    success: bool
    created: bool
    changed: bool
    blocked: bool = False
    key: str
    size_bytes: int
    previous_value: Optional[str] = None
    previous_size_bytes: Optional[int] = None
    warnings: list[str] = Field(default_factory=list)


class SetFromFileResponse(BaseModel):
    """Response from memory_set_from_file."""

    success: bool
    created: bool
    changed: bool
    blocked: bool = False
    key: str
    file_path: str
    file_size_bytes: int
    content_type: str
    previous_value: Optional[str] = None
    previous_size_bytes: Optional[int] = None
    warnings: list[str] = Field(default_factory=list)


class GetResponse(BaseModel):
    """Response from memory_get."""

    key: str
    value: str
    content_type: str
    tags: list[str]
    created_at: str
    updated_at: str
    accessed_at: str
    access_count: int
    warnings: list[str] = Field(default_factory=list)


class DeleteResponse(BaseModel):
    """Response from memory_delete."""

    success: bool
    deleted: bool
    warnings: list[str] = Field(default_factory=list)


class ListResponse(BaseModel):
    """Response from memory_list."""

    memories: list[MemoryListItem]
    total: int
    warnings: list[str] = Field(default_factory=list)


class SearchResponse(BaseModel):
    """Response from memory_search."""

    results: list[MemorySearchResult]
    warnings: list[str] = Field(default_factory=list)


class FulltextResponse(BaseModel):
    """Response from memory_fulltext."""

    results: list[MemoryFulltextResult]
    total: int
    warnings: list[str] = Field(default_factory=list)


class TagsResponse(BaseModel):
    """Response from memory_tags."""

    tags: list[TagCount]
    warnings: list[str] = Field(default_factory=list)


class StatsResponse(BaseModel):
    """Response from memory_stats."""

    total_memories: int
    total_tags: int
    total_size_bytes: int
    embedded_count: int
    oldest: Optional[str]
    newest: Optional[str]
    warnings: list[str] = Field(default_factory=list)


class EmbedResponse(BaseModel):
    """Response from memory_embed."""

    embedded_count: int
    failed_count: int
    warnings: list[str] = Field(default_factory=list)


class HistoryResponse(BaseModel):
    """Response from memory_history."""

    history: list[HistoryEntry]
    truncated: bool
    warnings: list[str] = Field(default_factory=list)


class ImportResult(BaseModel):
    """Internal result from deserialize_memories — shared by restore + listen."""

    restored: int = 0
    skipped: int = 0
    filtered: int = 0
    conflicts: int = 0
    embedded: int = 0
    embed_failed: int = 0
    keys_stored: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class BackupResponse(BaseModel):
    """Response from memory_backup."""

    success: bool
    path: str
    memory_count: int
    size_bytes: int
    warnings: list[str] = Field(default_factory=list)


class RestoreResponse(BaseModel):
    """Response from memory_restore."""

    success: bool
    restored: int = 0
    skipped: int = 0
    conflicts: int = 0
    warnings: list[str] = Field(default_factory=list)


class ListenResponse(BaseModel):
    """Response from memory_listen."""

    success: bool
    received: int = 0
    filtered: int = 0
    skipped: int = 0
    keys_stored: list[str] = Field(default_factory=list)
    elapsed: float = 0.0
    timed_out: bool = False
    error: Optional[str] = None
    warnings: list[str] = Field(default_factory=list)


class SendResponse(BaseModel):
    """Response from memory_send."""

    success: bool
    sent: int = 0
    accepted: int = 0
    filtered: int = 0
    skipped: int = 0
    error: Optional[str] = None
    warnings: list[str] = Field(default_factory=list)


class ErrorResponse(BaseModel):
    """Error response for fatal errors."""

    success: bool = False
    error: str
