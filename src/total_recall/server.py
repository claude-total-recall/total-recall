"""MCP server for Total Recall."""

import os
from pathlib import Path

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from . import db
from . import embeddings as emb
from .models import (
    DeleteResponse,
    EmbedResponse,
    ErrorResponse,
    FulltextResponse,
    GetResponse,
    HistoryResponse,
    ListResponse,
    SearchResponse,
    SetFromFileResponse,
    SetResponse,
    StatsResponse,
    TagsResponse,
)

app = Server("total-recall")


def auto_embed_enabled() -> bool:
    """Check if auto-embedding is enabled."""
    return os.environ.get("TOTAL_RECALL_AUTO_EMBED", "true").lower() == "true"


@app.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools."""
    return [
        Tool(
            name="memory_set",
            description="""Store or update a memory in Total Recall.

Use this to save:
- User preferences and conventions
- Project decisions and architecture notes
- Reference information that may be needed later

Keys should use dot notation (e.g., "project.myapp.conventions").
Content can be plain text, markdown, or JSON.
Embeddings are generated automatically for semantic search.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "description": "Unique identifier for the memory",
                    },
                    "value": {
                        "type": "string",
                        "description": "Content to store",
                    },
                    "content_type": {
                        "type": "string",
                        "description": "MIME type hint (text/plain, text/markdown, application/json)",
                        "default": "text/plain",
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Categorization labels",
                    },
                    "embed": {
                        "type": "boolean",
                        "description": "Generate embedding for semantic search",
                        "default": True,
                    },
                },
                "required": ["key", "value"],
            },
        ),
        Tool(
            name="memory_set_from_file",
            description="""Store file contents as a memory.

Reads the file directly and stores its contents verbatim.
Use this to preserve full fidelity without summarization.

Content type is auto-detected from file extension:
- .md → text/markdown
- .json → application/json
- .py, .js, etc → text/plain""",
            inputSchema={
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "description": "Unique identifier for the memory",
                    },
                    "file_path": {
                        "type": "string",
                        "description": "Absolute path to the file",
                    },
                    "content_type": {
                        "type": "string",
                        "description": "Override auto-detected content type",
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Categorization labels",
                    },
                    "embed": {
                        "type": "boolean",
                        "description": "Generate embedding for semantic search",
                        "default": True,
                    },
                },
                "required": ["key", "file_path"],
            },
        ),
        Tool(
            name="memory_get",
            description="""Retrieve a specific memory by its exact key.

Use when you know the exact key. For discovery or fuzzy matching,
use memory_search (semantic) or memory_list (browse) instead.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "description": "The exact key to retrieve",
                    },
                },
                "required": ["key"],
            },
        ),
        Tool(
            name="memory_delete",
            description="Remove a memory by its key.",
            inputSchema={
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "description": "The key to delete",
                    },
                },
                "required": ["key"],
            },
        ),
        Tool(
            name="memory_list",
            description="""Browse memories with optional filtering.

Use to:
- See all memories under a prefix: pattern="project.myapp.*"
- Find memories with a tag: tag="architecture"
- Get an overview of stored knowledge

Pattern uses * as wildcard. Without wildcard, pattern is exact match.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Key pattern with wildcards (* = any chars)",
                    },
                    "tag": {
                        "type": "string",
                        "description": "Filter by tag",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max results (default 50)",
                        "default": 50,
                    },
                    "offset": {
                        "type": "integer",
                        "description": "Pagination offset",
                        "default": 0,
                    },
                },
            },
        ),
        Tool(
            name="memory_search",
            description="""Search memories by meaning using semantic similarity.

Use natural language queries like:
- "user's coding style preferences"
- "how authentication works in this project"
- "database naming conventions"

Returns memories ranked by relevance. Good for discovering
related context when you don't know exact keys.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language query",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max results (default 10)",
                        "default": 10,
                    },
                    "threshold": {
                        "type": "number",
                        "description": "Minimum similarity score 0-1 (default 0.3)",
                        "default": 0.3,
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="memory_fulltext",
            description="Full-text search (substring/keyword match) across memories.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search terms",
                    },
                    "search_keys": {
                        "type": "boolean",
                        "description": "Search in keys",
                        "default": True,
                    },
                    "search_values": {
                        "type": "boolean",
                        "description": "Search in values",
                        "default": True,
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max results (default 20)",
                        "default": 20,
                    },
                    "offset": {
                        "type": "integer",
                        "description": "Pagination offset",
                        "default": 0,
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="memory_tags",
            description="List all tags in use with their counts.",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
        Tool(
            name="memory_stats",
            description="Get storage statistics.",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
        Tool(
            name="memory_embed",
            description="Force (re)generation of embeddings for one or all memories.",
            inputSchema={
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "description": "Specific key, or all if omitted",
                    },
                    "force": {
                        "type": "boolean",
                        "description": "Regenerate even if exists",
                        "default": False,
                    },
                },
            },
        ),
        Tool(
            name="memory_history",
            description="Get version history for a key.",
            inputSchema={
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "description": "The key to get history for",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max versions (default 10)",
                        "default": 10,
                    },
                },
                "required": ["key"],
            },
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle tool calls."""
    try:
        if name == "memory_set":
            result = await handle_memory_set(arguments)
        elif name == "memory_set_from_file":
            result = await handle_memory_set_from_file(arguments)
        elif name == "memory_get":
            result = await handle_memory_get(arguments)
        elif name == "memory_delete":
            result = await handle_memory_delete(arguments)
        elif name == "memory_list":
            result = await handle_memory_list(arguments)
        elif name == "memory_search":
            result = await handle_memory_search(arguments)
        elif name == "memory_fulltext":
            result = await handle_memory_fulltext(arguments)
        elif name == "memory_tags":
            result = await handle_memory_tags()
        elif name == "memory_stats":
            result = await handle_memory_stats()
        elif name == "memory_embed":
            result = await handle_memory_embed(arguments)
        elif name == "memory_history":
            result = await handle_memory_history(arguments)
        else:
            result = ErrorResponse(error=f"Unknown tool: {name}")

        return [TextContent(type="text", text=result.model_dump_json(indent=2))]

    except Exception as e:
        error = ErrorResponse(error=str(e))
        return [TextContent(type="text", text=error.model_dump_json(indent=2))]


async def handle_memory_set(args: dict) -> SetResponse:
    """Handle memory_set tool."""
    key = args["key"]
    value = args["value"]
    content_type = args.get("content_type", "text/plain")
    tags = args.get("tags", [])
    should_embed = args.get("embed", True) and auto_embed_enabled()

    embedding = None
    embedding_model = None
    warnings = []

    if should_embed:
        embedding = emb.generate_embedding(value)
        if embedding is None:
            warnings.append("Failed to generate embedding")
        else:
            embedding_model = emb.get_model_name()

    created, changed, previous_value, previous_size_bytes, db_warnings = db.memory_set(
        key=key,
        value=value,
        content_type=content_type,
        tags=tags,
        embedding=embedding,
        embedding_model=embedding_model,
    )

    warnings.extend(db_warnings)

    # Calculate current size
    size_bytes = len(value.encode("utf-8"))

    # Warn if content significantly reduced (potential accidental truncation)
    if changed and previous_size_bytes is not None and size_bytes < previous_size_bytes * 0.5:
        warnings.append(
            f"Content reduced by >50% (was {previous_size_bytes} bytes, now {size_bytes} bytes). "
            "Accidental truncation?"
        )

    return SetResponse(
        success=True,
        created=created,
        changed=changed,
        key=key.lower(),
        size_bytes=size_bytes,
        previous_value=previous_value if changed else None,
        previous_size_bytes=previous_size_bytes,
        warnings=warnings,
    )


def _detect_content_type(file_path: Path) -> str:
    """Detect content type from file extension."""
    ext_map = {
        ".md": "text/markdown",
        ".json": "application/json",
        ".yaml": "text/yaml",
        ".yml": "text/yaml",
        ".xml": "application/xml",
        ".html": "text/html",
        ".css": "text/css",
        ".js": "text/javascript",
        ".ts": "text/typescript",
        ".py": "text/x-python",
        ".sh": "text/x-shellscript",
        ".sql": "text/x-sql",
        ".toml": "text/x-toml",
        ".ini": "text/x-ini",
        ".cfg": "text/x-ini",
        ".txt": "text/plain",
    }
    return ext_map.get(file_path.suffix.lower(), "text/plain")


async def handle_memory_set_from_file(
    args: dict,
) -> SetFromFileResponse | ErrorResponse:
    """Handle memory_set_from_file tool."""
    key = args["key"]
    file_path_str = args["file_path"]
    content_type_override = args.get("content_type")
    tags = args.get("tags", [])
    should_embed = args.get("embed", True) and auto_embed_enabled()

    # Expand ~ and resolve path
    file_path = Path(file_path_str).expanduser().resolve()

    # Validate file exists
    if not file_path.exists():
        return ErrorResponse(error=f"File not found: {file_path}")

    if not file_path.is_file():
        return ErrorResponse(error=f"Not a file: {file_path}")

    # Read file
    try:
        content = file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return ErrorResponse(error=f"Cannot read binary file: {file_path}")
    except PermissionError:
        return ErrorResponse(error=f"Permission denied: {file_path}")
    except Exception as e:
        return ErrorResponse(error=f"Failed to read file: {e}")

    warnings: list[str] = []
    file_size = file_path.stat().st_size

    # Auto-detect content type from extension
    content_type = content_type_override or _detect_content_type(file_path)

    # Generate embedding
    embedding = None
    embedding_model = None
    if should_embed:
        embedding = emb.generate_embedding(content)
        if embedding is None:
            warnings.append("Failed to generate embedding")
        else:
            embedding_model = emb.get_model_name()

    # Store via existing memory_set
    created, changed, previous_value, previous_size_bytes, db_warnings = db.memory_set(
        key=key,
        value=content,
        content_type=content_type,
        tags=tags,
        embedding=embedding,
        embedding_model=embedding_model,
    )
    warnings.extend(db_warnings)

    return SetFromFileResponse(
        success=True,
        created=created,
        changed=changed,
        key=key.lower(),
        file_path=str(file_path),
        file_size_bytes=file_size,
        content_type=content_type,
        previous_value=previous_value if changed else None,
        previous_size_bytes=previous_size_bytes,
        warnings=warnings,
    )


async def handle_memory_get(args: dict) -> GetResponse | ErrorResponse:
    """Handle memory_get tool."""
    key = args["key"]

    record, warnings = db.memory_get(key)

    if record is None:
        return ErrorResponse(error=f"Memory not found: {key}")

    return GetResponse(
        key=record.key,
        value=record.value,
        content_type=record.content_type,
        tags=record.tags,
        created_at=record.created_at.isoformat(),
        updated_at=record.updated_at.isoformat(),
        accessed_at=record.accessed_at.isoformat(),
        access_count=record.access_count,
        warnings=warnings,
    )


async def handle_memory_delete(args: dict) -> DeleteResponse:
    """Handle memory_delete tool."""
    key = args["key"]

    deleted, warnings = db.memory_delete(key)

    return DeleteResponse(
        success=True,
        deleted=deleted,
        warnings=warnings,
    )


async def handle_memory_list(args: dict) -> ListResponse:
    """Handle memory_list tool."""
    pattern = args.get("pattern")
    tag = args.get("tag")
    limit = args.get("limit", 50)
    offset = args.get("offset", 0)

    items, total, warnings = db.memory_list(
        pattern=pattern,
        tag=tag,
        limit=limit,
        offset=offset,
    )

    return ListResponse(
        memories=items,
        total=total,
        warnings=warnings,
    )


async def handle_memory_search(args: dict) -> SearchResponse:
    """Handle memory_search tool."""
    query = args["query"]
    limit = args.get("limit", 10)
    threshold = args.get("threshold", 0.3)
    warnings = []

    # Generate query embedding
    query_embedding = emb.generate_embedding(query)
    if query_embedding is None:
        return SearchResponse(
            results=[],
            warnings=["Failed to generate query embedding"],
        )

    # Get all embeddings
    all_embeddings = db.get_all_embeddings()

    if not all_embeddings:
        return SearchResponse(
            results=[],
            warnings=["No embedded memories found"],
        )

    # Search
    matches = emb.search_embeddings(
        query_embedding=query_embedding,
        embeddings=all_embeddings,
        limit=limit,
        threshold=threshold,
    )

    # Build results
    results = []
    for key, score in matches:
        data = db.get_memory_with_tags(key)
        if data:
            value, content_type, tags = data
            results.append(
                {
                    "key": key,
                    "value": value,
                    "content_type": content_type,
                    "score": round(score, 4),
                    "tags": tags,
                }
            )

    return SearchResponse(
        results=[
            type("MemorySearchResult", (), r)
            for r in results
        ]
        if False
        else SearchResponse(results=results, warnings=warnings),
    ).model_copy() if False else SearchResponse(
        results=results,
        warnings=warnings,
    )


async def handle_memory_fulltext(args: dict) -> FulltextResponse:
    """Handle memory_fulltext tool."""
    query = args["query"]
    search_keys = args.get("search_keys", True)
    search_values = args.get("search_values", True)
    limit = args.get("limit", 20)
    offset = args.get("offset", 0)

    results, total, warnings = db.memory_fulltext(
        query=query,
        search_keys=search_keys,
        search_values=search_values,
        limit=limit,
        offset=offset,
    )

    return FulltextResponse(
        results=results,
        total=total,
        warnings=warnings,
    )


async def handle_memory_tags() -> TagsResponse:
    """Handle memory_tags tool."""
    tags = db.get_all_tags()
    return TagsResponse(tags=tags, warnings=[])


async def handle_memory_stats() -> StatsResponse:
    """Handle memory_stats tool."""
    stats = db.get_stats()
    return StatsResponse(
        total_memories=stats["total_memories"],
        total_tags=stats["total_tags"],
        total_size_bytes=stats["total_size_bytes"],
        embedded_count=stats["embedded_count"],
        oldest=stats["oldest"],
        newest=stats["newest"],
        warnings=[],
    )


async def handle_memory_embed(args: dict) -> EmbedResponse:
    """Handle memory_embed tool."""
    key = args.get("key")
    force = args.get("force", False)
    warnings = []

    model_name = emb.get_model_name()

    if key:
        # Single key
        value = db.get_memory_value(key)
        if value is None:
            return EmbedResponse(
                embedded_count=0,
                failed_count=0,
                warnings=[f"Memory not found: {key}"],
            )

        embedding = emb.generate_embedding(value)
        if embedding is None:
            return EmbedResponse(
                embedded_count=0,
                failed_count=1,
                warnings=[f"Failed to embed: {key}"],
            )

        db.update_embedding(key, embedding, model_name)
        return EmbedResponse(embedded_count=1, failed_count=0, warnings=[])

    # All keys
    if force:
        keys = db.get_all_keys()
    else:
        keys = db.get_keys_without_embedding()

    embedded = 0
    failed = 0

    for k in keys:
        value = db.get_memory_value(k)
        if value is None:
            continue

        embedding = emb.generate_embedding(value)
        if embedding is None:
            failed += 1
            warnings.append(f"Failed to embed: {k}")
        else:
            db.update_embedding(k, embedding, model_name)
            embedded += 1

    return EmbedResponse(
        embedded_count=embedded,
        failed_count=failed,
        warnings=warnings,
    )


async def handle_memory_history(args: dict) -> HistoryResponse:
    """Handle memory_history tool."""
    key = args["key"]
    limit = args.get("limit", 10)

    entries, truncated, warnings = db.get_history(key, limit)

    return HistoryResponse(
        history=entries,
        truncated=truncated,
        warnings=warnings,
    )


def main():
    """Run the MCP server."""
    import asyncio

    db.init_db()

    async def run():
        async with stdio_server() as (read_stream, write_stream):
            await app.run(read_stream, write_stream, app.create_initialization_options())

    asyncio.run(run())


if __name__ == "__main__":
    main()
