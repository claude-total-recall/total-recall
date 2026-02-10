"""MCP server for Total Recall."""

import os
from pathlib import Path

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from . import crypto
from . import db
from . import embeddings as emb
from . import net
from . import serialization
from .models import (
    BackupResponse,
    DeleteResponse,
    EmbedResponse,
    ErrorResponse,
    FulltextResponse,
    GetResponse,
    HistoryResponse,
    ListResponse,
    RestoreResponse,
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
        Tool(
            name="memory_backup",
            description="""Create an encrypted backup of memories.

Exports selected memories to a password-protected .trbak file.
Defaults to ~/.total_recall/backups/ with a timestamped filename.
Use 'keys' glob to select which memories to include.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "password": {
                        "type": "string",
                        "description": "Encryption password",
                    },
                    "path": {
                        "type": "string",
                        "description": "Output file path (default: auto-generated in ~/.total_recall/backups/)",
                    },
                    "keys": {
                        "type": "string",
                        "description": "Glob filter for which memories to include (default: '*' = all)",
                        "default": "*",
                    },
                    "include_history": {
                        "type": "boolean",
                        "description": "Include version history",
                        "default": True,
                    },
                    "include_embeddings": {
                        "type": "boolean",
                        "description": "Include embedding vectors (large, regenerable)",
                        "default": False,
                    },
                },
                "required": ["password"],
            },
        ),
        Tool(
            name="memory_restore",
            description="""Restore memories from an encrypted .trbak backup file.

Decrypts and imports memories with configurable merge strategy.
Memories are auto-embedded on import if they lack embeddings.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "password": {
                        "type": "string",
                        "description": "Decryption password",
                    },
                    "path": {
                        "type": "string",
                        "description": "Path to .trbak backup file",
                    },
                    "key_filter": {
                        "type": "string",
                        "description": "Only restore memories matching this glob (default: '*')",
                        "default": "*",
                    },
                    "merge": {
                        "type": "string",
                        "description": "Merge strategy: 'newer_wins', 'skip_existing', or 'overwrite'",
                        "default": "newer_wins",
                        "enum": ["newer_wins", "skip_existing", "overwrite"],
                    },
                },
                "required": ["password", "path"],
            },
        ),
        *net.net_tools(),
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
        elif name == "memory_backup":
            result = await handle_memory_backup(arguments)
        elif name == "memory_restore":
            result = await handle_memory_restore(arguments)
        elif name == "memory_listen":
            result = await net.handle_memory_listen(arguments)
        elif name == "memory_send":
            result = await net.handle_memory_send(arguments)
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


async def handle_memory_backup(args: dict) -> BackupResponse | ErrorResponse:
    """Handle memory_backup tool."""
    import struct

    password = args["password"]
    keys_pattern = args.get("keys", "*")
    include_history = args.get("include_history", True)
    include_embeddings = args.get("include_embeddings", False)

    # Default path
    if "path" in args and args["path"]:
        backup_path = Path(args["path"]).expanduser().resolve()
    else:
        from datetime import datetime as dt

        backup_dir = Path.home() / ".total_recall" / "backups"
        timestamp = dt.now().strftime("%Y-%m-%dT%H%M")
        backup_path = backup_dir / f"backup_{timestamp}.trbak"

    # Serialize
    payload = serialization.serialize_memories(
        keys=[keys_pattern],
        include_history=include_history,
        include_embeddings=include_embeddings,
    )

    if not payload["memories"]:
        return ErrorResponse(error="No memories matched the key pattern")

    # Encrypt
    salt, ciphertext = crypto.encrypt_payload(payload, password)

    # Write file
    backup_path.parent.mkdir(parents=True, exist_ok=True)

    with open(backup_path, "wb") as f:
        f.write(b"TRBK")                              # 4 bytes magic
        f.write(struct.pack(">H", 1))                  # 2 bytes version
        f.write(salt)                                   # 16 bytes salt
        f.write(struct.pack(">I", len(ciphertext)))     # 4 bytes payload length
        f.write(ciphertext)                             # N bytes ciphertext

    file_size = backup_path.stat().st_size

    return BackupResponse(
        success=True,
        path=str(backup_path),
        memory_count=len(payload["memories"]),
        size_bytes=file_size,
    )


async def handle_memory_restore(args: dict) -> RestoreResponse | ErrorResponse:
    """Handle memory_restore tool."""
    import struct

    from cryptography.fernet import InvalidToken

    password = args["password"]
    path = Path(args["path"]).expanduser().resolve()
    key_filter = args.get("key_filter", "*")
    merge = args.get("merge", "newer_wins")

    if not path.exists():
        return ErrorResponse(error=f"File not found: {path}")

    with open(path, "rb") as f:
        # Read header
        magic = f.read(4)
        if magic != b"TRBK":
            return ErrorResponse(error=f"Not a valid .trbak file (bad magic: {magic!r})")

        version = struct.unpack(">H", f.read(2))[0]
        if version != 1:
            return ErrorResponse(error=f"Unsupported backup version: {version}")

        salt = f.read(16)
        payload_len = struct.unpack(">I", f.read(4))[0]
        ciphertext = f.read(payload_len)

    # Decrypt
    try:
        payload = crypto.decrypt_payload(ciphertext, password, salt)
    except InvalidToken:
        return ErrorResponse(error="Decryption failed — wrong password")

    # Deserialize and import
    result = serialization.deserialize_memories(
        payload, key_filter=key_filter, merge=merge, auto_embed=True
    )

    return RestoreResponse(
        success=True,
        restored=result.restored,
        skipped=result.skipped,
        conflicts=result.conflicts,
        warnings=result.warnings,
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
