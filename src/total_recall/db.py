"""SQLite database operations for Total Recall."""

import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .models import (
    HistoryEntry,
    MemoryFulltextResult,
    MemoryListItem,
    MemoryRecord,
    TagCount,
)

# Size warning threshold (100KB)
SIZE_WARNING_THRESHOLD = 100 * 1024


def get_db_path() -> Path:
    """Get the database path from config or default."""
    db_path = os.environ.get("TOTAL_RECALL_DB")
    if db_path:
        return Path(db_path)
    return Path.home() / ".total_recall" / "memory.db"


def get_max_history() -> int:
    """Get max history versions from config."""
    return int(os.environ.get("TOTAL_RECALL_MAX_HISTORY", "50"))


def history_enabled() -> bool:
    """Check if history tracking is enabled."""
    return os.environ.get("TOTAL_RECALL_HISTORY", "true").lower() == "true"


@contextmanager
def get_connection():
    """Get a database connection with proper settings."""
    db_path = get_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")

    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    """Initialize the database schema."""
    with get_connection() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS memories (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                content_type TEXT DEFAULT 'text/plain',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                accessed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                access_count INTEGER DEFAULT 0,
                embedding BLOB,
                embedding_model TEXT,
                metadata JSON
            );

            CREATE TABLE IF NOT EXISTS tags (
                key TEXT NOT NULL,
                tag TEXT NOT NULL,
                PRIMARY KEY (key, tag),
                FOREIGN KEY (key) REFERENCES memories(key) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                content_type TEXT,
                changed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (key) REFERENCES memories(key) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_tags_tag ON tags(tag);
            CREATE INDEX IF NOT EXISTS idx_memories_updated ON memories(updated_at);
            CREATE INDEX IF NOT EXISTS idx_memories_accessed ON memories(accessed_at);
        """)


def serialize_embedding(embedding: Optional[list[float]]) -> Optional[bytes]:
    """Serialize embedding to bytes for storage."""
    if embedding is None:
        return None
    import struct

    return struct.pack(f"{len(embedding)}f", *embedding)


def _deserialize_embedding(data: Optional[bytes]) -> Optional[list[float]]:
    """Deserialize embedding from bytes."""
    if data is None:
        return None
    import struct

    count = len(data) // 4
    return list(struct.unpack(f"{count}f", data))


def normalize_key(key: str) -> str:
    """Normalize key to lowercase."""
    return key.lower()


def _normalize_tag(tag: str) -> str:
    """Normalize tag: lowercase and strip whitespace."""
    return tag.strip().lower()


def memory_exists(key: str) -> bool:
    """Check if a memory exists."""
    key = normalize_key(key)
    with get_connection() as conn:
        row = conn.execute(
            "SELECT 1 FROM memories WHERE key = ?", (key,)
        ).fetchone()
        return row is not None


def memory_set(
    key: str,
    value: str,
    content_type: str = "text/plain",
    tags: Optional[list[str]] = None,
    embedding: Optional[list[float]] = None,
    embedding_model: Optional[str] = None,
) -> tuple[bool, bool, Optional[str], Optional[int], list[str]]:
    """Create or update a memory.

    Returns:
        Tuple of (created, changed, previous_value, previous_size_bytes, warnings)
    """
    key = normalize_key(key)
    tags = [_normalize_tag(t) for t in (tags or []) if t.strip()]
    tags = list(set(tags))  # deduplicate
    warnings = []

    # Check size warning
    if len(value.encode("utf-8")) > SIZE_WARNING_THRESHOLD:
        warnings.append(f"Value exceeds {SIZE_WARNING_THRESHOLD // 1024}KB")

    with get_connection() as conn:
        now = datetime.now(timezone.utc).isoformat()

        # Check existing
        existing = conn.execute(
            "SELECT value, content_type FROM memories WHERE key = ?", (key,)
        ).fetchone()

        if existing is None:
            # Create new
            conn.execute(
                """
                INSERT INTO memories (key, value, content_type, created_at, updated_at,
                                     accessed_at, access_count, embedding, embedding_model)
                VALUES (?, ?, ?, ?, ?, ?, 0, ?, ?)
                """,
                (
                    key,
                    value,
                    content_type,
                    now,
                    now,
                    now,
                    serialize_embedding(embedding),
                    embedding_model,
                ),
            )

            # Add tags
            for tag in tags:
                conn.execute(
                    "INSERT OR IGNORE INTO tags (key, tag) VALUES (?, ?)",
                    (key, tag),
                )

            return True, True, None, None, warnings

        # Update existing
        old_value = existing["value"]
        old_size = len(old_value.encode("utf-8"))

        if old_value == value:
            # No change to value - only update tags if needed
            # First clear existing tags
            conn.execute("DELETE FROM tags WHERE key = ?", (key,))
            for tag in tags:
                conn.execute(
                    "INSERT OR IGNORE INTO tags (key, tag) VALUES (?, ?)",
                    (key, tag),
                )

            # Update content_type and embedding if provided
            if embedding is not None:
                conn.execute(
                    """
                    UPDATE memories SET content_type = ?, embedding = ?, embedding_model = ?
                    WHERE key = ?
                    """,
                    (content_type, serialize_embedding(embedding), embedding_model, key),
                )
            else:
                conn.execute(
                    "UPDATE memories SET content_type = ? WHERE key = ?",
                    (content_type, key),
                )

            # No change - return previous value info but changed=False
            return False, False, old_value, old_size, warnings

        # Value changed - record history if enabled
        if history_enabled():
            conn.execute(
                """
                INSERT INTO history (key, value, content_type, changed_at)
                VALUES (?, ?, ?, ?)
                """,
                (key, old_value, existing["content_type"], now),
            )

            # Prune old history
            max_history = get_max_history()
            count = conn.execute(
                "SELECT COUNT(*) FROM history WHERE key = ?", (key,)
            ).fetchone()[0]

            if count > max_history:
                pruned = count - max_history
                conn.execute(
                    """
                    DELETE FROM history WHERE key = ? AND id IN (
                        SELECT id FROM history WHERE key = ? ORDER BY changed_at ASC LIMIT ?
                    )
                    """,
                    (key, key, pruned),
                )
                warnings.append(f"Pruned {pruned} old history version(s)")

        # Update memory
        conn.execute(
            """
            UPDATE memories SET value = ?, content_type = ?, updated_at = ?,
                               embedding = ?, embedding_model = ?
            WHERE key = ?
            """,
            (
                value,
                content_type,
                now,
                serialize_embedding(embedding),
                embedding_model,
                key,
            ),
        )

        # Update tags
        conn.execute("DELETE FROM tags WHERE key = ?", (key,))
        for tag in tags:
            conn.execute(
                "INSERT OR IGNORE INTO tags (key, tag) VALUES (?, ?)",
                (key, tag),
            )

        return False, True, old_value, old_size, warnings


def memory_get(key: str) -> tuple[Optional[MemoryRecord], list[str]]:
    """Get a memory by key, updating access tracking.

    Returns:
        Tuple of (record or None, warnings)
    """
    key = normalize_key(key)
    warnings = []

    with get_connection() as conn:
        now = datetime.now(timezone.utc).isoformat()

        # Update access tracking
        conn.execute(
            """
            UPDATE memories SET accessed_at = ?, access_count = access_count + 1
            WHERE key = ?
            """,
            (now, key),
        )

        row = conn.execute(
            """
            SELECT key, value, content_type, created_at, updated_at, accessed_at,
                   access_count, embedding, embedding_model
            FROM memories WHERE key = ?
            """,
            (key,),
        ).fetchone()

        if row is None:
            return None, warnings

        # Get tags
        tag_rows = conn.execute(
            "SELECT tag FROM tags WHERE key = ?", (key,)
        ).fetchall()
        tags = [r["tag"] for r in tag_rows]

        return MemoryRecord(
            key=row["key"],
            value=row["value"],
            content_type=row["content_type"],
            tags=tags,
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            accessed_at=datetime.fromisoformat(row["accessed_at"]),
            access_count=row["access_count"],
            embedding=_deserialize_embedding(row["embedding"]),
            embedding_model=row["embedding_model"],
        ), warnings


def memory_delete(key: str) -> tuple[bool, list[str]]:
    """Delete a memory.

    Returns:
        Tuple of (deleted, warnings)
    """
    key = normalize_key(key)
    warnings = []

    with get_connection() as conn:
        cursor = conn.execute("DELETE FROM memories WHERE key = ?", (key,))
        return cursor.rowcount > 0, warnings


def memory_list(
    pattern: Optional[str] = None,
    tag: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
) -> tuple[list[MemoryListItem], int, list[str]]:
    """List memories with optional filtering.

    Returns:
        Tuple of (items, total_count, warnings)
    """
    warnings = []

    with get_connection() as conn:
        # Build query
        where_clauses = []
        params = []

        if pattern:
            # Convert glob to SQL LIKE
            sql_pattern = pattern.replace("*", "%")
            where_clauses.append("key LIKE ?")
            params.append(sql_pattern)

        if tag:
            tag = _normalize_tag(tag)
            where_clauses.append("key IN (SELECT key FROM tags WHERE tag = ?)")
            params.append(tag)

        where_sql = " AND ".join(where_clauses) if where_clauses else "1=1"

        # Get total count
        total = conn.execute(
            f"SELECT COUNT(*) FROM memories WHERE {where_sql}", params
        ).fetchone()[0]

        # Get items
        rows = conn.execute(
            f"""
            SELECT key, content_type, updated_at FROM memories
            WHERE {where_sql}
            ORDER BY updated_at DESC
            LIMIT ? OFFSET ?
            """,
            params + [limit, offset],
        ).fetchall()

        items = []
        for row in rows:
            # Get tags for each memory
            tag_rows = conn.execute(
                "SELECT tag FROM tags WHERE key = ?", (row["key"],)
            ).fetchall()
            tags = [r["tag"] for r in tag_rows]

            items.append(
                MemoryListItem(
                    key=row["key"],
                    content_type=row["content_type"],
                    tags=tags,
                    updated_at=datetime.fromisoformat(row["updated_at"]),
                )
            )

        return items, total, warnings


def memory_fulltext(
    query: str,
    search_keys: bool = True,
    search_values: bool = True,
    limit: int = 20,
    offset: int = 0,
) -> tuple[list[MemoryFulltextResult], int, list[str]]:
    """Full-text search in memories.

    Returns:
        Tuple of (results, total_count, warnings)
    """
    warnings = []

    with get_connection() as conn:
        where_clauses = []

        if search_keys:
            where_clauses.append("key LIKE ?")
        if search_values:
            where_clauses.append("value LIKE ?")

        if not where_clauses:
            return [], 0, warnings

        where_sql = " OR ".join(where_clauses)
        search_pattern = f"%{query}%"
        params = [search_pattern] * len(where_clauses)

        # Get total
        total = conn.execute(
            f"SELECT COUNT(*) FROM memories WHERE {where_sql}", params
        ).fetchone()[0]

        # Get results
        rows = conn.execute(
            f"""
            SELECT key, value, content_type FROM memories
            WHERE {where_sql}
            ORDER BY updated_at DESC
            LIMIT ? OFFSET ?
            """,
            params + [limit, offset],
        ).fetchall()

        results = []
        for row in rows:
            # Create snippet around match
            value = row["value"]
            pos = value.lower().find(query.lower())
            if pos >= 0:
                start = max(0, pos - 50)
                end = min(len(value), pos + len(query) + 50)
                snippet = value[start:end]
                if start > 0:
                    snippet = "..." + snippet
                if end < len(value):
                    snippet = snippet + "..."
            else:
                snippet = value[:100] + ("..." if len(value) > 100 else "")

            # Get tags
            tag_rows = conn.execute(
                "SELECT tag FROM tags WHERE key = ?", (row["key"],)
            ).fetchall()
            tags = [r["tag"] for r in tag_rows]

            results.append(
                MemoryFulltextResult(
                    key=row["key"],
                    value=row["value"],
                    snippet=snippet,
                    content_type=row["content_type"],
                    tags=tags,
                )
            )

        return results, total, warnings


def get_all_embeddings() -> list[tuple[str, list[float]]]:
    """Get all memory keys with their embeddings."""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT key, embedding FROM memories WHERE embedding IS NOT NULL"
        ).fetchall()

        return [
            (row["key"], _deserialize_embedding(row["embedding"]))
            for row in rows
            if row["embedding"] is not None
        ]


def get_memory_value(key: str) -> Optional[str]:
    """Get just the value for a memory (without access tracking)."""
    key = normalize_key(key)
    with get_connection() as conn:
        row = conn.execute(
            "SELECT value FROM memories WHERE key = ?", (key,)
        ).fetchone()
        return row["value"] if row else None


def get_memory_with_tags(key: str) -> Optional[tuple[str, str, list[str]]]:
    """Get value, content_type, and tags for a memory (without access tracking)."""
    key = normalize_key(key)
    with get_connection() as conn:
        row = conn.execute(
            "SELECT value, content_type FROM memories WHERE key = ?", (key,)
        ).fetchone()

        if row is None:
            return None

        tag_rows = conn.execute(
            "SELECT tag FROM tags WHERE key = ?", (key,)
        ).fetchall()
        tags = [r["tag"] for r in tag_rows]

        return row["value"], row["content_type"], tags


def get_all_tags() -> list[TagCount]:
    """Get all tags with their usage counts."""
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT tag, COUNT(*) as count FROM tags
            GROUP BY tag ORDER BY count DESC
            """
        ).fetchall()

        return [TagCount(tag=row["tag"], count=row["count"]) for row in rows]


def get_stats() -> dict:
    """Get storage statistics."""
    with get_connection() as conn:
        total = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        total_tags = conn.execute("SELECT COUNT(DISTINCT tag) FROM tags").fetchone()[0]

        # Calculate total size
        size_row = conn.execute(
            "SELECT SUM(LENGTH(value)) FROM memories"
        ).fetchone()
        total_size = size_row[0] or 0

        embedded = conn.execute(
            "SELECT COUNT(*) FROM memories WHERE embedding IS NOT NULL"
        ).fetchone()[0]

        oldest_row = conn.execute(
            "SELECT MIN(created_at) FROM memories"
        ).fetchone()
        newest_row = conn.execute(
            "SELECT MAX(created_at) FROM memories"
        ).fetchone()

        return {
            "total_memories": total,
            "total_tags": total_tags,
            "total_size_bytes": total_size,
            "embedded_count": embedded,
            "oldest": oldest_row[0],
            "newest": newest_row[0],
        }


def get_history(key: str, limit: int = 10) -> tuple[list[HistoryEntry], bool, list[str]]:
    """Get version history for a key.

    Returns:
        Tuple of (history_entries, truncated, warnings)
    """
    key = normalize_key(key)
    warnings = []

    with get_connection() as conn:
        # Check total count
        total = conn.execute(
            "SELECT COUNT(*) FROM history WHERE key = ?", (key,)
        ).fetchone()[0]

        rows = conn.execute(
            """
            SELECT value, content_type, changed_at FROM history
            WHERE key = ?
            ORDER BY changed_at DESC
            LIMIT ?
            """,
            (key, limit),
        ).fetchall()

        entries = [
            HistoryEntry(
                value=row["value"],
                content_type=row["content_type"],
                changed_at=datetime.fromisoformat(row["changed_at"]),
            )
            for row in rows
        ]

        return entries, total > limit, warnings


def get_keys_without_embedding() -> list[str]:
    """Get keys that don't have embeddings."""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT key FROM memories WHERE embedding IS NULL"
        ).fetchall()
        return [row["key"] for row in rows]


def get_all_keys() -> list[str]:
    """Get all memory keys."""
    with get_connection() as conn:
        rows = conn.execute("SELECT key FROM memories").fetchall()
        return [row["key"] for row in rows]


def update_embedding(key: str, embedding: list[float], model: str):
    """Update the embedding for a specific key."""
    key = normalize_key(key)
    with get_connection() as conn:
        conn.execute(
            "UPDATE memories SET embedding = ?, embedding_model = ? WHERE key = ?",
            (serialize_embedding(embedding), model, key),
        )


# --- Export / Import functions for backup & network sharing ---


def resolve_key_patterns(patterns: list[str]) -> list[str]:
    """Expand glob patterns to matching keys via SQL LIKE.

    Patterns use '*' as wildcard (converted to SQL '%'). Deduplicates results.
    """
    with get_connection() as conn:
        keys: set[str] = set()
        for pattern in patterns:
            sql_pattern = pattern.replace("*", "%")
            rows = conn.execute(
                "SELECT key FROM memories WHERE key LIKE ?", (sql_pattern,)
            ).fetchall()
            keys.update(row["key"] for row in rows)
        return sorted(keys)


def get_memories_for_export(
    keys: list[str],
    include_history: bool = True,
    include_embeddings: bool = False,
) -> list[dict]:
    """Bulk read memories for export with tags, optional history and embeddings."""
    import base64

    with get_connection() as conn:
        results = []
        for key in keys:
            row = conn.execute(
                """SELECT key, value, content_type, created_at, updated_at,
                          access_count, embedding, embedding_model
                   FROM memories WHERE key = ?""",
                (key,),
            ).fetchone()
            if row is None:
                continue

            entry: dict = {
                "key": row["key"],
                "value": row["value"],
                "content_type": row["content_type"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
                "access_count": row["access_count"],
            }

            # Tags
            tag_rows = conn.execute(
                "SELECT tag FROM tags WHERE key = ?", (key,)
            ).fetchall()
            entry["tags"] = [r["tag"] for r in tag_rows]

            # History
            if include_history:
                hist_rows = conn.execute(
                    """SELECT value, content_type, changed_at FROM history
                       WHERE key = ? ORDER BY changed_at ASC""",
                    (key,),
                ).fetchall()
                entry["history"] = [
                    {
                        "value": h["value"],
                        "content_type": h["content_type"],
                        "changed_at": h["changed_at"],
                    }
                    for h in hist_rows
                ]

            # Embeddings
            if include_embeddings and row["embedding"] is not None:
                entry["embedding"] = base64.b64encode(row["embedding"]).decode()
                entry["embedding_model"] = row["embedding_model"]

            results.append(entry)
        return results


def import_memory(
    conn,
    key: str,
    value: str,
    content_type: str,
    tags: list[str],
    created_at: str,
    updated_at: str,
    history: Optional[list[dict]] = None,
    embedding_bytes: Optional[bytes] = None,
    embedding_model: Optional[str] = None,
):
    """Import a single memory using an existing connection (for atomicity).

    Deletes existing key first (cascade cleans tags + history), inserts fresh
    with access_count=0, accessed_at=now. Preserves source created_at/updated_at.
    """
    now = datetime.now(timezone.utc).isoformat()

    # Delete existing (cascade handles tags + history)
    conn.execute("DELETE FROM memories WHERE key = ?", (key,))

    conn.execute(
        """INSERT INTO memories (key, value, content_type, created_at, updated_at,
                                accessed_at, access_count, embedding, embedding_model)
           VALUES (?, ?, ?, ?, ?, ?, 0, ?, ?)""",
        (key, value, content_type, created_at, updated_at, now,
         embedding_bytes, embedding_model),
    )

    for tag in tags:
        conn.execute(
            "INSERT OR IGNORE INTO tags (key, tag) VALUES (?, ?)", (key, tag)
        )

    if history:
        for h in history:
            conn.execute(
                """INSERT INTO history (key, value, content_type, changed_at)
                   VALUES (?, ?, ?, ?)""",
                (key, h["value"], h.get("content_type"), h["changed_at"]),
            )


def get_memory_updated_at(key: str) -> Optional[str]:
    """Lightweight timestamp lookup for merge strategy. No access tracking."""
    key = normalize_key(key)
    with get_connection() as conn:
        row = conn.execute(
            "SELECT updated_at FROM memories WHERE key = ?", (key,)
        ).fetchone()
        return row["updated_at"] if row else None
