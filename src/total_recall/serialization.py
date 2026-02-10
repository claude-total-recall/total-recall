"""Memory serialization and deserialization for backup and network sharing."""

import base64
import fnmatch
import platform
from datetime import datetime, timezone

from . import db
from . import embeddings as emb
from .models import ImportResult


def serialize_memories(
    keys: list[str],
    include_history: bool = True,
    include_embeddings: bool = False,
) -> dict:
    """Build the payload dict from selected memories.

    Args:
        keys: Glob patterns resolved against DB before serializing.
        include_history: Include version history per memory.
        include_embeddings: Include base64-encoded embedding blobs.

    Returns:
        Envelope dict ready for encryption.
    """
    resolved = db.resolve_key_patterns(keys)
    memories = db.get_memories_for_export(
        resolved,
        include_history=include_history,
        include_embeddings=include_embeddings,
    )
    return {
        "version": 1,
        "source": platform.node(),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "memories": memories,
    }


def deserialize_memories(
    payload: dict,
    key_filter: str = "*",
    merge: str = "newer_wins",
    auto_embed: bool = True,
) -> ImportResult:
    """Apply key filter, merge strategy, store into DB, auto-embed.

    All writes happen in a single DB transaction for atomicity.

    Args:
        payload: Decrypted payload dict with "memories" list.
        key_filter: fnmatch glob — only accept memories matching this.
        merge: "skip_existing" | "overwrite" | "newer_wins"
        auto_embed: Generate embeddings for memories arriving without them.

    Returns:
        ImportResult with counts.
    """
    result = ImportResult()
    memories = payload.get("memories", [])

    with db.get_connection() as conn:
        for mem in memories:
            key = db.normalize_key(mem["key"])

            # Key filter
            if not fnmatch.fnmatch(key, key_filter):
                result.filtered += 1
                continue

            # Merge check
            existing_updated_at = conn.execute(
                "SELECT updated_at FROM memories WHERE key = ?", (key,)
            ).fetchone()

            if existing_updated_at is not None:
                existing_ts = existing_updated_at["updated_at"]
                incoming_ts = mem.get("updated_at", "")

                if merge == "skip_existing":
                    result.skipped += 1
                    continue
                elif merge == "newer_wins":
                    if incoming_ts <= existing_ts:
                        result.skipped += 1
                        result.conflicts += 1
                        continue

            # Embedding handling
            embedding_bytes = None
            embedding_model = None

            if "embedding" in mem and mem["embedding"]:
                embedding_bytes = base64.b64decode(mem["embedding"])
                embedding_model = mem.get("embedding_model")
            elif auto_embed:
                emb_list = emb.generate_embedding(mem["value"])
                if emb_list is not None:
                    embedding_bytes = db.serialize_embedding(emb_list)
                    embedding_model = emb.get_model_name()
                    result.embedded += 1
                else:
                    result.embed_failed += 1

            # Import
            db.import_memory(
                conn,
                key=key,
                value=mem["value"],
                content_type=mem.get("content_type", "text/plain"),
                tags=mem.get("tags", []),
                created_at=mem.get("created_at", datetime.now(timezone.utc).isoformat()),
                updated_at=mem.get("updated_at", datetime.now(timezone.utc).isoformat()),
                history=mem.get("history"),
                embedding_bytes=embedding_bytes,
                embedding_model=embedding_model,
            )
            result.restored += 1
            result.keys_stored.append(key)

    return result
