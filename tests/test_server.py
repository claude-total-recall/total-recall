"""Tests for MCP server handlers."""

import os
import tempfile

import pytest

# Set test database path before importing
_test_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
os.environ["TOTAL_RECALL_DB"] = _test_db.name

from total_recall import db
from total_recall.server import (
    handle_memory_delete,
    handle_memory_fulltext,
    handle_memory_get,
    handle_memory_history,
    handle_memory_list,
    handle_memory_search,
    handle_memory_set,
    handle_memory_stats,
    handle_memory_tags,
)


@pytest.fixture(autouse=True)
def reset_db():
    """Reset database before each test."""
    with db.get_connection() as conn:
        conn.execute("DELETE FROM history")
        conn.execute("DELETE FROM tags")
        conn.execute("DELETE FROM memories")
    yield


@pytest.fixture(scope="module", autouse=True)
def init_database():
    """Initialize database once for all tests."""
    db.init_db()
    yield


@pytest.mark.asyncio
class TestMemorySetHandler:
    async def test_create_memory(self):
        result = await handle_memory_set({
            "key": "test.create",
            "value": "test value",
        })
        assert result.success is True
        assert result.created is True
        assert result.key == "test.create"

    async def test_create_with_tags(self):
        result = await handle_memory_set({
            "key": "test.tags",
            "value": "value",
            "tags": ["a", "b"],
        })
        assert result.success is True

        get_result = await handle_memory_get({"key": "test.tags"})
        assert set(get_result.tags) == {"a", "b"}

    async def test_update_memory(self):
        await handle_memory_set({"key": "test.update", "value": "v1"})
        result = await handle_memory_set({"key": "test.update", "value": "v2"})
        assert result.created is False
        assert result.changed is True

    async def test_returns_size_bytes(self):
        result = await handle_memory_set({"key": "test.size", "value": "hello"})
        assert result.size_bytes == 5

    async def test_returns_previous_value_on_update(self):
        await handle_memory_set({"key": "test.prev", "value": "original content"})
        result = await handle_memory_set({"key": "test.prev", "value": "new"})
        assert result.previous_value == "original content"
        assert result.previous_size_bytes == len("original content".encode("utf-8"))

    async def test_no_previous_value_on_create(self):
        result = await handle_memory_set({"key": "test.new", "value": "fresh"})
        assert result.previous_value is None
        assert result.previous_size_bytes is None

    async def test_truncation_warning(self):
        # Create with large content
        large_content = "x" * 1000
        await handle_memory_set({"key": "test.truncate", "value": large_content})
        # Update with much smaller content (>50% reduction)
        result = await handle_memory_set({"key": "test.truncate", "value": "tiny"})
        assert any("reduced by >50%" in w for w in result.warnings)
        assert any("Accidental truncation" in w for w in result.warnings)

    async def test_no_truncation_warning_above_threshold(self):
        await handle_memory_set({"key": "test.ok", "value": "1234567890"})  # 10 bytes
        result = await handle_memory_set({"key": "test.ok", "value": "123456"})  # 6 bytes, 60%
        assert not any("reduced by >50%" in w for w in result.warnings)


@pytest.mark.asyncio
class TestMemoryGetHandler:
    async def test_get_existing(self):
        await handle_memory_set({"key": "get.existing", "value": "hello"})
        result = await handle_memory_get({"key": "get.existing"})
        assert result.value == "hello"

    async def test_get_not_found(self):
        result = await handle_memory_get({"key": "never.existed"})
        assert hasattr(result, "error")


@pytest.mark.asyncio
class TestMemoryDeleteHandler:
    async def test_delete_existing(self):
        await handle_memory_set({"key": "delete.me", "value": "bye"})
        result = await handle_memory_delete({"key": "delete.me"})
        assert result.deleted is True

    async def test_delete_nonexistent(self):
        result = await handle_memory_delete({"key": "ghost"})
        assert result.deleted is False


@pytest.mark.asyncio
class TestMemoryListHandler:
    async def test_list_all(self):
        await handle_memory_set({"key": "list.one", "value": "1"})
        await handle_memory_set({"key": "list.two", "value": "2"})
        result = await handle_memory_list({})
        assert result.total >= 2

    async def test_list_with_pattern(self):
        await handle_memory_set({"key": "prefix.a", "value": "a"})
        await handle_memory_set({"key": "prefix.b", "value": "b"})
        await handle_memory_set({"key": "other", "value": "x"})
        result = await handle_memory_list({"pattern": "prefix.*"})
        assert result.total == 2


@pytest.mark.asyncio
class TestMemorySearchHandler:
    async def test_semantic_search(self):
        await handle_memory_set({
            "key": "search.python",
            "value": "Python is a programming language known for readability",
            "embed": True,
        })
        await handle_memory_set({
            "key": "search.recipe",
            "value": "How to make chocolate chip cookies",
            "embed": True,
        })

        result = await handle_memory_search({
            "query": "coding in python",
            "limit": 5,
        })
        assert len(result.results) >= 1
        # Python-related memory should rank higher
        if len(result.results) >= 2:
            keys = [r["key"] if isinstance(r, dict) else r.key for r in result.results]
            assert keys[0] == "search.python"


@pytest.mark.asyncio
class TestMemoryFulltextHandler:
    async def test_fulltext_search(self):
        await handle_memory_set({
            "key": "fulltext.item",
            "value": "The quick brown fox jumps",
        })
        result = await handle_memory_fulltext({"query": "brown fox"})
        assert result.total >= 1


@pytest.mark.asyncio
class TestMemoryTagsHandler:
    async def test_list_tags(self):
        await handle_memory_set({"key": "t1", "value": "v", "tags": ["alpha"]})
        await handle_memory_set({"key": "t2", "value": "v", "tags": ["alpha", "beta"]})
        result = await handle_memory_tags()
        tags = {t.tag: t.count for t in result.tags}
        assert tags.get("alpha", 0) >= 2
        assert tags.get("beta", 0) >= 1


@pytest.mark.asyncio
class TestMemoryStatsHandler:
    async def test_get_stats(self):
        await handle_memory_set({"key": "stats.item", "value": "content"})
        result = await handle_memory_stats()
        assert result.total_memories >= 1


@pytest.mark.asyncio
class TestMemoryHistoryHandler:
    async def test_get_history(self):
        await handle_memory_set({"key": "hist.key", "value": "v1"})
        await handle_memory_set({"key": "hist.key", "value": "v2"})
        await handle_memory_set({"key": "hist.key", "value": "v3"})

        result = await handle_memory_history({"key": "hist.key"})
        assert len(result.history) == 2  # v1 and v2 in history
        values = [h.value for h in result.history]
        assert "v1" in values
        assert "v2" in values
