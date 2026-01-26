"""Tests for database operations."""

import os
import tempfile
from pathlib import Path

import pytest

# Set test database path before importing db
_test_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
os.environ["TOTAL_RECALL_DB"] = _test_db.name

from total_recall import db


@pytest.fixture(autouse=True)
def reset_db():
    """Reset database before each test."""
    # Clear existing data
    with db.get_connection() as conn:
        conn.execute("DELETE FROM history")
        conn.execute("DELETE FROM tags")
        conn.execute("DELETE FROM memories")
    yield
    # Cleanup after tests


@pytest.fixture(scope="module", autouse=True)
def init_database():
    """Initialize database once for all tests."""
    db.init_db()
    yield
    # Cleanup temp file
    Path(_test_db.name).unlink(missing_ok=True)


class TestMemorySet:
    def test_create_new_memory(self):
        created, changed, prev_val, prev_size, warnings = db.memory_set("test.key", "test value")
        assert created is True
        assert changed is True
        assert prev_val is None
        assert prev_size is None
        assert warnings == []

    def test_update_existing_memory(self):
        db.memory_set("update.key", "original")
        created, changed, prev_val, prev_size, warnings = db.memory_set("update.key", "updated")
        assert created is False
        assert changed is True
        assert prev_val == "original"
        assert prev_size == len("original".encode("utf-8"))

    def test_no_change_same_value(self):
        db.memory_set("same.key", "same value")
        created, changed, prev_val, prev_size, warnings = db.memory_set("same.key", "same value")
        assert created is False
        assert changed is False
        assert prev_val == "same value"
        assert prev_size == len("same value".encode("utf-8"))

    def test_key_normalized_to_lowercase(self):
        db.memory_set("MyKey", "value")
        record, _ = db.memory_get("mykey")
        assert record is not None
        assert record.key == "mykey"

    def test_tags_normalized(self):
        db.memory_set("tagged", "value", tags=["Python", "  ASYNC  "])
        record, _ = db.memory_get("tagged")
        assert set(record.tags) == {"python", "async"}

    def test_tags_deduplicated(self):
        db.memory_set("dupes", "value", tags=["a", "A", "a"])
        record, _ = db.memory_get("dupes")
        assert record.tags == ["a"]

    def test_size_warning(self):
        large_value = "x" * (101 * 1024)
        _, _, _, _, warnings = db.memory_set("large", large_value)
        assert any("100KB" in w for w in warnings)


class TestMemoryGet:
    def test_get_existing(self):
        db.memory_set("get.test", "the value", tags=["tag1"])
        record, warnings = db.memory_get("get.test")
        assert record is not None
        assert record.key == "get.test"
        assert record.value == "the value"
        assert "tag1" in record.tags

    def test_get_nonexistent(self):
        record, _ = db.memory_get("nonexistent")
        assert record is None

    def test_access_count_increments(self):
        db.memory_set("counter", "value")
        record1, _ = db.memory_get("counter")
        record2, _ = db.memory_get("counter")
        assert record2.access_count == record1.access_count + 1


class TestMemoryDelete:
    def test_delete_existing(self):
        db.memory_set("to.delete", "value")
        deleted, _ = db.memory_delete("to.delete")
        assert deleted is True

    def test_delete_nonexistent(self):
        deleted, _ = db.memory_delete("never.existed")
        assert deleted is False


class TestMemoryList:
    def test_list_all(self):
        db.memory_set("list.a", "a")
        db.memory_set("list.b", "b")
        items, total, _ = db.memory_list()
        assert total >= 2

    def test_list_with_pattern(self):
        db.memory_set("pattern.one", "1")
        db.memory_set("pattern.two", "2")
        db.memory_set("other", "x")
        items, total, _ = db.memory_list(pattern="pattern.*")
        assert total == 2
        assert all(i.key.startswith("pattern.") for i in items)

    def test_list_with_tag(self):
        db.memory_set("tagged.a", "a", tags=["special"])
        db.memory_set("tagged.b", "b", tags=["other"])
        items, total, _ = db.memory_list(tag="special")
        assert total == 1
        assert items[0].key == "tagged.a"

    def test_list_pagination(self):
        for i in range(5):
            db.memory_set(f"page.{i}", str(i))
        items, total, _ = db.memory_list(pattern="page.*", limit=2, offset=0)
        assert len(items) == 2
        assert total == 5


class TestMemoryFulltext:
    def test_search_in_value(self):
        db.memory_set("fulltext.test", "The quick brown fox")
        results, total, _ = db.memory_fulltext("brown")
        assert total >= 1
        assert any(r.key == "fulltext.test" for r in results)

    def test_search_in_key(self):
        db.memory_set("searchable.item", "value")
        results, total, _ = db.memory_fulltext("searchable")
        assert total >= 1

    def test_snippet_generation(self):
        long_text = "prefix " * 20 + "FINDME" + " suffix" * 20
        db.memory_set("snippet.test", long_text)
        results, _, _ = db.memory_fulltext("FINDME")
        assert any("FINDME" in r.snippet for r in results)


class TestHistory:
    def test_history_created_on_change(self):
        db.memory_set("history.test", "v1")
        db.memory_set("history.test", "v2")
        entries, truncated, _ = db.get_history("history.test")
        assert len(entries) == 1
        assert entries[0].value == "v1"

    def test_no_history_on_no_change(self):
        db.memory_set("no.change", "same")
        db.memory_set("no.change", "same")
        entries, _, _ = db.get_history("no.change")
        assert len(entries) == 0


class TestTags:
    def test_get_all_tags(self):
        db.memory_set("tags.a", "a", tags=["alpha", "beta"])
        db.memory_set("tags.b", "b", tags=["alpha"])
        tags = db.get_all_tags()
        alpha = next((t for t in tags if t.tag == "alpha"), None)
        assert alpha is not None
        assert alpha.count >= 2


class TestStats:
    def test_get_stats(self):
        db.memory_set("stats.test", "value")
        stats = db.get_stats()
        assert stats["total_memories"] >= 1
        assert stats["total_size_bytes"] > 0
