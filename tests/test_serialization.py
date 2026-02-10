"""Tests for serialization module."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

_test_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
os.environ["TOTAL_RECALL_DB"] = _test_db.name

from total_recall import db, serialization
from total_recall.models import ImportResult


@pytest.fixture(scope="module", autouse=True)
def init_database():
    db.init_db()
    yield
    Path(_test_db.name).unlink(missing_ok=True)


@pytest.fixture(autouse=True)
def reset_db():
    with db.get_connection() as conn:
        conn.execute("DELETE FROM history")
        conn.execute("DELETE FROM tags")
        conn.execute("DELETE FROM memories")
    yield


def _seed():
    db.memory_set("proj.a", "aaa", tags=["t1"])
    db.memory_set("proj.b", "bbb", tags=["t2"])
    db.memory_set("other.x", "xxx")


class TestSerialize:
    def test_basic_serialize(self):
        _seed()
        payload = serialization.serialize_memories(["proj.*"])
        assert payload["version"] == 1
        assert len(payload["memories"]) == 2
        keys = {m["key"] for m in payload["memories"]}
        assert keys == {"proj.a", "proj.b"}

    def test_serialize_all(self):
        _seed()
        payload = serialization.serialize_memories(["*"])
        assert len(payload["memories"]) == 3

    def test_serialize_includes_tags(self):
        _seed()
        payload = serialization.serialize_memories(["proj.a"])
        mem = payload["memories"][0]
        assert "t1" in mem["tags"]

    def test_serialize_includes_history(self):
        db.memory_set("hist.key", "v1")
        db.memory_set("hist.key", "v2")
        payload = serialization.serialize_memories(["hist.key"], include_history=True)
        mem = payload["memories"][0]
        assert len(mem["history"]) == 1
        assert mem["history"][0]["value"] == "v1"

    def test_serialize_no_history(self):
        db.memory_set("hist.key", "v1")
        db.memory_set("hist.key", "v2")
        payload = serialization.serialize_memories(["hist.key"], include_history=False)
        mem = payload["memories"][0]
        assert "history" not in mem

    def test_serialize_empty_result(self):
        payload = serialization.serialize_memories(["nonexistent.*"])
        assert len(payload["memories"]) == 0

    def test_serialize_has_source(self):
        _seed()
        payload = serialization.serialize_memories(["*"])
        assert "source" in payload
        assert "created_at" in payload


class TestDeserialize:
    def test_basic_import(self):
        payload = {
            "version": 1,
            "source": "test",
            "created_at": "2026-01-01T00:00:00",
            "memories": [
                {
                    "key": "import.test",
                    "value": "imported",
                    "content_type": "text/plain",
                    "tags": ["imported"],
                    "created_at": "2026-01-01T00:00:00",
                    "updated_at": "2026-01-01T00:00:00",
                }
            ],
        }
        result = serialization.deserialize_memories(payload, auto_embed=False)
        assert result.restored == 1
        assert "import.test" in result.keys_stored

        record, _ = db.memory_get("import.test")
        assert record.value == "imported"
        assert "imported" in record.tags

    def test_key_filter(self):
        payload = {
            "version": 1,
            "memories": [
                {"key": "accept.me", "value": "yes", "created_at": "2026-01-01", "updated_at": "2026-01-01"},
                {"key": "reject.me", "value": "no", "created_at": "2026-01-01", "updated_at": "2026-01-01"},
            ],
        }
        result = serialization.deserialize_memories(
            payload, key_filter="accept.*", auto_embed=False
        )
        assert result.restored == 1
        assert result.filtered == 1

    def test_merge_skip_existing(self):
        db.memory_set("exists.key", "original")
        payload = {
            "version": 1,
            "memories": [
                {"key": "exists.key", "value": "new", "created_at": "2026-01-01", "updated_at": "2026-02-01"},
            ],
        }
        result = serialization.deserialize_memories(
            payload, merge="skip_existing", auto_embed=False
        )
        assert result.skipped == 1
        assert result.restored == 0

        record, _ = db.memory_get("exists.key")
        assert record.value == "original"

    def test_merge_overwrite(self):
        db.memory_set("exists.key", "original")
        payload = {
            "version": 1,
            "memories": [
                {"key": "exists.key", "value": "new", "created_at": "2026-01-01", "updated_at": "2026-01-01"},
            ],
        }
        result = serialization.deserialize_memories(
            payload, merge="overwrite", auto_embed=False
        )
        assert result.restored == 1

        record, _ = db.memory_get("exists.key")
        assert record.value == "new"

    def test_merge_newer_wins_incoming_newer(self):
        db.memory_set("ts.key", "old")
        # Get current updated_at
        with db.get_connection() as conn:
            row = conn.execute("SELECT updated_at FROM memories WHERE key = 'ts.key'").fetchone()
            old_ts = row["updated_at"]

        payload = {
            "version": 1,
            "memories": [
                {"key": "ts.key", "value": "newer", "created_at": "2026-01-01",
                 "updated_at": "2099-01-01T00:00:00"},
            ],
        }
        result = serialization.deserialize_memories(
            payload, merge="newer_wins", auto_embed=False
        )
        assert result.restored == 1

        record, _ = db.memory_get("ts.key")
        assert record.value == "newer"

    def test_merge_newer_wins_tie_local_wins(self):
        db.memory_set("tie.key", "local")
        with db.get_connection() as conn:
            row = conn.execute("SELECT updated_at FROM memories WHERE key = 'tie.key'").fetchone()
            local_ts = row["updated_at"]

        payload = {
            "version": 1,
            "memories": [
                {"key": "tie.key", "value": "remote", "created_at": "2026-01-01",
                 "updated_at": local_ts},
            ],
        }
        result = serialization.deserialize_memories(
            payload, merge="newer_wins", auto_embed=False
        )
        assert result.skipped == 1
        assert result.conflicts == 1

        record, _ = db.memory_get("tie.key")
        assert record.value == "local"

    def test_import_resets_access_stats(self):
        payload = {
            "version": 1,
            "memories": [
                {
                    "key": "stats.test",
                    "value": "val",
                    "created_at": "2026-01-01",
                    "updated_at": "2026-01-01",
                    "access_count": 99,
                }
            ],
        }
        serialization.deserialize_memories(payload, auto_embed=False)
        record, _ = db.memory_get("stats.test")
        # access_count should be 1 (from the get), not 99
        assert record.access_count == 1

    def test_atomicity_on_error(self):
        """If import fails partway, nothing should be stored."""
        payload = {
            "version": 1,
            "memories": [
                {"key": "atom.ok", "value": "fine", "created_at": "2026-01-01", "updated_at": "2026-01-01"},
                # This one has a bad value that will cause import_memory to fail
            ],
        }

        # Patch import_memory to fail on second call
        original_import = db.import_memory
        call_count = [0]

        def failing_import(conn, **kwargs):
            call_count[0] += 1
            if call_count[0] == 2:
                raise RuntimeError("Simulated failure")
            return original_import(conn, **kwargs)

        # Add a second memory
        payload["memories"].append(
            {"key": "atom.fail", "value": "boom", "created_at": "2026-01-01", "updated_at": "2026-01-01"}
        )

        with patch.object(db, "import_memory", side_effect=failing_import):
            with pytest.raises(RuntimeError):
                serialization.deserialize_memories(payload, auto_embed=False)

        # Neither should exist
        record, _ = db.memory_get("atom.ok")
        assert record is None
