"""Tests for backup/restore functionality."""

import os
import struct
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

# Set test database path before importing
_test_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
os.environ["TOTAL_RECALL_DB"] = _test_db.name

from total_recall import crypto, db, serialization
from total_recall.server import handle_memory_backup, handle_memory_restore


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


@pytest.fixture
def tmp_backup(tmp_path):
    return tmp_path / "test.trbak"


def _seed_memories():
    """Seed some test data."""
    db.memory_set("project.alpha", "alpha content", tags=["test"])
    db.memory_set("project.beta", "beta content", tags=["test", "beta"])
    db.memory_set("other.key", "other content")
    # Create some history
    db.memory_set("project.alpha", "alpha v2")


class TestBackup:
    @pytest.mark.asyncio
    async def test_basic_backup(self, tmp_backup):
        _seed_memories()
        result = await handle_memory_backup({
            "password": "test123",
            "path": str(tmp_backup),
        })
        assert result.success is True
        assert result.memory_count == 3
        assert result.size_bytes > 0
        assert tmp_backup.exists()

    @pytest.mark.asyncio
    async def test_backup_file_format(self, tmp_backup):
        _seed_memories()
        await handle_memory_backup({
            "password": "pw",
            "path": str(tmp_backup),
        })
        with open(tmp_backup, "rb") as f:
            assert f.read(4) == b"TRBK"
            version = struct.unpack(">H", f.read(2))[0]
            assert version == 1
            salt = f.read(16)
            assert len(salt) == 16
            payload_len = struct.unpack(">I", f.read(4))[0]
            assert payload_len > 0

    @pytest.mark.asyncio
    async def test_backup_with_key_filter(self, tmp_backup):
        _seed_memories()
        result = await handle_memory_backup({
            "password": "pw",
            "path": str(tmp_backup),
            "keys": "project.*",
        })
        assert result.success is True
        assert result.memory_count == 2

    @pytest.mark.asyncio
    async def test_backup_no_matches(self, tmp_backup):
        _seed_memories()
        result = await handle_memory_backup({
            "password": "pw",
            "path": str(tmp_backup),
            "keys": "nonexistent.*",
        })
        assert result.success is False

    @pytest.mark.asyncio
    async def test_backup_default_path(self):
        _seed_memories()
        result = await handle_memory_backup({"password": "pw"})
        assert result.success is True
        assert ".total_recall/backups/" in result.path
        assert result.path.endswith(".trbak")
        # Cleanup
        Path(result.path).unlink(missing_ok=True)


class TestRestore:
    @pytest.mark.asyncio
    async def test_basic_restore(self, tmp_backup):
        _seed_memories()
        await handle_memory_backup({
            "password": "secret",
            "path": str(tmp_backup),
        })

        # Clear DB
        with db.get_connection() as conn:
            conn.execute("DELETE FROM history")
            conn.execute("DELETE FROM tags")
            conn.execute("DELETE FROM memories")

        result = await handle_memory_restore({
            "password": "secret",
            "path": str(tmp_backup),
        })
        assert result.success is True
        assert result.restored == 3

        # Verify data
        record, _ = db.memory_get("project.alpha")
        assert record is not None
        assert record.value == "alpha v2"

    @pytest.mark.asyncio
    async def test_restore_wrong_password(self, tmp_backup):
        _seed_memories()
        await handle_memory_backup({
            "password": "correct",
            "path": str(tmp_backup),
        })
        result = await handle_memory_restore({
            "password": "wrong",
            "path": str(tmp_backup),
        })
        assert result.success is False
        assert "wrong password" in result.error.lower()

    @pytest.mark.asyncio
    async def test_restore_file_not_found(self):
        result = await handle_memory_restore({
            "password": "pw",
            "path": "/tmp/nonexistent.trbak",
        })
        assert result.success is False

    @pytest.mark.asyncio
    async def test_restore_bad_magic(self, tmp_path):
        bad_file = tmp_path / "bad.trbak"
        bad_file.write_bytes(b"NOPE" + b"\x00" * 100)
        result = await handle_memory_restore({
            "password": "pw",
            "path": str(bad_file),
        })
        assert result.success is False
        assert "magic" in result.error.lower()

    @pytest.mark.asyncio
    async def test_restore_with_key_filter(self, tmp_backup):
        _seed_memories()
        await handle_memory_backup({
            "password": "pw",
            "path": str(tmp_backup),
        })

        with db.get_connection() as conn:
            conn.execute("DELETE FROM history")
            conn.execute("DELETE FROM tags")
            conn.execute("DELETE FROM memories")

        result = await handle_memory_restore({
            "password": "pw",
            "path": str(tmp_backup),
            "key_filter": "project.*",
        })
        assert result.restored == 2

    @pytest.mark.asyncio
    async def test_restore_skip_existing(self, tmp_backup):
        _seed_memories()
        await handle_memory_backup({
            "password": "pw",
            "path": str(tmp_backup),
        })

        # Don't clear — restore with skip_existing
        result = await handle_memory_restore({
            "password": "pw",
            "path": str(tmp_backup),
            "merge": "skip_existing",
        })
        assert result.skipped == 3
        assert result.restored == 0

    @pytest.mark.asyncio
    async def test_restore_overwrite(self, tmp_backup):
        _seed_memories()
        await handle_memory_backup({
            "password": "pw",
            "path": str(tmp_backup),
        })

        # Modify a memory
        db.memory_set("project.alpha", "modified locally")

        result = await handle_memory_restore({
            "password": "pw",
            "path": str(tmp_backup),
            "merge": "overwrite",
        })
        assert result.restored == 3

        record, _ = db.memory_get("project.alpha")
        assert record.value == "alpha v2"  # Restored from backup

    @pytest.mark.asyncio
    async def test_restore_preserves_history(self, tmp_backup):
        _seed_memories()
        await handle_memory_backup({
            "password": "pw",
            "path": str(tmp_backup),
            "include_history": True,
        })

        with db.get_connection() as conn:
            conn.execute("DELETE FROM history")
            conn.execute("DELETE FROM tags")
            conn.execute("DELETE FROM memories")

        await handle_memory_restore({
            "password": "pw",
            "path": str(tmp_backup),
        })

        entries, _, _ = db.get_history("project.alpha")
        assert len(entries) == 1
        assert entries[0].value == "alpha content"

    @pytest.mark.asyncio
    async def test_restore_resets_access_stats(self, tmp_backup):
        _seed_memories()
        # Bump access count
        db.memory_get("project.alpha")
        db.memory_get("project.alpha")

        await handle_memory_backup({
            "password": "pw",
            "path": str(tmp_backup),
        })

        with db.get_connection() as conn:
            conn.execute("DELETE FROM history")
            conn.execute("DELETE FROM tags")
            conn.execute("DELETE FROM memories")

        await handle_memory_restore({
            "password": "pw",
            "path": str(tmp_backup),
        })

        record, _ = db.memory_get("project.alpha")
        # access_count should be 1 (from the get we just did), not 2+
        assert record.access_count == 1
