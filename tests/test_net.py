"""Tests for network sharing (memory_listen / memory_send)."""

import asyncio
import os
import tempfile
from pathlib import Path

import pytest

_test_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
os.environ["TOTAL_RECALL_DB"] = _test_db.name

from total_recall import db
from total_recall.net import handle_memory_listen, handle_memory_send


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
    db.memory_set("net.alpha", "alpha value", tags=["net"])
    db.memory_set("net.beta", "beta value", tags=["net"])
    db.memory_set("other.gamma", "gamma value")


class TestListenTimeout:
    @pytest.mark.asyncio
    async def test_timeout_returns_cleanly(self):
        result = await handle_memory_listen({
            "password": "pw",
            "port": 17782,
            "timeout": 1,
        })
        assert result.success is True
        assert result.timed_out is True
        assert result.received == 0


class TestSendNoListener:
    @pytest.mark.asyncio
    async def test_send_connection_refused(self):
        _seed()
        result = await handle_memory_send({
            "keys": ["net.*"],
            "host": "127.0.0.1",
            "port": 17783,
            "password": "pw",
        })
        assert result.success is False
        assert "connect" in result.error.lower()


class TestSendReceive:
    @pytest.mark.asyncio
    async def test_full_transfer(self):
        _seed()
        port = 17784

        async def listener():
            return await handle_memory_listen({
                "password": "secret",
                "port": port,
                "timeout": 10,
                "merge": "overwrite",
            })

        async def sender():
            # Small delay to let listener bind
            await asyncio.sleep(0.3)
            return await handle_memory_send({
                "keys": ["net.*"],
                "host": "127.0.0.1",
                "port": port,
                "password": "secret",
            })

        listen_result, send_result = await asyncio.gather(listener(), sender())

        assert listen_result.success is True
        assert listen_result.received == 2
        assert set(listen_result.keys_stored) == {"net.alpha", "net.beta"}

        assert send_result.success is True
        assert send_result.sent == 2
        assert send_result.accepted == 2

    @pytest.mark.asyncio
    async def test_wrong_password(self):
        _seed()
        port = 17785

        async def listener():
            return await handle_memory_listen({
                "password": "receiver_pw",
                "port": port,
                "timeout": 10,
            })

        async def sender():
            await asyncio.sleep(0.3)
            return await handle_memory_send({
                "keys": ["net.*"],
                "host": "127.0.0.1",
                "port": port,
                "password": "different_pw",
            })

        listen_result, send_result = await asyncio.gather(listener(), sender())

        assert listen_result.success is False
        assert "password" in listen_result.error.lower()

        assert send_result.success is False

    @pytest.mark.asyncio
    async def test_key_filter_on_receiver(self):
        _seed()
        port = 17786

        async def listener():
            return await handle_memory_listen({
                "password": "pw",
                "port": port,
                "timeout": 10,
                "key_filter": "net.alpha",
                "merge": "overwrite",
            })

        async def sender():
            await asyncio.sleep(0.3)
            return await handle_memory_send({
                "keys": ["net.*"],
                "host": "127.0.0.1",
                "port": port,
                "password": "pw",
            })

        listen_result, send_result = await asyncio.gather(listener(), sender())

        assert listen_result.success is True
        assert listen_result.received == 1
        assert listen_result.filtered == 1
        assert listen_result.keys_stored == ["net.alpha"]

        assert send_result.accepted == 1
        assert send_result.filtered == 1

    @pytest.mark.asyncio
    async def test_merge_skip_existing(self):
        _seed()
        port = 17787

        async def listener():
            return await handle_memory_listen({
                "password": "pw",
                "port": port,
                "timeout": 10,
                "merge": "skip_existing",
            })

        async def sender():
            await asyncio.sleep(0.3)
            return await handle_memory_send({
                "keys": ["net.*"],
                "host": "127.0.0.1",
                "port": port,
                "password": "pw",
            })

        listen_result, send_result = await asyncio.gather(listener(), sender())

        # All already exist locally, so all skipped
        assert listen_result.success is True
        assert listen_result.skipped == 2
        assert listen_result.received == 0

    @pytest.mark.asyncio
    async def test_send_empty_pattern(self):
        result = await handle_memory_send({
            "keys": ["nonexistent.*"],
            "host": "127.0.0.1",
            "port": 17788,
            "password": "pw",
        })
        assert result.success is False
        assert "no memories" in result.error.lower()
