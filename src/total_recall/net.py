"""Network sharing tools for Total Recall — memory_listen / memory_send."""

import asyncio
import struct
import time

from cryptography.fernet import InvalidToken
from mcp.types import Tool

from . import crypto, serialization
from .models import ErrorResponse, ListenResponse, SendResponse

HEADER_SIZE = 26  # 4 (magic) + 2 (version) + 16 (salt) + 4 (payload_len)
ACK_HEADER_SIZE = 8  # 4 (magic) + 4 (response_len)
DEFAULT_PORT = 7782


def net_tools() -> list[Tool]:
    """Tool definitions for network sharing."""
    return [
        Tool(
            name="memory_listen",
            description="""Listen for incoming memory transfer on TCP.

Blocks until a sender connects or timeout expires. Accepts one connection,
decrypts, applies key filter and merge strategy, then returns results.
Password and address are shared out-of-band.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "password": {
                        "type": "string",
                        "description": "Shared encryption password",
                    },
                    "port": {
                        "type": "integer",
                        "description": "TCP port to listen on (default: 7782)",
                        "default": DEFAULT_PORT,
                    },
                    "key_filter": {
                        "type": "string",
                        "description": "Glob — only accept memories matching this (default: '*')",
                        "default": "*",
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Seconds before giving up (default: 120)",
                        "default": 120,
                    },
                    "merge": {
                        "type": "string",
                        "description": "Merge strategy: 'newer_wins', 'skip_existing', or 'overwrite'",
                        "default": "newer_wins",
                        "enum": ["newer_wins", "skip_existing", "overwrite"],
                    },
                },
                "required": ["password"],
            },
        ),
        Tool(
            name="memory_send",
            description="""Send memories to a listening receiver over TCP.

Connects to a receiver running memory_listen, encrypts and transmits
selected memories, and waits for acknowledgment.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "keys": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Glob patterns for memories to send",
                    },
                    "host": {
                        "type": "string",
                        "description": "Receiver hostname or IP",
                    },
                    "port": {
                        "type": "integer",
                        "description": "Receiver TCP port (default: 7782)",
                        "default": DEFAULT_PORT,
                    },
                    "password": {
                        "type": "string",
                        "description": "Shared encryption password",
                    },
                    "include_history": {
                        "type": "boolean",
                        "description": "Include version history",
                        "default": False,
                    },
                    "include_embeddings": {
                        "type": "boolean",
                        "description": "Include embedding vectors",
                        "default": False,
                    },
                },
                "required": ["keys", "host", "password"],
            },
        ),
    ]


async def handle_memory_listen(args: dict) -> ListenResponse | ErrorResponse:
    """TCP receiver: listen, accept one connection, decrypt, import."""
    password = args["password"]
    port = args.get("port", DEFAULT_PORT)
    key_filter = args.get("key_filter", "*")
    timeout_secs = args.get("timeout", 120)
    merge = args.get("merge", "newer_wins")

    start = time.monotonic()
    result_future: asyncio.Future = asyncio.get_event_loop().create_future()

    async def _handle_connection(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        try:
            # Read header
            header = await asyncio.wait_for(reader.readexactly(HEADER_SIZE), timeout=30)
            magic = header[:4]
            if magic != b"TRMX":
                result_future.set_result(
                    ListenResponse(success=False, error=f"Bad magic: {magic!r}")
                )
                writer.close()
                return

            version = struct.unpack(">H", header[4:6])[0]
            if version != 1:
                result_future.set_result(
                    ListenResponse(success=False, error=f"Unsupported version: {version}")
                )
                writer.close()
                return

            salt = header[6:22]
            payload_len = struct.unpack(">I", header[22:26])[0]

            # Read ciphertext
            ciphertext = await asyncio.wait_for(reader.readexactly(payload_len), timeout=300)

            # Decrypt
            try:
                payload = crypto.decrypt_payload(ciphertext, password, salt)
            except InvalidToken:
                # Wrong password — can't encrypt a response they can read, just close
                result_future.set_result(
                    ListenResponse(
                        success=False,
                        elapsed=time.monotonic() - start,
                        error="Decryption failed — sender used a different password. No memories were stored.",
                    )
                )
                writer.close()
                return

            # Deserialize and import
            import_result = serialization.deserialize_memories(
                payload, key_filter=key_filter, merge=merge, auto_embed=True
            )

            # Build ack
            ack_data = {
                "accepted": import_result.restored,
                "filtered": import_result.filtered,
                "skipped": import_result.skipped,
            }

            # Encrypt ack with same key
            key = crypto.derive_key(password, salt)
            from cryptography.fernet import Fernet

            f = Fernet(key)
            import json

            ack_json = json.dumps(ack_data).encode()
            encrypted_ack = f.encrypt(ack_json)

            # Send TRAK response
            writer.write(b"TRAK")
            writer.write(struct.pack(">I", len(encrypted_ack)))
            writer.write(encrypted_ack)
            await writer.drain()
            writer.close()

            elapsed = time.monotonic() - start
            result_future.set_result(
                ListenResponse(
                    success=True,
                    received=import_result.restored,
                    filtered=import_result.filtered,
                    skipped=import_result.skipped,
                    keys_stored=import_result.keys_stored,
                    elapsed=round(elapsed, 2),
                    warnings=import_result.warnings,
                )
            )

        except Exception as e:
            if not result_future.done():
                result_future.set_result(
                    ListenResponse(
                        success=False,
                        elapsed=time.monotonic() - start,
                        error=str(e),
                    )
                )
            try:
                writer.close()
            except Exception:
                pass

    server = await asyncio.start_server(_handle_connection, "0.0.0.0", port)

    try:
        result = await asyncio.wait_for(result_future, timeout=timeout_secs)
    except asyncio.TimeoutError:
        result = ListenResponse(
            success=True,
            timed_out=True,
            elapsed=round(time.monotonic() - start, 2),
        )
    finally:
        server.close()
        await server.wait_closed()

    return result


async def handle_memory_send(args: dict) -> SendResponse | ErrorResponse:
    """TCP sender: connect, encrypt, transmit, wait for ack."""
    keys = args["keys"]
    host = args["host"]
    port = args.get("port", DEFAULT_PORT)
    password = args["password"]
    include_history = args.get("include_history", False)
    include_embeddings = args.get("include_embeddings", False)

    # Serialize
    payload = serialization.serialize_memories(
        keys=keys,
        include_history=include_history,
        include_embeddings=include_embeddings,
    )

    if not payload["memories"]:
        return ErrorResponse(error="No memories matched the key patterns")

    sent_count = len(payload["memories"])

    # Encrypt
    salt, ciphertext = crypto.encrypt_payload(payload, password)

    # Connect
    try:
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(host, port), timeout=30
        )
    except (ConnectionRefusedError, OSError, asyncio.TimeoutError) as e:
        return SendResponse(
            success=False,
            sent=0,
            error=f"Could not connect to {host}:{port} — {e}",
        )

    try:
        # Send TRMX frame
        writer.write(b"TRMX")
        writer.write(struct.pack(">H", 1))  # version
        writer.write(salt)
        writer.write(struct.pack(">I", len(ciphertext)))
        writer.write(ciphertext)
        await writer.drain()

        # Read TRAK ack
        try:
            ack_header = await asyncio.wait_for(reader.readexactly(ACK_HEADER_SIZE), timeout=300)
        except (asyncio.IncompleteReadError, asyncio.TimeoutError):
            return SendResponse(
                success=False,
                sent=sent_count,
                error="Connection closed by receiver without acknowledgment — likely wrong password.",
            )

        ack_magic = ack_header[:4]
        if ack_magic != b"TRAK":
            return SendResponse(
                success=False,
                sent=sent_count,
                error=f"Bad ack magic: {ack_magic!r}",
            )

        ack_len = struct.unpack(">I", ack_header[4:8])[0]

        try:
            encrypted_ack = await asyncio.wait_for(reader.readexactly(ack_len), timeout=30)
        except (asyncio.IncompleteReadError, asyncio.TimeoutError):
            return SendResponse(
                success=False,
                sent=sent_count,
                error="Failed to read ack body",
            )

        # Decrypt ack
        try:
            ack_data = crypto.decrypt_payload(encrypted_ack, password, salt)
        except InvalidToken:
            return SendResponse(
                success=False,
                sent=sent_count,
                error="Failed to decrypt acknowledgment",
            )

        return SendResponse(
            success=True,
            sent=sent_count,
            accepted=ack_data.get("accepted", 0),
            filtered=ack_data.get("filtered", 0),
            skipped=ack_data.get("skipped", 0),
        )

    finally:
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass
