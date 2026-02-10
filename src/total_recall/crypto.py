"""Encryption primitives for Total Recall backup and network sharing."""

import base64
import json
import os

from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

PBKDF2_ITERATIONS = 600_000


def derive_key(password: str, salt: bytes) -> bytes:
    """Derive a Fernet key from password + salt using PBKDF2-HMAC-SHA256."""
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=PBKDF2_ITERATIONS,
    )
    return base64.urlsafe_b64encode(kdf.derive(password.encode()))


def encrypt_payload(data: dict, password: str) -> tuple[bytes, bytes]:
    """Serialize dict to JSON, encrypt with Fernet.

    Returns:
        (salt, ciphertext) — salt is 16 random bytes, ciphertext is the Fernet token.
    """
    salt = os.urandom(16)
    key = derive_key(password, salt)
    f = Fernet(key)
    payload_json = json.dumps(data, ensure_ascii=False).encode()
    ciphertext = f.encrypt(payload_json)
    return salt, ciphertext


def decrypt_payload(ciphertext: bytes, password: str, salt: bytes) -> dict:
    """Decrypt Fernet token, deserialize JSON.

    Raises:
        InvalidToken: Wrong password (HMAC mismatch).
    """
    key = derive_key(password, salt)
    f = Fernet(key)
    payload_json = f.decrypt(ciphertext)
    return json.loads(payload_json)
