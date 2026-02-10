"""Tests for crypto module."""

import os

import pytest
from cryptography.fernet import InvalidToken

from total_recall.crypto import decrypt_payload, derive_key, encrypt_payload


class TestDeriveKey:
    def test_deterministic(self):
        salt = b"\x00" * 16
        k1 = derive_key("password", salt)
        k2 = derive_key("password", salt)
        assert k1 == k2

    def test_different_salts_different_keys(self):
        k1 = derive_key("password", b"\x00" * 16)
        k2 = derive_key("password", b"\x01" * 16)
        assert k1 != k2

    def test_different_passwords_different_keys(self):
        salt = os.urandom(16)
        k1 = derive_key("alpha", salt)
        k2 = derive_key("beta", salt)
        assert k1 != k2

    def test_key_length(self):
        key = derive_key("test", os.urandom(16))
        # Fernet keys are 44-byte base64-encoded strings
        assert len(key) == 44


class TestEncryptDecrypt:
    def test_round_trip(self):
        data = {"memories": [{"key": "test", "value": "hello"}]}
        salt, ciphertext = encrypt_payload(data, "secret")
        result = decrypt_payload(ciphertext, "secret", salt)
        assert result == data

    def test_wrong_password_raises(self):
        data = {"test": True}
        salt, ciphertext = encrypt_payload(data, "correct")
        with pytest.raises(InvalidToken):
            decrypt_payload(ciphertext, "wrong", salt)

    def test_unicode_content(self):
        data = {"text": "Hello \u2014 World \U0001f600"}
        salt, ciphertext = encrypt_payload(data, "pw")
        result = decrypt_payload(ciphertext, "pw", salt)
        assert result["text"] == "Hello \u2014 World \U0001f600"

    def test_salt_is_random(self):
        data = {"x": 1}
        salt1, _ = encrypt_payload(data, "pw")
        salt2, _ = encrypt_payload(data, "pw")
        assert salt1 != salt2

    def test_large_payload(self):
        data = {"big": "x" * 100_000}
        salt, ciphertext = encrypt_payload(data, "pw")
        result = decrypt_payload(ciphertext, "pw", salt)
        assert result["big"] == "x" * 100_000
