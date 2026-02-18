"""Fernet encryption for sensitive credentials."""

import os
from cryptography.fernet import Fernet

_KEY_FILE = os.path.join(os.path.dirname(__file__), ".encryption_key")


def _get_or_create_key() -> bytes:
    if os.path.exists(_KEY_FILE):
        with open(_KEY_FILE, "rb") as f:
            return f.read()
    key = Fernet.generate_key()
    with open(_KEY_FILE, "wb") as f:
        f.write(key)
    return key


_fernet = Fernet(_get_or_create_key())


def encrypt(plaintext: str) -> str:
    return _fernet.encrypt(plaintext.encode()).decode()


def decrypt(ciphertext: str) -> str:
    return _fernet.decrypt(ciphertext.encode()).decode()
