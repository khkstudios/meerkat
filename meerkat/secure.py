"""Encryption at rest for biometric data (face embeddings), plus secure deletion.

Galleries are encrypted with Fernet (AES-128-CBC + HMAC-SHA256, from `cryptography`). The key never
touches the project folder:

  * macOS: stored in the login Keychain (via `keyring`), service "meerkat", account "gallery-key".
  * elsewhere: the OS keyring (Secret Service on Linux, Credential Locker on Windows).
  * headless / CI: set MEERKAT_GALLERY_KEY to a Fernet key (`python -m meerkat gallery newkey`).

Losing the key makes the gallery unreadable, which is the point: `meerkat forget` deletes both.
Under POPIA face embeddings are special personal information; see RESEARCH.md section 11.
"""
from __future__ import annotations

import os
import secrets
import sys
from pathlib import Path

SERVICE, ACCOUNT, ENV_KEY = "meerkat", "gallery-key", "MEERKAT_GALLERY_KEY"


class KeyUnavailable(RuntimeError):
    pass


def _keyring():
    try:
        import keyring
        from keyring.backends.fail import Keyring as FailKeyring
    except ImportError as e:
        raise KeyUnavailable("pip install keyring (or set MEERKAT_GALLERY_KEY)") from e
    if isinstance(keyring.get_keyring(), FailKeyring):
        raise KeyUnavailable("no OS keyring available here: set MEERKAT_GALLERY_KEY "
                             "(generate one with `python -m meerkat gallery newkey`)")
    return keyring


def new_key() -> str:
    from cryptography.fernet import Fernet
    return Fernet.generate_key().decode()


def get_key(create: bool = False) -> bytes | None:
    """The gallery key: env var first, then the OS keyring. create=True makes one if missing."""
    if os.environ.get(ENV_KEY):
        return os.environ[ENV_KEY].encode()
    kr = _keyring()
    key = kr.get_password(SERVICE, ACCOUNT)
    if key is None and create:
        key = new_key()
        kr.set_password(SERVICE, ACCOUNT, key)
        print("[secure] created a gallery encryption key in the OS keychain "
              f"(service '{SERVICE}', account '{ACCOUNT}')", file=sys.stderr)
    return key.encode() if key else None


def delete_key() -> bool:
    if os.environ.get(ENV_KEY):
        return False  # not ours to delete; the caller owns the env var
    try:
        kr = _keyring()
        if kr.get_password(SERVICE, ACCOUNT) is None:
            return False
        kr.delete_password(SERVICE, ACCOUNT)
        return True
    except KeyUnavailable:
        return False


def encrypt(data: bytes) -> bytes:
    from cryptography.fernet import Fernet
    return Fernet(get_key(create=True)).encrypt(data)


def decrypt(token: bytes) -> bytes:
    from cryptography.fernet import Fernet, InvalidToken
    key = get_key(create=False)
    if key is None:
        raise KeyUnavailable("the gallery is encrypted but its key is missing (deleted from the "
                             "keychain, or MEERKAT_GALLERY_KEY unset). Re-enrol to start over.")
    try:
        return Fernet(key).decrypt(token)
    except InvalidToken as e:
        raise KeyUnavailable("the gallery key doesn't match this gallery (wrong key, or the file "
                             "was modified)") from e


def write_private(path: Path, data: bytes) -> None:
    """Atomic write, owner-only permissions (0600) from the moment the file exists."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def secure_delete(path: Path) -> bool:
    """Overwrite with random bytes, then unlink. Best effort: SSDs and APFS snapshots may keep old
    blocks, which is why galleries are encrypted in the first place."""
    if not path.exists():
        return False
    size = path.stat().st_size
    with open(path, "r+b", buffering=0) as f:
        f.write(secrets.token_bytes(size))
        os.fsync(f.fileno())
    path.unlink()
    return True
