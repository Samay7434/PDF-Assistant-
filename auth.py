import hashlib
import json
import re
import secrets

from config import USERS_FILE


USERNAME_RE = re.compile(r"^[a-zA-Z0-9_.-]{3,32}$")


def normalize_username(username: str) -> str:
    return username.strip().lower()


def validate_username(username: str) -> str | None:
    if not USERNAME_RE.match(username):
        return "Username must be 3-32 characters and use only letters, numbers, dots, hyphens, or underscores."
    return None


def validate_password(password: str) -> str | None:
    if len(password) < 6:
        return "Password must be at least 6 characters."
    return None


def _load_users() -> dict:
    if not USERS_FILE.exists():
        return {}
    try:
        with USERS_FILE.open(encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save_users(users: dict) -> None:
    USERS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with USERS_FILE.open("w", encoding="utf-8") as f:
        json.dump(users, f, indent=2)


def _hash_password(password: str, salt: str) -> str:
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        bytes.fromhex(salt),
        200_000,
    )
    return digest.hex()


def create_user(username: str, password: str) -> tuple[bool, str]:
    username = normalize_username(username)
    username_error = validate_username(username)
    if username_error:
        return False, username_error

    password_error = validate_password(password)
    if password_error:
        return False, password_error

    users = _load_users()
    if username in users:
        return False, "That username already exists."

    salt = secrets.token_hex(16)
    users[username] = {
        "salt": salt,
        "password_hash": _hash_password(password, salt),
    }
    _save_users(users)
    return True, username


def authenticate_user(username: str, password: str) -> bool:
    username = normalize_username(username)
    users = _load_users()
    record = users.get(username)
    if not record:
        return False

    expected = record.get("password_hash", "")
    salt = record.get("salt", "")
    if not expected or not salt:
        return False

    actual = _hash_password(password, salt)
    return secrets.compare_digest(actual, expected)
