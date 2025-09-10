"""Authentication utilities for the Tailwind API.

This package exposes helpers for password hashing, JWT token creation,
and FastAPI dependencies to retrieve the current user from a bearer token.
"""

from .dependencies import get_current_user, authenticate_user
from .tokens import create_access_token
from .security import ACCESS_TOKEN_EXPIRE_MINUTES

__all__ = [
    "get_current_user",
    "authenticate_user",
    "create_access_token",
    "ACCESS_TOKEN_EXPIRE_MINUTES",
]

