from datetime import timedelta
import os
from fastapi.security import OAuth2PasswordBearer
from passlib.context import CryptContext

# Configurable via env var in real deployments; use default for dev
SECRET_KEY = os.getenv("TAILWIND_SECRET_KEY", "change-me")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("TAILWIND_ACCESS_TOKEN_MINUTES", "1440"))

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 scheme definition; token endpoint path is defined in main.py
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

__all__ = [
    "SECRET_KEY",
    "ALGORITHM",
    "ACCESS_TOKEN_EXPIRE_MINUTES",
    "pwd_context",
    "oauth2_scheme",
]

