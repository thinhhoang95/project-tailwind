from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from jose import jwt
from .security import SECRET_KEY, ALGORITHM


def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


__all__ = ["create_access_token"]

