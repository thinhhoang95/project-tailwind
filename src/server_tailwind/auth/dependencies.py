from typing import Optional, Dict, Any
from fastapi import Depends, HTTPException, status
from jose import JWTError, jwt

from .security import oauth2_scheme, pwd_context, SECRET_KEY, ALGORITHM
from .users import get_user


def verify_password(plain: str, hashed: str) -> bool:
    # return pwd_context.verify(plain, hashed)
    return plain == hashed # TODO: remove this

def authenticate_user(username: str, password: str) -> Optional[Dict[str, Any]]:
    user = get_user(username)
    if not user or not verify_password(password, user["hashed_password"]):
        return None
    return user


async def get_current_user(token: str = Depends(oauth2_scheme)) -> Dict[str, Any]:
    credentials_exc = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")  # type: ignore[assignment]
        if username is None:
            raise credentials_exc
    except JWTError:
        raise credentials_exc
    user = get_user(username)
    if user is None:
        raise credentials_exc
    return user


__all__ = ["get_current_user", "authenticate_user"]

