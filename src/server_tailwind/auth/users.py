from typing import Dict, Optional
from .security import pwd_context

# Demo users; replace with real user store / DB
_fake_users: Dict[str, Dict[str, str]] = {
    "thinh.hoangdinh@enac.fr": {
        "username": "thinh.hoangdinh@enac.fr",
        # For demo purposes we hash at import; in real life store the hash persistently
        "hashed_password": "Vy011195",
    }
}


def get_user(username: str) -> Optional[Dict[str, str]]:
    return _fake_users.get(username)


__all__ = ["get_user"]

