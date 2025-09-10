from typing import Dict, Optional
from .security import pwd_context

# Demo users; replace with real user store / DB
_fake_users: Dict[str, Dict[str, str]] = {
    "thinh.hoangdinh@enac.fr": {
        "username": "thinh.hoangdinh@enac.fr",
        # For demo purposes we hash at import; in real life store the hash persistently
        "hashed_password": "Vy011195",
        "display_name": "Thinh Hoang",
        "organization": "ENAC"
    },
    "nm@intuelle.com": {
        "username": "nm@intuelle.com",
        "hashed_password": "nm123",
        "display_name": "Network Manager",
        "organization": "EUROCONTROL"
    },
    "ltm@intuelle.com": {
        "username": "ltm@intuelle.com",
        "hashed_password": "ltm123",
        "display_name": "Local Traffic Manager",
        "organization": "ANSP"
    }
}


def get_user(username: str) -> Optional[Dict[str, str]]:
    return _fake_users.get(username)


__all__ = ["get_user"]

