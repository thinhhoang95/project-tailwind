from typing import Any, Optional, Tuple

_GLOBAL_INDEXER: Optional[Any] = None
_GLOBAL_FL: Optional[Any] = None


def set_global_resources(indexer: Any, flight_list: Any) -> None:
    global _GLOBAL_INDEXER, _GLOBAL_FL
    _GLOBAL_INDEXER = indexer
    _GLOBAL_FL = flight_list


def get_global_resources() -> Tuple[Optional[Any], Optional[Any]]:
    return _GLOBAL_INDEXER, _GLOBAL_FL


