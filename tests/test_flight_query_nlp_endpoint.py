import asyncio
import sys
from pathlib import Path
from types import ModuleType

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

if "geopandas" not in sys.modules:
    geopandas_stub = ModuleType("geopandas")
    geopandas_stub.read_file = lambda *args, **kwargs: None
    geopandas_stub.GeoDataFrame = object
    sys.modules["geopandas"] = geopandas_stub

fastapi = pytest.importorskip("fastapi")
from fastapi import HTTPException  # noqa: E402

from server_tailwind.query.NLPQueryParser import NLPQueryParserResult  # noqa: E402
from server_tailwind import main as server_main  # noqa: E402


def test_flight_query_nlp_endpoint_success(monkeypatch):
    async def fake_parse(prompt, model=None, debug=False):
        assert prompt == "Flights that cross TVA"
        assert debug is True
        return NLPQueryParserResult(
            query={"type": "cross", "tv": "TVA"},
            ast={"query": {"type": "cross", "tv": "TVA"}},
            llm={"model": "stub-model", "parse_ms": 1.2},
        )

    async def fake_evaluate(payload):
        assert payload == {
            "query": {"type": "cross", "tv": "TVA"},
            "options": {"select": "flight_ids", "order_by": "takeoff_time", "limit": 25, "debug": True},
        }
        return {"flight_ids": ["F1"], "metadata": {"time_bin_minutes": 15, "bins_per_tv": 96}}

    monkeypatch.setattr(server_main.nlp_query_parser, "parse", fake_parse)
    monkeypatch.setattr(server_main.query_wrapper, "evaluate", fake_evaluate)

    payload = {
        "prompt": "Flights that cross TVA",
        "options": {"select": "flight_ids", "order_by": "takeoff_time", "limit": 25, "debug": True},
        "model": "stub-model",
    }

    result = asyncio.run(server_main.post_flight_query_nlp(payload, current_user={"username": "tester"}))
    assert result["flight_ids"] == ["F1"]
    assert result["ast"]["query"]["tv"] == "TVA"
    assert result["metadata"]["llm"]["model"] == "stub-model"
    assert result["metadata"]["llm"]["parse_ms"] == 1.2


def test_flight_query_nlp_endpoint_invalid_option(monkeypatch):
    async def fail_parse(*args, **kwargs):  # pragma: no cover - should not run
        pytest.fail("parse should not be called for invalid options")

    monkeypatch.setattr(server_main.nlp_query_parser, "parse", fail_parse)

    payload = {"prompt": "Flights", "options": {"unknown": True}}
    with pytest.raises(HTTPException) as exc:
        asyncio.run(server_main.post_flight_query_nlp(payload, current_user={"username": "tester"}))
    assert exc.value.status_code == 400


def test_flight_query_nlp_endpoint_unknown_tv(monkeypatch):
    async def fake_parse(prompt, model=None, debug=False):
        return NLPQueryParserResult(
            query={"type": "cross", "tv": "UNKNOWN"},
            ast={"query": {"type": "cross", "tv": "UNKNOWN"}},
            llm={"model": "stub-model", "parse_ms": 3.4},
        )

    async def fake_evaluate(payload):
        raise ValueError("Unknown traffic volume id UNKNOWN")

    monkeypatch.setattr(server_main.nlp_query_parser, "parse", fake_parse)
    monkeypatch.setattr(server_main.query_wrapper, "evaluate", fake_evaluate)

    with pytest.raises(HTTPException) as exc:
        asyncio.run(server_main.post_flight_query_nlp({"prompt": "Flights"}, current_user={"username": "tester"}))
    assert exc.value.status_code == 404
