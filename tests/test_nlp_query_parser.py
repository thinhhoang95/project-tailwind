import asyncio
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

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

from server_tailwind.query.NLPQueryParser import (  # noqa: E402
    NLPQueryParser,
    NLPQueryParserError,
)


class StubUsage:
    def __init__(self, prompt_tokens=None, completion_tokens=None):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class StubResponse:
    def __init__(self, content: str, usage=None):
        self.choices = [SimpleNamespace(message=SimpleNamespace(content=content))]
        self.usage = usage


class StubCompletions:
    def __init__(self, response: StubResponse):
        self._response = response
        self.last_kwargs = None

    async def create(self, **kwargs):
        self.last_kwargs = kwargs
        return self._response


class StubClient:
    def __init__(self, response: StubResponse):
        self.chat = SimpleNamespace(completions=StubCompletions(response))


class StubFlightList:
    time_bin_minutes = 15
    num_time_bins_per_tv = 4
    tv_id_to_idx = {"TVA": 0, "TVB": 1}
    flight_metadata = {
        "F1": {"origin": "AAA", "destination": "BBB"},
        "F2": {"origin": "CCC", "destination": "DDD"},
    }


def test_nlp_query_parser_success(monkeypatch):
    flight_list = StubFlightList()
    resources = SimpleNamespace(flight_list=flight_list)
    query_wrapper = SimpleNamespace(flight_list=flight_list)

    content = json.dumps({"query": {"type": "cross", "tv": "TVA"}})
    usage = StubUsage(prompt_tokens=123, completion_tokens=45)
    client = StubClient(StubResponse(content, usage=usage))

    parser = NLPQueryParser(resources=resources, query_wrapper=query_wrapper, client=client)
    result = asyncio.run(parser.parse("Flights crossing TVA", model="stub-model", debug=True))

    assert result.query == {"type": "cross", "tv": "TVA"}
    assert result.ast == json.loads(content)
    assert result.llm["model"] == "stub-model"
    assert result.llm["prompt_tokens"] == 123
    assert result.llm["completion_tokens"] == 45
    assert "parse_ms" in result.llm

    context = parser._context_payload()
    assert context["time_bin_minutes"] == 15
    assert context["known_tv_ids"] == ["TVA", "TVB"]
    assert "known_airports" in context


def test_nlp_query_parser_invalid_json(monkeypatch):
    flight_list = StubFlightList()
    resources = SimpleNamespace(flight_list=flight_list)
    query_wrapper = SimpleNamespace(flight_list=flight_list)

    client = StubClient(StubResponse("not-json"))
    parser = NLPQueryParser(resources=resources, query_wrapper=query_wrapper, client=client)

    with pytest.raises(NLPQueryParserError) as exc:
        asyncio.run(parser.parse("Flights crossing TVA"))
    assert exc.value.status_code == 400
