"""Utilities for converting natural language flight queries into AST payloads."""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

try:  # pragma: no cover - exercised via integration environments
    from openai import AsyncOpenAI
    from openai import APIConnectionError, APITimeoutError, APIStatusError, OpenAIError
except Exception:  # pragma: no cover - fallback path for tests without OpenAI installed
    AsyncOpenAI = None  # type: ignore[assignment]
    APIConnectionError = APITimeoutError = APIStatusError = OpenAIError = Exception  # type: ignore[assignment]

from server_tailwind.core.resources import AppResources, get_resources
from server_tailwind.query.QueryAPIWrapper import QueryAPIWrapper


_logger = logging.getLogger("uvicorn.error").getChild(__name__)


SYSTEM_PROMPT = """You are FlightQueryAST, a deterministic semantic parser. Convert natural language requests about flights into a JSON object with exactly one top-level key: "query". The "query" value must be a valid AST node for the /flight_query_ast endpoint.

Output requirements:
- Output ONLY a single JSON object: { "query": { ... } } with no extra text.
- Do NOT include an "options" field and do NOT include "flight_ids" anywhere. The server will supply options.
- Use only supported node types and fields exactly as specified below.

Supported AST:
- Boolean:
  - { "type": "and", "children": [AST, ...] }
  - { "type": "or",  "children": [AST, ...] }
  - { "type": "not", "child": AST }  (you may also accept { "type": "not", "children": [AST] })
- Crossings:
  - { "type": "cross", "tv": TV_SELECTION, "time"?: TIME_WINDOW, "mode"?: "any" }
    TV_SELECTION:
      - "TVA" (string), or
      - ["TVA", "TVB"] (treated as anyOf), or
      - { "anyOf": ["TVA","TVB"] }, { "allOf": [...] }, { "noneOf": [...] }
- Sequences:
  - { "type": "sequence", "steps": [CROSS, CROSS, ...], "within"?: DURATION, "strict"?: boolean }
    where each step is a "cross" node. "within" represents maximum elapsed time from first to last step.
- Airports:
  - { "type": "origin", "airport": "LFPG" or ["LFPG","EGLL"] }
  - { "type": "destination", "airport": "LFPO" or ["LFPO","LFPG"] }
- Time windows on milestones:
  - { "type": "arrival_window", "clock"?: CLOCK or "bins"?: BINS, "method"?: "last_crossing" }
  - { "type": "takeoff_window", "clock"?: CLOCK or "bins"?: BINS }
- Capacity state:
  - { "type": "capacity_state", "tv": TV_SELECTION, "time": TIME_WINDOW,
      "condition": "overloaded" | "near_capacity" | "under_capacity",
      "threshold"?: { "occupancy_minus_capacity"?: number } }
- Duration between two events:
  - { "type": "duration_between", "from": CROSS, "to": CROSS,
      "op": "<" | "<=" | ">" | ">=", "value": DURATION }
- Counting crossings:
  - { "type": "count_crossings", "tv": TV_SELECTION, "time": TIME_WINDOW,
      "op": ">=" | "<=" | "==" | ">" | "<", "value": integer }
- Geographic region crossing:
  - { "type": "geo_region_cross", "region": { "tv_ids": [ ... ] | "bbox": [minLon,minLat,maxLon,maxLat] | "polygon": GeoJSON },
      "time"?: TIME_WINDOW, "mode"?: "any" }

Common specs:
- TIME_WINDOW: { "clock": CLOCK } | { "bins": BINS } | { "clock": { "from": "HH:MM[:SS]", "to": "HH:MM[:SS]" }, "inclusive"?: true }
  - Prefer "clock" when the user gives times like "between 11:15 and 11:45".
- CLOCK: { "from": "HH:MM[:SS]", "to": "HH:MM[:SS]" } or the shorthand { "from": "...", "to": "..." } inside "time".
- BINS: { "from": integer, "to": integer }
- DURATION: { "minutes": number } or { "bins": number }

Interpretation rules:
- “between T1 and T2” → TIME_WINDOW with { "clock": { "from": T1, "to": T2 } }.
- “then”, “followed by”, “after that” → "sequence".
- “within N minutes” in a sequence → { "within": { "minutes": N } }.
- “at least/exactly/at most N crossings” → "count_crossings" with the appropriate comparator.
- “over capacity”, “overloaded”, “under capacity”, “near capacity” → "capacity_state" with the matching "condition".
- If multiple TVs are mentioned with “or”, use { "tv": { "anyOf": [...] } }.
- If multiple TVs with “and” (must cross all), use { "tv": { "allOf": [...] } }.
- If a TV must be excluded, use { "tv": { "noneOf": [...] } } within a cross node or combine with "not".
- For arrivals “arriving at LFPO between T1–T2” use an "and" of:
  - { "type": "destination", "airport": "LFPO" }
  - { "type": "arrival_window", "clock": { "from": T1, "to": T2 }, "method": "last_crossing" }
- Do NOT invent traffic volume IDs or airport codes. Use only IDs present in the provided context. If an ID is not in context, produce a reasonable AST using only known items; omit unknown ones rather than fabricating.

Output format:
- A single JSON object with exactly one key: "query".
- Example (flights crossing TVA between 11:15–11:45):
  { "query": { "type": "cross", "tv": "TVA", "time": { "clock": { "from": "11:15", "to": "11:45" } } } }
"""


class NLPQueryParserError(Exception):
    """Exception raised when parsing fails for a known reason."""

    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.status_code = status_code


@dataclass
class NLPQueryParserResult:
    query: Dict[str, Any]
    ast: Dict[str, Any]
    llm: Dict[str, Any]


class NLPQueryParser:
    """Convert natural language prompts into Query API AST payloads via OpenAI."""

    MAX_TV_IDS = 200
    MAX_AIRPORTS = 100

    def __init__(
        self,
        resources: Optional[AppResources] = None,
        query_wrapper: Optional[QueryAPIWrapper] = None,
        client: Optional[Any] = None,
    ) -> None:
        self._resources = resources or get_resources()
        self._query_wrapper = query_wrapper
        self._client = client
        self._api_key = os.getenv("OPENAI_API_KEY")
        self._default_model = os.getenv("FLIGHT_QUERY_NLP_MODEL", "gpt-5")
        timeout_str = os.getenv("FLIGHT_QUERY_NLP_TIMEOUT_S", "20")
        try:
            self._timeout_s = float(timeout_str)
        except ValueError:
            self._timeout_s = 20.0
        self._context_cache: Optional[Dict[str, Any]] = None

    @property
    def resources(self) -> AppResources:
        return self._resources

    def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client
        if AsyncOpenAI is None:
            raise NLPQueryParserError(
                "OpenAI client library is not installed", status_code=500
            )
        if not self._api_key:
            raise NLPQueryParserError("OpenAI API key is not configured", status_code=500)
        self._client = AsyncOpenAI(api_key=self._api_key)
        return self._client

    def _build_context(self) -> Dict[str, Any]:
        flight_list = self._query_wrapper.flight_list if self._query_wrapper else self._resources.flight_list
        time_bin_minutes = int(getattr(flight_list, "time_bin_minutes", 0))
        bins_per_tv = int(getattr(flight_list, "num_time_bins_per_tv", 0))
        tv_ids = self._take_first_n(sorted(flight_list.tv_id_to_idx.keys()), self.MAX_TV_IDS)
        airports = self._collect_airports(flight_list)
        context: Dict[str, Any] = {
            "time_bin_minutes": time_bin_minutes,
            "bins_per_tv": bins_per_tv,
            "known_tv_ids": tv_ids,
        }
        if airports:
            context["known_airports"] = airports
        return context

    def _context_payload(self) -> Dict[str, Any]:
        if self._context_cache is None:
            self._context_cache = self._build_context()
        return self._context_cache

    @staticmethod
    def _take_first_n(items: Iterable[str], n: int) -> List[str]:
        out: List[str] = []
        for item in items:
            out.append(str(item))
            if len(out) >= n:
                break
        return out

    def _collect_airports(self, flight_list: Any) -> List[str]:
        metadata = getattr(flight_list, "flight_metadata", {})
        if not isinstance(metadata, dict):
            return []
        seen = set()
        airports: List[str] = []
        for entry in metadata.values():
            if not isinstance(entry, dict):
                continue
            for key in ("origin", "destination"):
                value = entry.get(key)
                if not isinstance(value, str):
                    continue
                code = value.strip().upper()
                if not code or code in seen:
                    continue
                seen.add(code)
                airports.append(code)
                if len(airports) >= self.MAX_AIRPORTS:
                    return airports
        return airports

    async def parse(self, prompt: str, *, model: Optional[str] = None, debug: bool = False) -> NLPQueryParserResult:
        if not isinstance(prompt, str) or not prompt.strip():
            raise NLPQueryParserError("'prompt' must be a non-empty string")
        resolved_model = (model or self._default_model or "").strip()
        if not resolved_model:
            raise NLPQueryParserError("No model configured for NLP parsing", status_code=500)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "prompt": prompt,
                        # "context": self._context_payload(), # currently context is disabled
                    },
                    separators=(",", ":"),
                ),
            },
        ]

        client = self._ensure_client()
        start = time.perf_counter()
        try:
            response = await client.chat.completions.create(
                model=resolved_model,
                messages=messages,
                response_format={"type": "json_object"},
                timeout=self._timeout_s,
                reasoning_effort="medium"
            )
        except (APITimeoutError, APIConnectionError) as exc:
            raise NLPQueryParserError("LLM parser request timed out", status_code=502) from exc
        except APIStatusError as exc:
            import traceback
            error_details = f"OpenAI error: {type(exc).__name__}: {str(exc)}\nTraceback:\n{traceback.format_exc()}"
            raise NLPQueryParserError(f"LLM parser request failed: {error_details}", status_code=502) from exc
        except OpenAIError as exc:  # pragma: no cover - defensive
            import traceback
            error_details = f"OpenAI error: {type(exc).__name__}: {str(exc)}\nTraceback:\n{traceback.format_exc()}"
            raise NLPQueryParserError(f"LLM parser request failed: {error_details}", status_code=502) from exc

        latency_ms = (time.perf_counter() - start) * 1000.0
        if not response.choices:
            raise NLPQueryParserError("LLM response missing choices", status_code=502)
        message = response.choices[0].message
        content = getattr(message, "content", None)
        if not content:
            raise NLPQueryParserError("LLM response did not include content", status_code=502)

        try:
            parsed = json.loads(content)
            _logger.info("Parsed query payload: %s", json.dumps(parsed, separators=(",", ":")))
        except json.JSONDecodeError as exc:
            raise NLPQueryParserError("LLM returned invalid JSON", status_code=400) from exc

        query = parsed.get("query") if isinstance(parsed, dict) else None
        if not isinstance(query, dict):
            raise NLPQueryParserError("LLM response missing 'query' object", status_code=400)

        usage = getattr(response, "usage", None)
        llm_meta: Dict[str, Any] = {
            "model": resolved_model,
            "parse_ms": latency_ms,
        }
        if usage is not None:
            prompt_tokens = getattr(usage, "prompt_tokens", None)
            completion_tokens = getattr(usage, "completion_tokens", None)
            if prompt_tokens is not None:
                llm_meta["prompt_tokens"] = prompt_tokens
            if completion_tokens is not None:
                llm_meta["completion_tokens"] = completion_tokens

        return NLPQueryParserResult(query=query, ast=parsed if debug else query, llm=llm_meta)


__all__ = [
    "NLPQueryParser",
    "NLPQueryParserError",
    "NLPQueryParserResult",
]
