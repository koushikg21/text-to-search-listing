"""
LLM-backed input parser for turning free-form chat messages into Airbnb MCP tool arguments.

This module stays decoupled from Streamlit/UI so it can be imported and unit tested directly.
"""

from __future__ import annotations

import json
import os
import re
from datetime import date, datetime, timedelta
from typing import Any, TypedDict

import requests
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool


class AgentResult(TypedDict):
    filters: dict[str, Any]
    question: str | None
    tool_trace: list[str]


SEARCH_ALLOWED_FIELDS = [
    "location",
    "placeId",
    "checkin",
    "checkout",
    "adults",
    "children",
    "infants",
    "pets",
    "minPrice",
    "maxPrice",
    "cursor",
    "minRating",
]

LISTING_ALLOWED_FIELDS = [
    "id",
    "checkin",
    "checkout",
    "adults",
    "children",
    "infants",
    "pets",
]

SYSTEM_PROMPT = """You are an input parser agent for Airbnb MCP tools.
Given recent user chat messages, extract the best-guess tool arguments or ask a concise clarifying question when required info is missing.

Response format (JSON only):
{
  "filters": { ... },   // only allowed keys for the tool
  "question": null | "short clarifying question"
}

Rules:
- Allowed keys vary per tool (provided below). Do not invent keys.
- For dates, use YYYY-MM-DD. For numbers, use integers. If year not mentioned, take the most sensible year (e.g., next occurrence).
- Do not hallucinate values. If required fields are missing or ambiguous, leave filters empty and set a one-sentence question.
- If all required fields are present, set question to null.
- You can call resolve_place_id to turn a location description into a Google placeId.
- You can call resolve_date_range for fuzzy dates like "this weekend" or "next Friday" or unclear dates to get concrete check-in/out dates.
"""

def resolve_place_id(
    query: str,
    region: str | None = None,
    language: str | None = None,
) -> dict[str, Any]:
    """
    Resolve a location query to a Google place_id using Places Find Place From Text.
    """
    g_api_key = os.getenv("GOOGLE_PLACES_API_KEY")
    if not g_api_key:
        return {
            "place_id": None,
            "formatted_address": None,
            "name": None,
            "types": None,
            "confidence": "low",
            "note": "Missing GOOGLE_PLACES_API_KEY.",
        }

    params: dict[str, str] = {
        "input": query,
        "inputtype": "textquery",
        "fields": "place_id,formatted_address,name,types",
        "key": g_api_key,
    }
    if region:
        params["region"] = region
    if language:
        params["language"] = language
    try:
        
        response = requests.get(
            "https://maps.googleapis.com/maps/api/place/findplacefromtext/json",
            params=params,
            timeout=10,
        )
        data = response.json()
    except Exception as exc:
        return {
            "place_id": None,
            "formatted_address": None,
            "name": None,
            "types": None,
            "confidence": "low",
            "note": f"Places API error: {exc}",
        }

    candidates = data.get("candidates", []) or []
    if not candidates:
        return {
            "place_id": None,
            "formatted_address": None,
            "name": None,
            "types": None,
            "confidence": "low",
            "note": "No candidates returned.",
        }

    best = candidates[0]
    return {
        "place_id": best.get("place_id"),
        "formatted_address": best.get("formatted_address"),
        "name": best.get("name"),
        "types": best.get("types"),
        "confidence": "medium",
        "note": "Top candidate from Places API.",
    }


def resolve_date_range(
    text: str,
    reference_date: str,
    timezone: str,
    default_nights: int = 2,
) -> dict[str, str | None]:
    """
    Resolve fuzzy or partial dates into concrete check-in/out strings.
    """
    today = date.fromisoformat(reference_date)
    low = text.strip().lower()
    note = ""

    def next_weekday(start: date, weekday: int, strict_next: bool) -> date:
        days_ahead = (weekday - start.weekday()) % 7
        if strict_next and days_ahead == 0:
            days_ahead = 7
        return start + timedelta(days=days_ahead)

    iso_range = re.search(r"(\d{4}-\d{2}-\d{2})\s*(?:to|-|–|—)\s*(\d{4}-\d{2}-\d{2})", low)
    if iso_range:
        return {
            "checkin": iso_range.group(1),
            "checkout": iso_range.group(2),
            "confidence": "high",
            "note": "Parsed explicit ISO date range.",
        }

    iso_single = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", low)
    if iso_single:
        start = date.fromisoformat(iso_single.group(1))
        end = start + timedelta(days=default_nights)
        return {
            "checkin": start.isoformat(),
            "checkout": end.isoformat(),
            "confidence": "medium",
            "note": "Parsed explicit date with default stay length.",
        }

    month_map = {
        "jan": 1, "january": 1,
        "feb": 2, "february": 2,
        "mar": 3, "march": 3,
        "apr": 4, "april": 4,
        "may": 5,
        "jun": 6, "june": 6,
        "jul": 7, "july": 7,
        "aug": 8, "august": 8,
        "sep": 9, "sept": 9, "september": 9,
        "oct": 10, "october": 10,
        "nov": 11, "november": 11,
        "dec": 12, "december": 12,
    }
    month_range = re.search(
        r"\b([a-z]+)\s+(\d{1,2})(?:st|nd|rd|th)?\s*(?:to|-|–|—)\s*(\d{1,2})(?:st|nd|rd|th)?\b",
        low,
    )
    month_range_full = re.search(
        r"\b(?:between|from)?\s*([a-z]+)\s+(\d{1,2})(?:st|nd|rd|th)?\s*(?:to|-|–|—)\s*([a-z]+)?\s*(\d{1,2})(?:st|nd|rd|th)?\b",
        low,
    )
    match = month_range_full or month_range
    if match:
        start_month_name = match.group(1)
        end_month_name = match.group(3) if month_range_full else None
        if start_month_name in month_map and (end_month_name is None or end_month_name in month_map):
            start_month = month_map[start_month_name]
            end_month = month_map[end_month_name] if end_month_name else start_month
            day_start = int(match.group(2))
            day_end = int(match.group(4) if month_range_full else match.group(3))
            year = today.year
            start = date(year, start_month, day_start)
            if start < today:
                start = date(year + 1, start_month, day_start)
                year += 1
            end = date(year, end_month, day_end)
            if end < start:
                end = start + timedelta(days=default_nights)
                note = "End date adjusted with default stay length."
            return {
                "checkin": start.isoformat(),
                "checkout": end.isoformat(),
                "confidence": "medium",
                "note": note or "Parsed month/day range.",
            }

    if "this weekend" in low:
        friday = next_weekday(today, 4, strict_next=False)
        sunday = friday + timedelta(days=2)
        return {
            "checkin": friday.isoformat(),
            "checkout": sunday.isoformat(),
            "confidence": "medium",
            "note": f"Assumed Fri-Sun for weekend in {timezone}.",
        }

    weekday_map = {
        "monday": 0, "mon": 0,
        "tuesday": 1, "tue": 1, "tues": 1,
        "wednesday": 2, "wed": 2,
        "thursday": 3, "thu": 3, "thurs": 3,
        "friday": 4, "fri": 4,
        "saturday": 5, "sat": 5,
        "sunday": 6, "sun": 6,
    }
    next_match = re.search(r"\bnext\s+([a-z]+)\b", low)
    if next_match and next_match.group(1) in weekday_map:
        weekday = weekday_map[next_match.group(1)]
        start = next_weekday(today, weekday, strict_next=True)
        end = start + timedelta(days=default_nights)
        return {
            "checkin": start.isoformat(),
            "checkout": end.isoformat(),
            "confidence": "medium",
            "note": "Resolved 'next' weekday with default stay length.",
        }
    this_match = re.search(r"\bthis\s+([a-z]+)\b", low)
    if this_match and this_match.group(1) in weekday_map:
        weekday = weekday_map[this_match.group(1)]
        start = next_weekday(today, weekday, strict_next=False)
        end = start + timedelta(days=default_nights)
        return {
            "checkin": start.isoformat(),
            "checkout": end.isoformat(),
            "confidence": "medium",
            "note": "Resolved 'this' weekday with default stay length.",
        }

    return {
        "checkin": None,
        "checkout": None,
        "confidence": "low",
        "note": "Could not resolve dates.",
    }


@tool
def resolve_place_id_tool(
    query: str,
    region: str | None = None,
    language: str | None = None,
) -> dict[str, Any]:
    """
    Resolve a location query into a Google placeId.
    """
    return resolve_place_id(query, region=region, language=language)


@tool
def resolve_date_range_tool(
    text: str,
    reference_date: str,
    timezone: str,
    default_nights: int = 2,
) -> dict[str, str | None]:
    """
    Resolve fuzzy date phrases into concrete check-in/out dates.
    """
    return resolve_date_range(text, reference_date, timezone, default_nights)

import streamlit as st
def parse_filters_with_llm(
    text: str,
    tool_name: str,
    *,
    model: str | None = None,
    api_key: str | None = None,
) -> AgentResult:
    """
    Call an OpenAI-compatible model to extract tool filters from a single input string.
    Returns filters + optional clarifying question. Falls back to empty result on error.
    """
    if not text.strip():
        return {"filters": {}, "question": None, "tool_trace": []}
    api_key = api_key or os.getenv("OPENAI_API_KEY")
    model_name = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    allowed = SEARCH_ALLOWED_FIELDS if tool_name == "airbnb_search" else LISTING_ALLOWED_FIELDS
    required = ["location"] if tool_name == "airbnb_search" else ["id"]
    tool_hint = (
        "Tool: airbnb_search. Allowed keys: "
        + ", ".join(allowed)
        + f". Required: {', '.join(required)}."
    )
    if tool_name == "airbnb_listing_details":
        tool_hint = (
            "Tool: airbnb_listing_details. Allowed keys: "
            + ", ".join(allowed)
            + f". Required: {', '.join(required)}."
        )

    llm = ChatOpenAI(
        model=model_name,
        api_key=api_key,
        temperature=0,
    )
    llm_with_tools = llm.bind_tools([resolve_place_id_tool, resolve_date_range_tool])

    timezone_name = datetime.now().astimezone().tzname() or "UTC"
    today_str = date.today().isoformat()
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(
            content=tool_hint
            + f"\nReference date: {today_str}\nTimezone: {timezone_name}\n\nInput:\n"
            + text
        ),
    ]

    tool_trace: list[str] = []
    response = llm_with_tools.invoke(messages)
    if response.tool_calls:
        tool_messages = []
        for call in response.tool_calls:
            name = call.get("name")
            args = call.get("args", {}) or {}
            if name == "resolve_place_id_tool":
                resolved = resolve_place_id(
                    str(args.get("query", "")).strip(),
                    region=args.get("region"),
                    language=args.get("language"),
                )
                tool_trace.append(
                    "resolve_place_id_tool -> "
                    f"{resolved.get('place_id')} ({resolved.get('formatted_address')})"
                )
                tool_messages.append(
                    ToolMessage(
                        content=json.dumps(resolved),
                        tool_call_id=call.get("id"),
                    )
                )
            elif name == "resolve_date_range_tool":
                resolved = resolve_date_range(
                    str(args.get("text", "")).strip(),
                    str(args.get("reference_date", today_str)).strip(),
                    str(args.get("timezone", timezone_name)).strip(),
                    int(args.get("default_nights", 2)),
                )
                tool_trace.append(
                    "resolve_date_range_tool -> "
                    f"{resolved.get('checkin')} to {resolved.get('checkout')} "
                    f"({resolved.get('confidence')})"
                )
                tool_messages.append(
                    ToolMessage(
                        content=json.dumps(resolved),
                        tool_call_id=call.get("id"),
                    )
                )
        if tool_messages:
            messages = messages + [response] + tool_messages
            response = llm.invoke(messages)

    raw_text = response.content or "{}"

    try:
        parsed = json.loads(raw_text)
    except Exception:
        return {"filters": {}, "question": None, "tool_trace": tool_trace}

    filters = parsed.get("filters") or {}
    question = parsed.get("question")

    # Filter to allowed keys and drop empty/None values
    cleaned_filters = {
        key: value
        for key, value in filters.items()
        if key in allowed and value not in (None, "", [])
    }

    if required and not all(k in cleaned_filters for k in required):
        # If required info still missing, prompt with a question if provided.
        return {
            "filters": {},
            "question": question or f"Can you share the {', '.join(required)}?",
            "tool_trace": tool_trace,
        }

    return {"filters": cleaned_filters, "question": question, "tool_trace": tool_trace}
