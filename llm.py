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
from langgraph.prebuilt import create_react_agent

# --- 1. Define Strict Pydantic Schemas ---

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

def parse_filters_with_langgraph(user_input: str) -> AirbnbAgentResponse:
    # 1. Initialize the LLM
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    
    # 2. Define the tools
    tools = [resolve_place_id_tool, resolve_date_range_tool]
    
    # 3. Create the Agent
    # 'response_format' is the key: it forces the final node to return our Pydantic model
    agent = create_react_agent(
        llm, 
        tools=tools,
        prompt=(
            "You are a helpful Airbnb assistant. "
            "Use the tools provided to resolve locations and dates. "
            f"Today's date is {date.today().isoformat()}."
        ),
        response_format=AirbnbAgentResponse
    )

    # 4. Invoke and get the structured result
    # LangGraph handles the loop: LLM calls tool -> Tool runs -> LLM sees result -> Agent finishes
    result = agent.invoke({"messages": [("user", user_input)]})
    
    # The 'structured_response' key contains our AirbnbAgentResponse object
    return result["structured_response"]
