from __future__ import annotations
import os
from typing import Any, Optional, Literal
from datetime import date, datetime

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from llm import resolve_place_id, resolve_date_range
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

# --- 1. Define Strict Pydantic Schemas ---

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

class AirbnbFilters(BaseModel):
    """Strict search filters for the Airbnb API."""
    location: Optional[str] = Field(None, description="The destination or Google Place ID.")
    placeId: Optional[str] = Field(None, description="The resolved Google Place ID.")
    checkin: Optional[str] = Field(None, description="Check-in date in YYYY-MM-DD.")
    checkout: Optional[str] = Field(None, description="Check-out date in YYYY-MM-DD.")
    adults: int = Field(1, description="Number of adults.")
    children: int = Field(0, description="Number of children.")
    infants: int = Field(0, description="Number of infants.")
    pets: int = Field(0, description="Number of pets.")
    minPrice: Optional[int] = Field(None, description="Minimum price per night.")
    maxPrice: Optional[int] = Field(None, description="Maximum price per night.")
    minRating: Optional[float] = Field(None, description="Minimum star rating (0-5).")

class AirbnbAgentResponse(BaseModel):
    """The final structured output returned to the UI."""
    filters: AirbnbFilters = Field(description="The extracted and resolved search filters.")
    question: Optional[str] = Field(None, description="A question to ask the user if info is missing.")
    explanation: str = Field(description="A brief summary of what was found or resolved.")

# --- 2. Wrap your existing logic as Tools ---

@tool
def resolve_place_id_tool(query: str, region: str, language: str) -> dict[str, Any]:
    """Resolve a fuzzy location (e.g. 'near Eiffel Tower') into a Google place_id."""
    # ... (Your existing resolve_place_id logic here) ...
    return resolve_place_id(query, region=region, language=language)


@tool
def resolve_date_range_tool(text: str) -> dict:
    """
    Deterministic date parser. Call this for any date phrases like 'next week' or 'this weekend'.
    """
    # Just call your existing function and pass today's date automatically
    return resolve_date_range(
        text=text, 
        reference_date=date.today().isoformat(), 
        timezone="UTC"
    )
# --- 3. The Modern Parser Function ---

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
    result = agent.invoke({
    "messages": [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_input),
    ]
})
    
    # The 'structured_response' key contains our AirbnbAgentResponse object
    return result["structured_response"]

# --- 4. Example Usage ---
if __name__ == "__main__":
    user_query = "A place for me and my wife in north west arkansas for under 250$ between Jan 1 to Jan 4 with 4.9 and above rating"
    response = parse_filters_with_langgraph(user_query)
    
    print(f"Filters: {response.filters.model_dump_json(indent=2)}")
    print(f"Explanation: {response.explanation}")
