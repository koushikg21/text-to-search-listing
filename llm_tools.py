from __future__ import annotations

from datetime import date
from typing import Any

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent

from test import AirbnbAgentResponse, resolve_date_range_tool, resolve_place_id_tool


def _extract_tool_trace(result: dict) -> list[dict[str, Any]]:
    messages = result.get("messages") or []
    trace: list[dict[str, Any]] = []
    by_id: dict[str, dict[str, Any]] = {}
    for msg in messages:
        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls:
            for call in tool_calls:
                call_id = call.get("id") or call.get("tool_call_id") or f"call_{len(trace)}"
                entry = {
                    "id": call_id,
                    "name": call.get("name"),
                    "input": call.get("args"),
                    "output": None,
                }
                by_id[call_id] = entry
                trace.append(entry)
        if getattr(msg, "type", None) == "tool" or msg.__class__.__name__ == "ToolMessage":
            call_id = getattr(msg, "tool_call_id", None)
            content = getattr(msg, "content", None)
            name = getattr(msg, "name", None)
            if call_id in by_id:
                by_id[call_id]["output"] = content
                if name and not by_id[call_id].get("name"):
                    by_id[call_id]["name"] = name
            else:
                trace.append(
                    {"id": call_id, "name": name, "input": None, "output": content}
                )
    return trace


def parse_filters_with_langgraph_with_trace(
    user_input: str,
) -> tuple[AirbnbAgentResponse, list[dict[str, Any]]]:
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    tools = [resolve_place_id_tool, resolve_date_range_tool]
    agent = create_react_agent(
        llm,
        tools=tools,
        prompt=(
            "You are a helpful Airbnb assistant. "
            "Use the tools provided to resolve locations and dates. "
            f"Today's date is {date.today().isoformat()}."
        ),
        response_format=AirbnbAgentResponse,
    )
    result = agent.invoke({"messages": [("user", user_input)]})
    return result["structured_response"], _extract_tool_trace(result)


def parse_filters_with_llm_no_tools(user_input: str) -> AirbnbAgentResponse:
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    structured = llm.with_structured_output(AirbnbAgentResponse)
    messages = [
        SystemMessage(
            content=(
                "You are a helpful Airbnb assistant. "
                "Extract structured filters from the user's request. "
                "If info is missing, assume you don't need that."
                "If dates are mentioned, use YYYY-MM-DD format. And if year not mentioned, assume current year or next year."
            )
        ),
        HumanMessage(content=user_input),
    ]
    return structured.invoke(messages)
